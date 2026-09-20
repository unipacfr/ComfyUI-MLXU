"""Conditioning builders for the MiniMax H3 i2v (fl2va) and ref2va nodes.

Ports the geometry and reference-preparation logic of ComfyUI's
`comfy_extras/nodes_minimax_h3.py` (the runtime this project mirrors):
canvas adaptation, reference sizing, video-reference preparation, and the
encoding of keyframes/references through the ComfyUI `VAE` bridge (whose
`encode` already returns normalized latents, posterior mean).
"""

from __future__ import annotations

import math

import mlx.core as mx
import torch

from .native.minimax_h3.condition import KeyframeCond, RefBlock

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
# The fused SwiGLU projection [1, S, 2*14336] crosses i32::MAX elements above this many rows
# (mlx-gen-minimax-h3 cost.rs); matmul was measured exact on both sides there, so this only warns.
ROW_WARN_THRESHOLD = 74_898


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32
    (`nodes_minimax_h3.adapt_canvas`)."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    """`[B, H, W, C]` -> `[B, height, width, 3]` with ComfyUI's lanczos
    (`crop`: "disabled" = plain stretch, "center" = aspect-preserving cover)."""
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _round32(value: float) -> int:
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def ref_image_size(img_h: int, img_w: int, gen_width: int, gen_height: int, mode: str) -> tuple[int, int]:
    """Target `(tw, th)` of a reference image: "match" scales it (down only,
    keeping the aspect) to the generation's pixel area; "max" uses the
    reference pipeline's 2048 short edge. Multiples of 32."""
    if mode == "match":
        scale = min(1.0, math.sqrt((gen_width * gen_height) / (img_w * img_h)))
    elif mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(img_w, img_h))
    else:
        raise ValueError(f"ASDX MiniMax H3: ref_image_size mode must be 'match' or 'max', got {mode!r}")
    return _round32(img_w * scale), _round32(img_h * scale)


def prepare_ref_video(frames: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, int, int]:
    """Resize a reference video onto its adapted canvas, cap it at the target's
    frame count and crop it to the model's 17k+5 clip grid. Returns
    `(frames, canvas_w, canvas_h)`."""
    vh, vw = frames.shape[1], frames.shape[2]
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:  # never upscale a small reference
        cw, ch = _round32(vw), _round32(vh)
    frames = resize_image(frames, cw, ch, "disabled")
    if frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = frames.shape[0]
    if n < 5:
        raise ValueError("ASDX MiniMax H3: reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    return frames[:n], cw, ch


def qwen_video_frames(frames: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """The text encoder sees a reference video at 2 fps with timestamps."""
    idx = list(range(0, frames.shape[0], FPS // 2))
    return frames[idx], [i / 2.0 for i in range(len(idx))]


def estimate_packed_rows(
    width: int,
    height: int,
    latent_t: int,
    audio_t: int,
    keyframe_frames: int = 0,
    ref_image_sizes: tuple[tuple[int, int], ...] = (),
    ref_videos: tuple[tuple[int, int, int, int], ...] = (),
    ref_audio_t: int = 0,
    keyframe_audio_t: int = 0,
) -> int:
    """Rows of the packed DiT sequence EXCLUDING the text rows: target video and
    audio, keyframe rows (one latent frame each), reference images `(tw, th)`,
    reference videos `(latent_t, cw, ch, audio_t)`, standalone reference audio and the
    audio latent length carried by keyframes (`2 * keyframe_audio_t` cond_audio rows)."""
    def frame_rows(w: int, h: int) -> int:
        return (h // 16 // 2) * (w // 16 // 2)

    rows = latent_t * frame_rows(width, height) + 2 * audio_t
    rows += keyframe_frames * frame_rows(width, height)
    rows += sum(frame_rows(tw, th) for tw, th in ref_image_sizes)
    rows += sum(lt * frame_rows(cw, ch) + 2 * at for lt, cw, ch, at in ref_videos)
    rows += 2 * ref_audio_t
    rows += 2 * keyframe_audio_t
    return rows


def to_mx(t: torch.Tensor) -> mx.array:
    """Torch tensor to a float32 MLX array (via CPU)."""
    return mx.array(t.detach().to("cpu", torch.float32).numpy())


def encode_video_latent(vae, frames: torch.Tensor) -> mx.array:
    """`vae.encode` of `[T, H, W, 3]` frames -> MLX latent `[1, 24, t, H/16, W/16]`
    (already normalized by the VAE). Comfy's own OOM -> tiled retry never fires on MPS (plain
    `RuntimeError`), so the two MPS limits a tiled encode fixes are retried tiled here, as in
    `ASDX_VAEEncode._fallback_encode`; any other error propagates unchanged.

    Caveat (verified by reading comfy/sd.py and comfy/ldm/minimax/vae.py): the real MiniMax H3 video
    VAE sets `handles_tiling`, and its `encode_tiled` is just `encode` (it already chunks temporally
    with `clip_length`). For that VAE this retry re-runs an equivalent encode after a cache flush, so
    it does NOT lower the memory peak or avoid an INT_MAX tensor: if a real MPS limit is hit, reduce
    the canvas or the reference length instead. The retry stays for consistency with the other VAE
    paths and for VAEs whose `encode_tiled` really tiles."""
    from .vae import _needs_tiled_retry

    do_tile = False
    try:
        z = vae.encode(frames)
    except Exception as e:
        if not _needs_tiled_retry(e):
            raise
        print(f"[ASDX] MiniMax H3 VAE encode: MPS limit hit ({e}), retrying with tiled encode.")
        do_tile = True
    if do_tile:
        import comfy.model_management

        comfy.model_management.soft_empty_cache()
        z = vae.encode_tiled(frames)
    if z.ndim != 5 or z.shape[1] != 24:
        raise ValueError(
            f"ASDX MiniMax H3: the video VAE returned a latent of shape {tuple(z.shape)}, expected [1, 24, T, h, w]"
        )
    return to_mx(z)


def encode_audio_latent(audio_vae, audio: dict) -> tuple[mx.array, int]:
    """Encode an AUDIO input through the audio VAE (`nodes_minimax_h3._encode_ref_audio`):
    resample to the VAE's rate, first batch item, `[1, C, L]` -> `[1, L, C]`."""
    waveform, sr = audio["waveform"], audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    if z.ndim != 4 or z.shape[2] != 2:
        raise ValueError(
            f"ASDX MiniMax H3: the audio VAE returned a latent of shape {tuple(z.shape)}, expected [1, 32, 2, T] "
            f"(2 = stereo: audio references must be stereo waveforms)"
        )
    return to_mx(z), int(z.shape[-1])


def build_keyframes(
    vae, width: int, height: int, frame_count: int, first_frame, last_frame
) -> tuple[list[torch.Tensor], list[KeyframeCond]]:
    """fl2va anchors. The first frame is the geometry anchor (plain stretch to the canvas);
    the last frame is a follower (aspect-preserving cover crop) at the final pixel frame."""
    images: list[torch.Tensor] = []
    keyframes: list[KeyframeCond] = []
    for frame, crop, index in ((first_frame, "disabled", 0), (last_frame, "center", frame_count - 1)):
        if frame is None:
            continue
        image = resize_image(frame[:1], width, height, crop)
        images.append(image)
        keyframes.append(KeyframeCond(resolved_frame_index=index, latent=encode_video_latent(vae, image)))
    return images, keyframes


def _require_vision_tower(text_encoder: dict) -> None:
    """Images reach the text encoder through the vision tower: fail before any VAE encode."""
    if text_encoder.get("vision_tower") is None:
        raise ValueError(
            "ASDX MiniMax H3: images or videos were given but no vision tower is loaded "
            "(enable load_vision on the text encoder loader)"
        )


def _check_cap(name: str, count: int, cap: int) -> None:
    if count > cap:
        raise ValueError(f"ASDX MiniMax H3: at most {cap} {name} references are supported, got {count}")


def build_references(
    vae,
    audio_vae,
    width: int,
    height: int,
    frame_count: int,
    ref_image_size_mode: str,
    ref_images: dict,
    ref_videos: dict,
    ref_video_audios: dict,
    ref_audios: dict,
) -> tuple[list[dict], list[RefBlock]]:
    """ref2va references in request order: images, then videos (a video's soundtrack label
    right before it), then standalone audios. Returns `(ref_items for the tokenizer,
    refs for the DiT)`; without a VAE only `ref_items` (text conditioning) is produced."""
    images = {k: v for k, v in (ref_images or {}).items() if v is not None}
    videos = {k: v for k, v in (ref_videos or {}).items() if v is not None}
    audios = {k: v for k, v in (ref_audios or {}).items() if v is not None}
    _check_cap("image", len(images), MAX_REF_IMAGES)
    _check_cap("video", len(videos), MAX_REF_VIDEOS)
    _check_cap("audio", len(audios), MAX_REF_AUDIOS)

    ref_items: list[dict] = []
    refs: list[RefBlock] = []

    for img in images.values():
        tw, th = ref_image_size(img.shape[1], img.shape[2], width, height, ref_image_size_mode)
        resized = resize_image(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        if vae is not None:
            z = encode_video_latent(vae, resized)
            refs.append(RefBlock(kind="image", latent=z, latent_t=int(z.shape[2]), latent_h=th // 16, latent_w=tw // 16))

    soundtracks = ref_video_audios or {}
    for name, video in videos.items():
        soundtrack = soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        frames, cw, ch = prepare_ref_video(video, frame_count)
        if soundtrack is not None:
            ref_items.append({"type": "audio"})  # its <Audio j> label comes before <Video k>
        sampled, stamps = qwen_video_frames(frames)
        ref_items.append({"type": "video", "data": sampled, "timestamps": stamps})
        if vae is None:
            continue
        z = encode_video_latent(vae, frames)
        audio_latent, audio_t = None, 0
        if soundtrack is not None and audio_vae is not None:
            audio_latent, audio_t = encode_audio_latent(audio_vae, soundtrack)
        refs.append(
            RefBlock(
                kind="video_audio" if audio_t else "video",
                latent=z,
                audio_latent=audio_latent,
                latent_t=int(z.shape[2]),
                latent_h=ch // 16,
                latent_w=cw // 16,
                ref_audio_t=audio_t,
            )
        )

    for audio in audios.values():
        ref_items.append({"type": "audio"})
        if audio_vae is not None:
            audio_latent, audio_t = encode_audio_latent(audio_vae, audio)
            refs.append(RefBlock(kind="audio", audio_latent=audio_latent, ref_audio_t=audio_t))

    return ref_items, refs


def empty_av_latents(width: int, height: int, length: int) -> tuple[dict, dict, int, int, int]:
    """Zero video/audio latents on the model's 17k+5 grid (the sampler draws its own noise).
    Returns `(video_latent, audio_latent, frame_count, latent_t, audio_t)`."""
    from .minimax_h3_nodes import _temporal_shape

    frame_count, latent_t, audio_t = _temporal_shape(length)
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16])
    audio = torch.zeros([1, 32, 2, audio_t])
    return {"samples": video}, {"samples": audio}, frame_count, latent_t, audio_t


def _report_rows(text_rows: int, rows_without_text: int) -> None:
    total = text_rows + rows_without_text
    print(f"[ASDX] MiniMax H3 packed sequence: {total} rows ({text_rows} text + {rows_without_text} video/audio/conditions)")
    if total > ROW_WARN_THRESHOLD:
        print(
            f"[ASDX] MiniMax H3 WARNING: {total} rows exceeds {ROW_WARN_THRESHOLD}: the packed sequence is long "
            f"(memory and time grow with it; reference tokens ride through all 50 blocks at every step). "
            f"Consider ref_image_size='match', fewer references or a shorter/smaller generation."
        )


def build_i2v_conditioning(
    text_encoder: dict, vae, prompt: str, width: int, height: int, length: int, first_frame, last_frame
) -> tuple[dict, dict, dict]:
    """t2v / fl2va: prompt (+ optional first/last frame) -> (conditioning, video_latent, audio_latent)."""
    from .minimax_h3_nodes import encode_minimax_h3_prompt

    if (first_frame is not None or last_frame is not None) and vae is None:
        raise ValueError("ASDX MiniMax H3: first_frame/last_frame need the vae input to be encoded")
    if first_frame is not None or last_frame is not None:
        _require_vision_tower(text_encoder)
    video_latent, audio_latent, frame_count, latent_t, audio_t = empty_av_latents(width, height, length)
    images, keyframes = build_keyframes(vae, width, height, frame_count, first_frame, last_frame)
    conditioning = encode_minimax_h3_prompt(text_encoder, prompt, images=images)
    if keyframes:
        conditioning["keyframes"] = keyframes
    rows = estimate_packed_rows(width, height, latent_t, audio_t, keyframe_frames=len(keyframes))
    _report_rows(int(conditioning["hidden_states"].shape[0]), rows)
    return conditioning, video_latent, audio_latent


def build_reference_conditioning(
    text_encoder: dict,
    vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size_mode: str,
    ref_images: dict,
    ref_videos: dict,
    ref_video_audios: dict,
    ref_audios: dict,
) -> tuple[dict, dict, dict]:
    """ref2va: prompt + references -> (conditioning, video_latent, audio_latent). The row estimate
    is derived from the built `RefBlock`s, whose latent dims are what the packed layout validates."""
    from .minimax_h3_nodes import encode_minimax_h3_prompt

    if any(v is not None for v in (ref_images or {}).values()) or any(v is not None for v in (ref_videos or {}).values()):
        _require_vision_tower(text_encoder)
    video_latent, audio_latent, frame_count, latent_t, audio_t = empty_av_latents(width, height, length)
    ref_items, refs = build_references(
        vae, audio_vae, width, height, frame_count, ref_image_size_mode, ref_images, ref_videos, ref_video_audios, ref_audios
    )
    if (ref_images or ref_videos or ref_audios) and vae is None:
        print("[ASDX] MiniMax H3: no vae connected, the references only condition the text encoder")
    conditioning = encode_minimax_h3_prompt(text_encoder, prompt, ref_items=ref_items or None)
    if refs:
        conditioning["refs"] = refs
    image_sizes = tuple((r.latent_w * 16, r.latent_h * 16) for r in refs if r.kind == "image")
    videos = tuple(
        (r.latent_t, r.latent_w * 16, r.latent_h * 16, r.ref_audio_t) for r in refs if r.kind in ("video", "video_audio")
    )
    standalone_audio = sum(r.ref_audio_t for r in refs if r.kind == "audio")
    rows = estimate_packed_rows(
        width, height, latent_t, audio_t, ref_image_sizes=image_sizes, ref_videos=videos, ref_audio_t=standalone_audio
    )
    _report_rows(int(conditioning["hidden_states"].shape[0]), rows)
    return conditioning, video_latent, audio_latent
