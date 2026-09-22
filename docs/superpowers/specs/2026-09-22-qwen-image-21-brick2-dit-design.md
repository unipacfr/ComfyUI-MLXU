# Qwen Image 2.1 — brique 2 : DiT (design)

Date: 2026-09-22
Statut: approuvé (brainstorming), en attente du plan d'implémentation
Complète: `docs/superpowers/specs/2026-09-22-qwen-image-21-design.md` (brique 1, text encoder,
déjà fusionnée sur master)

## Contexte

La brique 1 a livré le text encoder Qwen3-VL-8B en MLX (`native/qwen_image21/text_encoder*.py`,
`rope.py`). Cette brique porte le DiT lui-même : `QwenImage21Transformer2DModel`
(`comfy/ldm/qwen_image21/model.py`, 353 lignes), 32 blocks single-stream, mixed-granularity
attention (segments texte causaux + image pleine attention), modulation "scale only, no shift"
partagée par tous les blocks.

## Périmètre — brique 2

**Dans le périmètre** :
- Le DiT complet pour T2I : `ZeroCenteredRMSNorm`, `TextProjection`, `TimestepProjEmbeddings`,
  `SwiGLUFeedForward`, `Attention`, `QwenImage21TransformerBlock`, `LastLayer`,
  `QwenImage21Transformer2DModel`
- RoPE FLUX-style (`pe_embedder`/`EmbedND`/`apply_rope1`), axes `(16, 56, 56)`
- Les deux formats DiT : `qwen_image_2.1_bf16.safetensors` et `qwen_image_2.1_Q8.gguf`
- Chargement checkpoint avec logging `matched N/M` (convention `verify-checkpoint`)

**Hors périmètre (différé)** :
- **Cache KV préfixe inter-étapes** (`PoseBranchCache`, stockage quantisé int8/int4 GPU/CPU) —
  décision explicite : recalcul systématique via `block_causal_attention`, mathématiquement
  identique à la référence avec cache, juste plus lent. Aucune autre famille du projet n'a cette
  optimisation ; perf en suivi si mesurée nécessaire.
- **`ref_latents`/`image_slots`** (jusqu'à 10 images de référence, édition) — `build_sequence`
  simplifié à texte + image cible uniquement, pas de boucle de références.
- **`patches`/`transformer_options` hooks** (points d'extension ComfyUI — LoRA-style patches,
  hooks de debug) — non utilisés ailleurs dans ce projet pour les DiT portés.
- Intégration nœuds/loader/bridge (brique 3, comme prévu).

## Architecture

`apple_silicon_nodes/native/qwen_image21/` (complète le package de la brique 1) :

- `config.py` — `in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128,
  num_attention_heads=32, context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16,56,56), eps=1e-6`
  (defaults de `QwenImage21Transformer2DModel.__init__`), détection depuis le header checkpoint
  (bf16 direct, GGUF via `native/gguf/`)
- `dit_rope.py` — `embed_nd`, `apply_rope`, `timestep_embedding` : portage direct de
  `native/flux2/model.py`'s `embed_nd`/`apply_rope`/`timestep_embedding` (même math FLUX-style —
  la référence ComfyUI de Qwen Image 2.1 importe littéralement `comfy.ldm.flux.layers.EmbedND`
  et `comfy.ldm.flux.math.apply_rope1`/`rope`), dupliqué par famille (convention déjà établie,
  voir Krea2's propre `rope.py`) avec `axes_dim=(16,56,56)` au lieu des dims Flux2
- `model.py` — port MLX complet :
  - `ZeroCenteredRMSNorm` : poids stocké = scale-1, `rms_norm(x.float(), weight+1.0, eps)`
  - `TextProjection` : `ZeroCenteredRMSNorm` -> Linear -> GELU(tanh) -> Linear
  - `TimestepProjEmbeddings` : `timestep_embedding(t, 256)` -> MLP (SiLU) -> `inner_dim`
  - `SwiGLUFeedForward` : fused `gate_up` Linear (2x hidden_dim) -> SiLU-gate -> `out`
  - `Attention` : `to_q/to_k/to_v` -> RMSNorm par tête (`norm_q`/`norm_k`) -> RoPE -> attention
    scaled-dot-product -> `to_out`. Porte la branche `in_training` de la référence (RMSNorm
    puis rope explicites), PAS la branche fused-kernel (`comfy.quant_ops.ck.rms_rope`) —
    mathématiquement identique, ComfyUI bascule sur la branche training justement pour éviter
    ces kernels compilés.
  - `QwenImage21TransformerBlock` : modulation "scale only, no shift" (`LayerNorm
    elementwise_affine=False` puis `x * (1+scale)`), gate résiduel séparé prefix/target
  - `LastLayer` : idem, scale seul depuis le timestep embedding
  - `QwenImage21Transformer2DModel` : `pe_embedder`, `time_text_embed`, `txt_in`, `img_in`,
    `modulation` (Sequential SiLU+Linear partagé par tous les blocks), `transformer_blocks`,
    `norm_out`, `proj_out`. `build_sequence` simplifié (pas de refs) : concatène texte (segment
    causal) + image cible (segment plein), positions RoPE `(t, h, w)` par token.
- `weight_map.py` — bf16 direct (`mx.load`, même pattern que la brique 1) + GGUF Q8 via
  `native/gguf/reader.py`/`dequant.py`, logging `matched N/M`

## Tests / vérification

Même protocole que la brique 1 :
1. `verify-checkpoint` skill (py_compile, forward pass réduit à poids aléatoires, chargement
   matched N/M sur `qwen_image_2.1_bf16.safetensors` réel, std chargé vs random-init)
2. Répéter le chargement matched N/M sur `qwen_image_2.1_Q8.gguf`
3. Agent `weight-map-reviewer` sur `weight_map.py`
4. Agent `comfy-reference-diff` — diff contre `comfy/ldm/qwen_image21/model.py`
5. Test manuel end-to-end sur ComfyUI réel différé à la brique 3 (pas de nœud/loader câblé
   avant cette brique)

## Risques identifiés

- **`build_sequence` simplifié** : la référence a une logique de bornes/segments assez dense
  (boucle sur refs + image cible avec accumulation de position). Même réduite au cas T2I seul
  (une seule paire texte+image), porter la logique de segments causaux/positions RoPE exactement
  doit être vérifié contre la référence pas-à-pas, pas halluciné depuis la lecture.
- **`ZeroCenteredRMSNorm`** : convention poids = scale-1, différente de la RMSNorm standard déjà
  utilisée pour le text encoder (brique 1) — vérifier qu'aucun code partagé n'assume par erreur
  la convention standard.
- **Précision fp16** : le bloc a un clip explicite `x.clip(-65504, 65504)` en fp16 — à porter
  (MLX n'a pas d'overflow automatique identique, mais le clip reste pertinent pour la stabilité).
