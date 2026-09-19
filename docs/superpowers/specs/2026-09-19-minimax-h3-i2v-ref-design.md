# MiniMax H3 : image-vers-vidéo (fl2va) et références (ref2va)

Date : 2026-09-19. Statut : à relire. Branche : `feat/node-extraction-depth-control`.

## Objectif

Ajouter au port MLX de MiniMax H3 (t2va seul aujourd'hui) :

1. **i2v / fl2va** : première et/ou dernière image comme ancres de la vidéo.
2. **ref2va** : références images, vidéos (avec ou sans piste son) et audios.

Critère de réussite : génération réelle correcte avec les fichiers présents sur le disque
(`fl2va_pruned` et `ref2va_pruned`, GGUF Q5_0 et safetensors INT8 ConvRot, LoRA turbo 4 pas),
et parité numérique avec ComfyUI (torch) sur chaque brique.

## Hors périmètre

- `MiniMaxH3AddGuide` (ancrage à un index de trame arbitraire) et les 3 nœuds Fun ControlNet.
- Mode d'échantillonnage PDD (« head bank » du `FinalLayer`, non implémenté, `model.py:264`).
- Format W4A8 (`h3ErosMax_beta5_fp8`), rejeté par le loader.
- Batch > 1.

## Références (sources de vérité pour le comportement)

- ComfyUI : `comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/model.py` (`PackedLayout`,
  `_forward`), `comfy/text_encoders/minimax.py`, `qwen3vl.py`, `qwen_vl.py`, `model_base.py:2142-2290`.
- Rust : `inference/crates/media/mlx-gen/mlx-gen-minimax-h3` (`keyframe.rs`, `reference.rs`,
  `conditioning.rs`, `dit/positions.rs`, `denoise/packing.rs`), `inference/crates/llm/mlx-llm`
  (`qwen35_vision.rs`, `deepstack.rs`).
- Règle du projet (canon « Porting from a reference... ») : le comportement vient des références,
  l'exécution est reconstruite sur MLX ; le résultat numérique de la référence est le test d'acceptation.

## Faits établis par inspection (2026-09-19)

- Les DiT fl2va et ref2va ont **exactement** les mêmes clés, formes et dtypes (932 tenseurs
  safetensors, 532 GGUF, 0 différence). Le loader actuel les charge sans changement ; la tâche ne
  vit que dans la séquence d'entrée.
- Les deux fichiers TE (safetensors et GGUF) contiennent la tour vision : 351 tenseurs `visual.*`
  (27 blocs, 3 `deepstack_merger_list`, 1 `merger`).
- Une keyframe fixe le canevas et devient l'image 0 ; une référence ne fixe pas la géométrie.
  Ce sont deux tâches distinctes.
- Les lignes de condition ne sont jamais débruitées : elles sont ré-injectées à chaque forward
  (pas de masque d'échantillonnage pour i2v/ref).

## Architecture : 4 briques (ordre de dépendance 1 -> 2 -> 3 -> 4 ; 1 et 3 indépendantes)

### 1. Tour vision Qwen3-VL (`native/minimax_h3/vision_tower.py`)

ViT 27 blocs (largeur 1152, ffn 4304, 16 têtes, patch 16, patch temporel 2, fusion spatiale 2),
mergers DeepStack aux blocs 8/16/24 (norm après fusion), merger principal, M-RoPE entrelacé.
Chargement par le même flux par tenseur que le reste (`checkpoint_source.py`), tour en float32 comme la référence
(~0,6 Md de paramètres) ; mesuré : pic 3,57 Go, actif 2,38 Go (safetensors), 2,38 / 2,38 (GGUF). Chargement optionnel dans le loader de l'encodeur.
Précision mesurée sur les vrais poids contre ComfyUI : tour sur GPU (flux par défaut) : cosinus par ligne minimal 0,99966, erreur relative L2 0,67 % (le cosinus global 0,99998 masque cela) ; même forward sur flux CPU : cosinus par ligne minimal 0,9999999994, erreur relative L2 9,0e-6. L'architecture est exacte (CPU) ; l'écart GPU vient du matmul fp32 de MLX (~7,5e-4 relatif par matmul), de l'ordre de l'arrondi bf16. Les briques 2 et 3 doivent régler leurs seuils de parité sur le cosinus par ligne et l'erreur relative L2, pas sur le cosinus aplati.
**Acceptation** : sortie (embeddings fusionnés + 3 tenseurs DeepStack) contre
`comfy.text_encoders.qwen3vl` sur les vrais poids.

### 2. Présentation du prompt et encodeur avec vision

- Tokenisation : `<Picture i>: ` + `<|vision_start|>` bloc `<|vision_end|>` ; vidéo : `<Video k>: ` puis,
  par bloc de 2 images à 2 fps, `<T.T seconds>` + bloc vision ; libellé `<Audio j>: ` (texte seul, la
  forme d'onde n'atteint jamais l'encodeur). Ordre fixe : images, vidéos (son avant vidéo), audios.
- Encodeur : embeddings vision insérés aux positions pad, injection DeepStack dans les premières
  couches, positions M-RoPE entrelacées quand des jetons image existent (sinon RoPE simple, inchangé).
- Sortie : `hidden_states` inchangé + `token_tags` (positions vision = modalité vidéo 0, texte = 1).
**Acceptation** : ids et tags identiques à `MiniMaxH3Tokenizer` ; contexte encodeur contre ComfyUI.

**Notes de passage vers la brique 2**
- `preprocess_image` ne sait pas exprimer un bloc vidéo : un bloc de 2 images à 2 fps porte 2 images DISTINCTES dans un patch temporel (grid_t = ceil(n_images/2)) alors que la fonction réplique une seule image ; la brique 2 doit ajouter `preprocess_video_block` (référence : `comfy/text_encoders/minimax.py::process_video_block`).
- Entrée naturelle ComfyUI `[1,H,W,C]` (4-D) : `preprocess_image` attend `[H,W,3]` et lève un ValueError peu clair ; la brique 2 doit adapter l'entrée.
- Sorties de la tour en float32, alors que l'encodeur tourne en float16/4 bits : la brique 2 doit caster.
- `checkpoint_source.TensorSource.get()` retire chaque tenseur lu : le chargement de l'encodeur et celui de `load_vision_tower` ne peuvent pas partager une même source ouverte ; ouvrir le fichier deux fois ou entrelacer les chargements.
- Le cas `t>1` a été vérifié numériquement contre la référence (4,8e-7) : la brique 2 peut s'y fier.

### 3. DiT avec lignes de condition

- `layout.py` : segments `cond`, `cond_audio`, `ref_img`, `ref_audio` avant les cibles (audio puis
  vidéo restent les deux derniers segments) ; positions `cond_t = curseur + FRAME_RESCALE * index` ;
  curseur des références avancé par bloc (image : +1, audio : +durée, vidéo : max(audio, durée vidéo)).
- `model.py` : lignes de condition = latents normalisés patchifiés, avec bruit d'augmentation
  (`aug*z + (1-aug)*bruit`, graine fixe, timestep 0,999 par défaut) ; timestep par segment
  `max(t, aug)` ; tags de modulation par type de segment ; assemblage par tranches ; sortie sur les
  seules lignes cibles.
- Charge utile portée par le dict de conditioning : `keyframes`, `refs`, `token_tags`, `seed`.
**Acceptation** : forward MLX contre `MiniMaxH3Model` de ComfyUI sur un petit modèle à poids
aléatoires, avec et sans keyframes/refs.

### 4. Nœuds (`minimax_h3_nodes.py`)

- `ASDX_MiniMaxH3ImageToVideo` : `text_encoder`, `vae`, prompt, largeur, hauteur, longueur (grille
  17k+5), `first_frame`/`last_frame` optionnels. Première image étirée sur le canevas, dernière
  recadrée (cover) ; latents encodés par le pont `comfy.sd.VAE`.
- `ASDX_MiniMaxH3ReferenceToVideo` : mêmes entrées + `audio_vae`, `ref_image_size`
  (`match` par défaut / `max`), listes d'images (<= 9), vidéos (<= 3), sons de vidéo, audios (<= 3).
  Images : réduction seule vers la surface de génération (`match`) ou arête courte 2048 (`max`),
  multiples de 32. Vidéos : canevas adapté, >= 5 images, tronquées à 17k+5, son apparié par index,
  échantillonnage 2 fps pour l'encodeur. Audios : rééchantillonnés à 32 kHz (`torchaudio`) puis
  encodés par l'audio VAE via le pont `comfy.sd.VAE` (aucun encodeur audio à porter).
- Sortie : conditioning + latent AV, consommés par `ASDX_MiniMaxH3Sampler` existant.

## Mémoire

- Les lignes de référence traversent les 50 blocs à chaque pas ; l'attention est quadratique.
  À 1344x768, 124 images : 1 008 lignes par image latente, ~37 000 lignes cibles ; une référence
  image « match » ajoute ~1 000 lignes, « max » jusqu'à ~7 fois plus (calculs à confirmer par mesure).
- Défaut `match` ; `max` avec avertissement chiffré (lignes ajoutées).
- Le gate mémoire (`_gate_minimax_h3_component`) ajoute un coût par nombre de lignes de condition
  (le multiplicateur 1,0 du canon du 2026-09-19 ne couvre pas les activations).
- Ordre des étapes conservé : encodeur (avec vision) -> encodage VAE -> libération -> DiT.

## Erreurs (fail-closed)

Batch 1 ; plafonds 9/3/3 ; vidéo < 5 images refusée ; débit audio inconnu refusé plutôt que deviné ;
`vae` absent : les références ne conditionnent que le texte, avec message explicite ; format de
checkpoint ou marqueur inconnu : exception (déjà en place) ; jamais de réordonnancement des références.

## Tests

- Synthétiques (toujours exécutés) : petit DiT/encodeur à poids aléatoires, layout, positions,
  timesteps par segment, alignement des tags.
- Parité ComfyUI (ignorés sans ComfyUI/poids) : briques 1, 2, 3.
- Réels (drapeau `ASDX_FULL_GGUF_TEST=1`) : chargement vision, forward avec keyframe et avec ref.
- Génération i2v puis ref : jugement visuel par l'utilisateur, avec GGUF et safetensors.
- Chaque nouvelle métrique est validée sur le cas nul (deux entrées identiques -> identique).

## Estimation (incertaine, à revoir après la brique 1)

Brique 1 : 2-3 jours. Brique 2 : ~2. Brique 3 : ~3. Brique 4 : i2v ~1, ref images ~1-2,
ref vidéos + audios ~2-3. Total ~11-14 jours ; livrable par paliers : i2v, puis ref images,
puis ref vidéos/audios.

## Risques et questions ouvertes

- Le calcul M-RoPE entrelacé et le DeepStack sont les points les plus faciles à rendre
  « plausibles mais faux » : la parité contre ComfyUI en est la seule garantie.
- Le pont `comfy.sd.VAE` pour l'encodage audio n'a pas été exécuté ici ; à vérifier au début de
  la brique 4 (repli : port de l'encodeur, ~2 jours de plus).
- Géométrie (décidé) : les nœuds gardent largeur/hauteur explicites, comme ComfyUI, et non la
  déduction du canevas depuis la première keyframe que fait le Rust.
- LoRA turbo ref2v v0.1 : recette déjà dans `lora.py` (4 pas, shift 12/3) ; à confirmer en réel.
