# RECON.md — Phase 0 Reconnaissance

## 1. Encodeur point-cloud existant

### LazSampler (shape_encoding.py:87-118)
Implémenté mais **non adapté DALES** :

- `load_laz_points_colors` lit x,y,z + RGB (optionnel)
- Applique **`mt.NDCnormalize` → viole l'invariant #1**
- Passe à `pointcloud_to_grid(pts, n, device)` avec `n=512` → voxel_size=2/512 en espace NDC
- Normales PCA via `estimate_normals_pca` (sklearn, k=30) sur les points NDC
- Couleurs → nearest-neighbour depuis le nuage

### process_GT_from_laz (shape_encoding.py:121-165)
Enchaîne LazSampler → PoNQ_grid → pyramide multi-résolution.
Utilise `PoNQ_grid.get_pool(size//t_size)` pour générer {512, 256, 128, 64, 32, 16}.pt.

**Ce qui manque pour DALES :**
- Pas de lecture de `legacy_semantic` ni `legacy_instance`
- NDCnormalize interdit (voxel_size non métrique)
- Pas de détrending du terrain
- Les canaux [6:9] sont des couleurs RGB (absentes dans DALES 2)

---

## 2. Format DiffusionTensor réel (diffusion_tensor.py:45-58)

10 canaux, slices **invariants** :

| Slice  | Nom actuel | Signification cible DALES (Route A) |
|--------|-----------|--------------------------------------|
| `[0:3]` | normals   | normale PCA par voxel               |
| `[3:6]` | offset    | (mean_pt − centre_voxel)/voxel_size **INVARIANT** |
| `[6:9]` | colors    | (intensité_norm, hauteur_sol_norm, sem_class_norm) |
| `[9]`   | mask      | 1=actif **INVARIANT**               |

Pas de changement de `get_feature_data`, `get_tensor_from_data`, `trilinear_upsample`, `get_global`, `get_local`.

---

## 3. Cartographie résolutions ↔ niveaux (DALES 500×500m)

| Fichier | voxel_size | voxels/côté (approx) | Niveau diffusion |
|---------|-----------|----------------------|-----------------|
| 16.pt   | 32 m      | ~16                  | 0 (base dense)  |
| 32.pt   | 16 m      | ~31                  | upsampler 1     |
| 64.pt   | 8 m       | ~62                  | upsampler 2     |
| 128.pt  | 4 m       | ~125                 | upsampler 3     |
| 256.pt  | 2 m       | ~250                 | upsampler 4     |

Construction : grille fine 1m/voxel (initiale, non sauvegardée), `initial_size=512`, pool 2× pour chaque cible.

`base_resolution=16` dans la config → cohérent : 16.pt donne un tenseur dense 16³ via `to_custom_dense()`.

---

## 4. Dataset DALES 2

- Chemin : `/data/moussabendjilali/archive/data/dales_2/`
- Train : 29 tuiles, Test : 11 tuiles
- Champs LAZ utiles : `legacy_semantic` (1–8), `intensity` (0–65535), x/y/z métrique
- RGB : nuls dans toutes les tuiles → remplacé par (intensité, hauteur, classe)
- Classe sol (détrending) : `legacy_semantic == 1`
- Tuiles 500×500m, ~11M pts, ~44 pts/m²

Classes DALES :
```
1=Ground  2=Vegetation  3=Cars  4=Trucks
5=PowerLines  6=Fences  7=Poles  8=Buildings
```

---

## 5. Écarts vs format natif

| Aspect | Format natif (mesh) | DALES (à implémenter) |
|--------|--------------------|-----------------------|
| Normalisation | NDCnormalize → [-1,1] | **Métrique fixe, 1m/voxel initial** |
| Canaux [6:9] | RGB couleur | intensité / hauteur_sol / sem_class |
| Détrending | N/A (mesh 3D) | DTM depuis points class=1 |
| Source champ sémantique | N/A | `legacy_semantic` (1–8) |
| Augmentation | Aucune | Rotations yaw + flip horizontal |

---

## 6. Emplacement du code à créer / modifier

**Créer :**
- `src/shape_encoding/pc_encoding.py` — encodeur DALES métrique
- `src/dataset/dales_dataset.py` — dataloader multi-tuiles (Phase 2)
- `data/dales_manifest.json` — manifeste 29+11 tuiles

**Modifier :**
- `src/diffusion/train_diffusion.py` — `get_gt_data`, `clip_data`, boucle, checkpoints
- `src/diffusion/train_upsamplers.py` — source données
- `src/diffusion/sample_diffusion.py` — `compute_canonical_base_grid`
- Configs : `epochs`, `src_path`, noms checkpoints
