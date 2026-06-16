# TODO — Adapter ShapeShifter (single-exemplar) au régime dataset DALES

> **Mission.** Faire passer le dépôt `nissmar/ShapeShifter` du régime *un modèle par forme* à un régime *un modèle par niveau entraîné sur l'ensemble des tuiles DALES* (nuages de points LiDAR aérien), tout en conservant l'architecture multi-échelle sparse-voxel. La fidélité grossière est acceptable. La cible finale (Phase 3) est le conditionnement par un embedding de graphe de scène (GVAE), avec génération inconditionnelle préservée.
>
> **Mode opératoire imposé.** Travailler par phases. Chaque phase a un *critère d'acceptation* vérifiable. **Ne pas commencer une phase tant que le smoke-test de la précédente n'est pas vert.** Committer phase par phase. En cas d'ambiguïté, lire le code réel et reporter avant de modifier — ne jamais supposer.

---

## RÈGLES D'OR (invariants — toute violation = bug)

1. **Aucune normalisation par-tuile.** Bannir `mt.NDCnormalize` dans le chemin de données DALES. La taille de voxel est **fixe en mètres** et identique pour toutes les tuiles, sinon un bâtiment n'a pas la même taille en voxels d'une tuile à l'autre et les statistiques de patchs ne s'alignent pas.
2. **`offset` reste au slice `[3:6]`** du vecteur de feature. `get_global()`, `get_local()`, `trilinear_upsample()` lui appliquent un traitement spatial spécifique. Ne pas le déplacer.
3. **`mask` reste au slice `[-1:]`.** Convention : `+1` = voxel actif (surface), `-1` = inactif (pruné par `remove_mask`).
4. **Rotation/flip = au niveau des points bruts, AVANT voxelisation.** Ne jamais tourner les indices `ijk` sans tourner aussi les vecteurs `offset` et `normal` — c'est une source d'erreur silencieuse. Le plus sûr : tourner le nuage de points, puis ré-encoder.
5. **Détrender le terrain par tuile** (mettre le sol à z≈0) avant voxelisation, sinon une tuile sur une colline a toute sa structure à une hauteur en voxels différente d'une tuile plate.
6. **Les 11 tuiles de test DALES ne doivent JAMAIS entrer dans le manifeste d'entraînement.** Elles servent uniquement à la validation de généralisation.
7. **Un modèle par niveau, pas par tuile.** Les checkpoints ne doivent plus être indexés par `model_name` de tuile mais par un identifiant de dataset (ex. `dales`).

## DO-NOT (pièges classiques à éviter)

- ❌ Ne pas garder `to_batch()` (réplication) dans le chemin d'entraînement dataset.
- ❌ Ne pas voxeliser en ré-instanciant une grille fVDB à chaque step si ça bottleneck — précalculer le cache (voir Phase 1.4).
- ❌ Ne pas oublier de propager `cond` jusque dans `ddpm_sample`/`ddim_sample` (Phase 3), sinon l'échantillonnage ignore le conditionnement entraîné.
- ❌ Ne pas changer le nombre de canaux sans mettre à jour **tous** les slices de `get_feature_data` ET `in_channels`/`out_channels` des modèles.
- ❌ Ne pas faire de flip vertical (gravité) ni de mise à l'échelle agressive (casse l'échelle métrique).

---

## PHASE 0 — Reconnaissance (obligatoire avant toute modif)

- [ ] **0.1** Lire intégralement : `src/shape_encoding/shape_encoding.py`, `src/utils/diffusion_tensor.py`, `src/diffusion/train_diffusion.py`, `src/diffusion/train_upsamplers.py`, `src/diffusion/sample_diffusion.py`, `src/utils/model.py`, `src/utils/fvdb_diffusion.py`, `src/utils/fvdb_utils.py`, `src/utils/PoNQ_grid.py`.
- [ ] **0.2** **Localiser l'encodeur point-cloud déjà écrit par l'utilisateur** pour entraîner la tuile DALES 25×25 m unique. `shape_encoding.py` natif est *mesh-only* (igl `signed_distance`, `.obj` watertight, `.glb` texturé, UV→couleur) : il a forcément été remplacé/contourné. Identifier ce code, comprendre comment il produit un `DiffusionTensor`, et **reporter** son emplacement + format avant d'écrire la Phase 1. Si introuvable, le signaler : la Phase 1 le construit alors de zéro.
- [ ] **0.3** Confirmer la cartographie résolutions↔niveaux : `base_resolution=16` ; niveau 0 → 16³ ; niveau `l>0` → `res_1 = 16·2^(l-1)`, `res_2 = 2·res_1`. Tenseurs attendus sous `data/GT_sparse_tensors/{name}/{res}.pt` pour res ∈ {16,32,64,128,256}.
- [ ] **0.4** Confirmer le format de feature dans `DiffusionTensor.get_feature_data` : `normals=[:3]`, `offset=[3:6]`, `colors=[6:9]`, `mask=[-1:]` (10 canaux).
- [ ] **Sortie de phase** : un court rapport `RECON.md` (emplacement de l'encodeur PC existant, format réel produit, écarts vs format natif).

---

## PHASE 1 — Pipeline de données → dataset de tuiles

Objectif : transformer N tuiles DALES en N jeux de tenseurs multi-résolution `data/GT_sparse_tensors/dales/{tile_id}/{res}.pt`, cohérents métriquement, augmentés.

### 1.1 — Schéma de feature (source unique de vérité)
- [ ] Centraliser le schéma dans `diffusion_tensor.py` (constantes de slices + commentaire). **Route A par défaut (zéro changement de slices, recommandée pour le 1er jet)** : réaffecter sémantiquement les 10 canaux sans toucher `get_feature_data` :
  - `[0:3]` = **normale** estimée par PCA locale (k≈16 voisins) — utile pour distinguer sol/toit/façade, peu coûteux. (Mettre à 0 si trop coûteux, mais préférer PCA.)
  - `[3:6]` = **offset local** (point − centre_voxel)/voxel_size, clampé à `[-0.5, 0.5]` (INCHANGÉ, invariant #2).
  - `[6:9]` = `(intensité_normalisée, hauteur_au_dessus_du_sol_normalisée, class_id_scalaire)`.
  - `[9]` = **mask** (INCHANGÉ).
- [ ] Documenter **Route B (montée en gamme ultérieure, optionnelle)** : passer à `[offset(3), normal(3), intensity(1), class_onehot(C), mask(1)]`. Si choisie, mettre à jour **ensemble** : `get_feature_data`, `get_tensor_from_data`, `trilinear_upsample` (divisions par mask), `get_global`/`get_local`, `fill_upsampled_with_gt`, `to_custom_dense`, `colored_PC`, + `in/out_channels`. Ne PAS faire Route B au premier jet.
- [ ] Note : la classe en scalaire continu (Route A) est tolérable car (a) coarse OK, (b) on arrondit au décodage, (c) en Phase 3 la classe viendra du graphe, pas de la diffusion.

### 1.2 — Encodeur point-cloud (remplace `MeshSampler`)
- [ ] Créer `src/shape_encoding/pc_encoding.py` : `tuile .ply/.las/.npy → DiffusionTensor multi-résolution`, produisant **exactement** le format `DiffusionTensor.get_tensor_from_data(grid, f0_3, offset, f6_9, mask)`.
- [ ] **Voxelisation métrique fixe** : `voxel_size_fine` constant (ex. 0.1–0.25 m), extent canonique de tuile fixe (ex. 25 m). Construire la `GridBatch` fVDB avec `voxel_sizes`/`origins` en mètres. **Ne pas** appeler `NDCnormalize`.
- [ ] **Détrending** : estimer le sol (points classe *ground* ou min-z par colonne), construire un DTM grossier lissé, soustraire de tous les z. Stocker l'offset DTM si un re-géoréférencement est voulu plus tard. Calculer `hauteur_au_dessus_du_sol` au passage (canal feature).
- [ ] **Agrégation par voxel** : offset = moyenne des (points−centre)/voxel_size ; intensité = moyenne ; classe = vote majoritaire ; normale = PCA. `mask=1` pour tout voxel occupé.
- [ ] **Pyramide multi-échelle** : réutiliser le mécanisme de pooling existant (`PoNQ_grid.get_pool` + `compute_local_offset`, moyenne QEM pour préserver les arêtes) pour générer les résolutions {16,32,64,128,256} comme dans `process_GT`. Sauvegarder chaque résolution en `.pt`.

### 1.3 — Manifeste & split
- [ ] Générer `data/dales_manifest.json` : liste des `tile_id` d'entraînement (29 tuiles) et de test (11), chemins, et **stats par classe par tuile** (pour l'échantillonnage pondéré en Phase 2).
- [ ] Garantir l'exclusion stricte des tuiles de test (invariant #6).

### 1.4 — Augmentation (précalcul hors-ligne)
- [ ] Pour chaque tuile d'entraînement, précalculer K variantes (ex. 8 rotations yaw uniformes + miroir horizontal) **au niveau des points bruts puis ré-encoder** (invariant #4). Les stocker comme tuiles distinctes dans le cache `.pt` (29×8×2 ≈ 460 jeux — trivial en disque pour DALES).
- [ ] Vérifier qu'après rotation, `offset` et `normale` sont cohérents (test : reconstruire le nuage via `colored_PC`/`get_global` et comparer visuellement à la rotation attendue).

- [ ] **Critère d'acceptation Phase 1** : `python src/shape_encoding/pc_encoding.py --tile <id>` produit les 5 `.pt` ; un script de visualisation (réutiliser `DiffusionTensor.get_global()` → export PLY) montre une tuile plausible, sol à z≈0, à la bonne échelle métrique. Deux tuiles différentes ont des grilles **différentes** (pas la réplication).

---

## PHASE 2 — Généralisation de l'entraînement (le cœur du régime dataset)

Objectif : un modèle de diffusion par niveau, entraîné sur des batches **multi-tuiles**. Les niveaux fins se généralisent quasi gratuitement (ils sont déjà conditionnés sur le contexte grossier local via crops à champ réceptif limité) ; le travail est surtout au chargement des données.

### 2.1 — Dataset/dataloader multi-tuiles (remplace la réplication)
- [ ] Créer `src/dataset/dales_dataset.py` exposant un échantillonnage qui, à chaque step, tire **B tuiles distinctes** (pondérées par présence des classes rares — voitures, camions, poteaux, lignes, clôtures — via les stats du manifeste) et assemble un vrai batch fVDB hétérogène via `fvdb.jcat` des grilles **différentes** + `fvdb.jcat` des données.
- [ ] **Remplacer** `DiffusionTensor.to_batch()` (réplication) dans le chemin d'entraînement par cet assemblage multi-tuiles. (Conserver `to_batch` pour la rétro-compat de l'échantillonnage single-shape si besoin, mais ne plus l'utiliser à l'entraînement.)
- [ ] Adapter `get_gt_data()` (dans `train_diffusion.py`) : au lieu de charger `{model_name}/{res}.pt`, charger un **mini-batch de tuiles** depuis le dataset. Garder la logique niveau 0 (dense via `to_custom_dense`) vs niveau>0 (paire `res_1`/`res_2` + `X_UP = trilinear_upsample` + `fill_upsampled_with_gt`).

### 2.2 — Cropping par élément de batch
- [ ] Refactorer `clip_data(X0, X0_BLUR, size)` : actuellement il échantillonne des centres sur **une** grille globale (`X0.grid.ijk.jdata`, `X0.grid_count`). Le réécrire pour cropper **par élément de batch** (boucle sur `grid_count`, tirer un centre parmi les voxels actifs de cet élément, `clip`, puis `jcat`). Conserver la taille de crop `2·clip_size` (40³ par défaut).
- [ ] Niveau 0 : pas de crop (grille dense complète 16³), inchangé sauf la source multi-tuiles.

### 2.3 — Boucle & checkpoints
- [ ] Remplacer la donnée fixe en mémoire par un **rechargement par step** depuis le dataset (la boucle actuelle re-bruite un X0 figé ; il faut maintenant re-tirer un batch). Préserver le clip de gradient (`clip_grad_norm_(…, 1.)`) et l'EMA de loss.
- [ ] Augmenter `epochs` (config) : on a beaucoup plus de données effectives → surveiller via loss held-out plutôt qu'un nombre fixe. Ajouter une **EMA des poids** du modèle (recommandé pour la stabilité).
- [ ] Indexer les checkpoints par dataset+niveau : `checkpoints/diffusion_models/dales_{level}_{time}.pt` (idem upsamplers : `checkpoints/upsamplers/dales_{level}.pt`).
- [ ] Répliquer les mêmes changements de source de données dans `train_upsamplers.py` (il importe `get_gt_data` — il bénéficiera automatiquement du refactor s'il est fait proprement ; vérifier que l'`UpSampler` voit bien des crops multi-tuiles).

### 2.4 — Validation de généralisation
- [ ] Ajouter une éval périodique : loss de diffusion/upsampler sur des crops de **tuiles de test tenues à l'écart** (jamais vues). C'est le signal clé que le régime dataset généralise (vs mémorisation).

- [ ] **Critère d'acceptation Phase 2** : entraîner les 5 niveaux sur ≥10 tuiles. Échantillonner (Phase 2.5) produit des scènes qui ne sont **aucune** des tuiles d'entraînement à l'identique, et la loss held-out décroît. Tient en ≤48 Go.

### 2.5 — Échantillonnage inconditionnel dataset
- [ ] Dans `sample_diffusion.py`, `compute_base_grid()` part aujourd'hui de la grille dense d'**une** tuile stockée. Pour l'inconditionnel dataset, fournir une **grille de base canonique** : un cube 16³ **plein** (tous voxels actifs) servant de support d'occupation initial, le modèle décidant le `mask`. Implémenter `compute_canonical_base_grid(base_res=16, extent_m=25, batch)`.
- [ ] Vérifier le pipeline complet noise→niveau0→…→niveau4 avec les checkpoints `dales_{level}` et l'export nuage de points (`remove_mask` → `get_global` → PLY).

---

## PHASE 3 — Conditionnement GVAE + CFG + standalone

Objectif : piloter le **niveau grossier** par l'embedding de graphe, tout en gardant la génération inconditionnelle (standalone). Le hook existe déjà.

### 3.1 — Threading du conditionnement
- [ ] `model.py` → `DiffusionCNN.forward(self, x, t, cond=None)` : `cond` est actuellement **ignoré**. Construire un conditionnement par-voxel en diffusant le vecteur de graphe (un vecteur par élément de batch) vers chaque voxel via `x.data.jidx` (index de batch par voxel), puis `torch.cat((x.data.jdata, t, cond_broadcast), -1)`. Mettre à jour le premier `SparseConv3d` : `in_channels + time_emb + cond_dim`.
- [ ] `fvdb_diffusion.py` → propager `cond` : `SparseDiffusion.forward(X, X_Blur, cond=None)` passe `cond` à `self.model(noisy_latents, times, cond)` ; idem dans `ddpm_sample` et `ddim_sample` (les appels `self.model(noisy_grid, …)`). **Sans ça, l'échantillonnage ignore le conditionnement** (DO-NOT #3).
- [ ] N'appliquer le conditionnement **qu'au niveau 0** dans un premier temps (le layout vit là). Les niveaux >0 restent inconditionnels (raffinement local par statistiques de patchs).

### 3.2 — Classifier-Free Guidance (standalone préservé)
- [ ] Embedding **null** appris (`nn.Parameter`). À l'entraînement, remplacer `cond` par le null avec proba `p_drop ≈ 0.1`. À l'échantillonnage, combiner cond/null avec une échelle de guidage `w`.
- [ ] **Standalone** = `cond=None` → null embedding → récupère exactement l'inconditionnel. Vérifier que le mode null reproduit la qualité Phase 2.5.

### 3.3 — Du graphe au niveau grossier
- [ ] Interface minimale d'abord : embedding **global** GVAE (un vecteur par scène) comme `cond`. Brancher l'API GVAE existante du dépôt utilisateur.
- [ ] Variante plus forte (optionnelle) : **splatter** les nœuds du graphe `(x,y,z,class,rot)` dans la grille de base 16³ (chaque instance → région de voxels avec sa classe) comme **initialisation** + comme features de conditionnement (analogue au « BEV map » de Control-3D-Scene, mais 3D au niveau grossier). Pour le standalone : échantillonner un graphe depuis le prior latent du GVAE → splat → diffuser.

- [ ] **Critère d'acceptation Phase 3** : à `w>1`, les scènes générées respectent mieux classes/positions demandées par le graphe qu'en mode null (mesurer l'occupation par classe aux positions demandées) ; le mode null reste un générateur inconditionnel valide.

---

## SMOKE-TESTS (à câbler tôt, exécuter à chaque phase)

- [ ] **T1 (Phase 1)** : encoder 2 tuiles → 2 grilles distinctes, sol à z≈0, échelle métrique correcte, round-trip rotation cohérent.
- [ ] **T2 (Phase 2)** : un batch d'entraînement contient des tuiles **différentes** (assert sur l'unicité des `ijk` entre éléments de batch). 200 steps sur 4 tuiles → loss décroît.
- [ ] **T3 (Phase 2.5)** : génération inconditionnelle complète 5 niveaux → PLY non vide, ≠ tuiles d'entraînement.
- [ ] **T4 (Phase 3)** : avec `cond=None` la sortie est identique (à l'aléa près) au mode Phase 2.5 ; avec un `cond` réel, la sortie change de façon cohérente.

## ORDRE D'EXÉCUTION RÉSUMÉ
Phase 0 (recon + RECON.md) → Phase 1 (données métriques + augmentation) → Phase 2 (dataloader multi-tuiles + crops par élément + éval held-out) → Phase 2.5 (échantillonnage inconditionnel) → Phase 3 (cond GVAE + CFG). **Ne pas anticiper Phase 3 avant que Phase 2.5 ne génère des scènes inconditionnelles correctes.**

## FICHIERS À MODIFIER / CRÉER (récapitulatif)
- Créer : `src/shape_encoding/pc_encoding.py`, `src/diffusion/dales_dataset.py`, `data/dales_manifest.json`, `RECON.md`.
- Modifier : `src/diffusion/train_diffusion.py` (`get_gt_data`, `clip_data`, boucle, checkpoints), `src/diffusion/train_upsamplers.py` (source données), `src/diffusion/sample_diffusion.py` (`compute_base_grid` → canonique/graph), `src/utils/diffusion_tensor.py` (schéma + ne plus répliquer), `src/utils/model.py` (`DiffusionCNN.forward` cond), `src/utils/fvdb_diffusion.py` (threading `cond` + CFG), configs `train_diffusion_0.yaml`/`train_diffusion_up.yaml`/`train_upsampler.yaml` (epochs, chemins dataset, `cond_dim`).
- Ne pas toucher (sauf Route B) : les slices de `get_feature_data`, le traitement spatial de `offset`/`mask`.