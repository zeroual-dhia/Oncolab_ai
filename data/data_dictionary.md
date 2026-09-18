# Dictionnaire de données — `dataset_longitudinal.csv`

**Source :** `Collect Data Onco Lab AI - Feuille 2.csv` (110 patients, 947 lignes brutes)
**Sortie :** 704 lignes × 59 colonnes · 108 patients
**Construit par :** `notebooks/1_data_preparation.ipynb`

Chaque ligne représente **une cure de chimiothérapie** pour un patient.
Les cibles sont les complications observées **pendant/après** cette cure.
Toutes les variables sont des valeurs connues **avant** cette cure, de sorte que la prédiction est causalement valide.

---

## 1 · Identification du patient

| Colonne | Type | Unité | Description |
|---|---|---|---|
| `patient_id` | chaîne | — | Identifiant unique du patient (`P1`, `P2`, …). Utilisé pour le regroupement en CV — jamais comme variable. |
| `cycle_num` | entier | indice de cure | 0 = première cure ; dans ce jeu de données toujours ≥ 2 (historique de 2 cycles requis). |

## 2 · Démographie et comorbidités (statiques par patient)

| Colonne | Type | Unité | Plage / valeurs normales | Description |
|---|---|---|---|---|
| `age` | entier | années | 18–95 | Âge à la première cure |
| `sex_M` | binaire | — | 0 = F, 1 = M | Sexe encodé |
| `bmi` | flottant | kg/m² | 18,5–25 normal | Indice de Masse Corporelle |
| `etat` | catégorie | — | `Sous-poids` / `Normale` / `Surpoids` | Statut nutritionnel |
| `hta` | binaire | — | 0/1 | Hypertension artérielle |
| `diabetes` | binaire | — | 0/1 | Diabète |
| `renal_disease` | binaire | — | 0/1 | Maladie rénale chronique |
| `hepatic_disease` | binaire | — | 0/1 | Maladie hépatique chronique |
| `comorbidity_count` | entier | — | 0–4 | Somme des quatre comorbidités ci-dessus |

## 3 · Caractéristiques du cancer (statiques par patient)

| Colonne | Type | Unité | Valeurs | Description |
|---|---|---|---|---|
| `cancer_type` | catégorie | — | `Colon` / `Rectum` / `Colon gauche` / … | Localisation tumorale |
| `T_stage` | entier | — | 0–4 (NaN possible) | Taille/profondeur de la tumeur |
| `N_stage` | entier | — | 0–4 (NaN possible) | Atteinte ganglionnaire régionale |
| `M_stage` | entier | — | 0 ou 1 | Métastase à distance |
| `has_metastasis` | binaire | — | 0/1 | Dérivé : maladie métastatique (O/N) |
| `treatment_intent` | catégorie | — | `Adj` / `Neoadj` | Intention du traitement : adjuvant ou néo-adjuvant |

## 4 · Paramètres de la cure (cure actuelle)

| Colonne | Type | Unité | Plage normale | Description |
|---|---|---|---|---|
| `interval_days` | entier | jours | 14 ou 21 | Jours depuis la cure précédente |
| `dose_oxali` | flottant | mg/m² | 85 (dose pleine) | Dose d'Oxaliplatine administrée |
| `dose_5fu_bolus` | flottant | mg/m² | 400 | Dose de 5-FU bolus |
| `dose_5fu_cont` | flottant | mg/m² | 2400 | Dose de 5-FU en perfusion continue |
| `dose_af` | flottant | mg/m² | 400 | Dose d'acide folinique (leucovorine) |

## 5 · Biomarqueurs (mesurés AVANT la cure actuelle)

| Colonne | Type | Unité | Plage normale | Description |
|---|---|---|---|---|
| `hb` | flottant | g/dL | 12–16 (F), 13–17 (M) | Hémoglobine |
| `neut` | flottant | G/L | 2,0–7,5 | Numération absolue des neutrophiles |
| `plq` | flottant | G/L | 150–400 | Numération plaquettaire |
| `uree` | flottant | g/L | 0,15–0,45 | Urée (fonction rénale) |
| `creat` | flottant | mg/L | 6–13 | Créatinine (fonction rénale) |
| `asat` | flottant | UI/L | < 40 | Aspartate aminotransférase (foie) |
| `alat` | flottant | UI/L | < 40 | Alanine aminotransférase (foie) |
| `bil` | flottant | mg/L | < 12 | Bilirubine totale (foie) |

## 6 · Biomarqueurs décalés (historique des cycles précédents)

Ces colonnes encodent « où en était le patient » il y a 1 et 2 cycles — connu avant le début de la cure actuelle.

| Colonne | Unité | Formule | Description |
|---|---|---|---|
| `hb_t1`, `neut_t1`, `plq_t1`, `creat_t1` | (identique à la base) | `shift(1)` par patient | Biomarqueur du cycle précédent |
| `hb_t2`, `neut_t2`, `plq_t2`, `creat_t2` | (identique à la base) | `shift(2)` par patient | Biomarqueur d'il y a deux cycles |

## 7 · Indicateurs de complications antérieures (historique des cycles précédents)

| Colonne | Type | Formule | Description |
|---|---|---|---|
| `anemia_t1`, `neutropenia_t1`, `thrombocytopenia_t1`, `renal_toxicity_t1`, `hepatic_toxicity_t1` | binaire | `shift(1)` de la cible | Cette complication est-elle survenue **après** le cycle précédent ? |
| `anemia_t2`, …, `hepatic_toxicity_t2` | binaire | `shift(2)` de la cible | Cette complication est-elle survenue il y a 2 cycles ? |

## 8 · Variables de trajectoire dérivées

| Colonne | Unité | Formule | Signification clinique |
|---|---|---|---|
| `delta_hb` | g/dL | `hb − hb_t1` | Variation de l'hémoglobine depuis le cycle précédent |
| `delta_neut` | G/L | `neut − neut_t1` | Variation des neutrophiles depuis le cycle précédent |
| `delta_plq` | G/L | `plq − plq_t1` | Variation des plaquettes depuis le cycle précédent |
| `trend_neut_3cycles` | G/L | `mean(neut_t2, neut_t1, neut)` | Niveau moyen des neutrophiles sur 3 cycles |
| `variability_neut` | G/L | `std(neut_t2, neut_t1, neut)` | Volatilité du taux de neutrophiles sur 3 cycles |
| `drop_rate_plq` | ratio | `(plq − plq_t2) / plq_t2` | Chute relative des plaquettes sur 2 cycles (−0,5 = 50 % de baisse) |

## 9 · Cibles (binaires, observées pendant/après la cure actuelle)

| Colonne | Prévalence | Description | Référence clinique |
|---|---|---|---|
| `target_anemia` | 68,0 % | Anémie | Hb < seuil normal pour le sexe |
| `target_neutropenia` | 5,1 % | Neutropénie | Neut < 1,5 G/L |
| `target_thrombocytopenia` | 17,9 % | Thrombocytopénie | Plq < 150 G/L |
| `target_renal_toxicity` | 9,1 % | Toxicité rénale | Élévation de la créatinine / urée |
| `target_hepatic_toxicity` | 28,6 % | Toxicité hépatique | ASAT ou ALAT > 3× limite supérieure de la normale |

---

## Notes sur la qualité des données

| Problème | Lignes / patients affectés | Résolution |
|---|---|---|
| Format décimal européen (`"11,7"`) | Toutes les colonnes numériques | Analysé à l'ingestion |
| Doublons `(patient, cure)` | 26 lignes supprimées, 20 patients | Conserver la dernière (= cure administrée) |
| Casse de l'identifiant patient (`p16` vs `P16`) | 1 patient | Mise en majuscules de tous les identifiants |
| Corruption de la chaîne TNM | P31 (plusieurs lignes) | Forcé au stade correct `T3N1M0` |
| Libellé `Stade 4` sans chiffre M | P107 | Défini `M_stage = 1` |
| 1 ligne avec `uree = 36` (erreur de saisie) | 1 ligne | Division par 100 → 0,36 g/L |
| Biomarqueurs en cellules/µL | `plq*`, `neut*`, colonnes dérivées | Division par 1000 dans le notebook de modélisation (G/L est le standard clinique) |
| `T_stage`/`N_stage` manquants | 2 lignes (P107) | Laissés en NaN — les modèles à base d'arbres les gèrent nativement |

---

## Provenance

| Fichier | Construit par | Notes |
|---|---|---|
| `Collect Data Onco Lab AI - Feuille 2.csv` | Équipe clinique (saisie manuelle, export Google Sheets) | Brut, non modifié, 947 lignes × 42 colonnes |
| `dataset_longitudinal.csv` | `notebooks/1_data_preparation.ipynb` | 704 lignes × 59 colonnes |
| Modèles entraînés (`models/*.pkl`) | Script d'entraînement dans `notebooks/2_model_longitudinal.ipynb` | Membres de l'ensemble par cible + normaliseur + liste des variables |
