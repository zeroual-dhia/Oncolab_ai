# OncoLab AI

**Projet de Fin d'Études — Pharmacie**
**Auteur et ML Engineer :** Zeroual Dhia Eddine
**Encadrant clinique :** Nadia Kerraouch

Système d'aide à la décision clinique qui prédit les **complications biologiques avant la prochaine cure de chimiothérapie FOLFOX** chez les patients traités pour un cancer colorectal.

Le modèle utilise l'état du cycle actuel du patient **et** son historique récent (bilans biologiques et complications des 1 à 2 cycles précédents) pour produire une probabilité pour chacune des 5 complications. L'objectif n'est pas de remplacer le jugement de l'oncologue mais de signaler les patients à haut risque précocement afin d'adapter les doses et d'éviter les hospitalisations non programmées.

---

## Structure du projet

```
oncolab-ai/
├── data/
│   ├── raw/                              # CSV original de l'équipe clinique — non modifié
│   ├── dataset_longitudinal.csv          # Jeu de données final pour le ML (704 lignes × 59 cols, 108 patients)
│   └── data_dictionary.md                # Documentation de chaque colonne
├── notebooks/
│   ├── 1_data_preparation.ipynb          # Données brutes → jeu de données longitudinal (pipeline complet)
│   └── 2_model_longitudinal.ipynb        # Entraînement + évaluation
├── models/                               # Modèles entraînés (.pkl) — 3 modèles × 5 cibles
├── outputs/
│   ├── feature_importance_longitudinal.png
│   └── per_patient_predictions.csv       # Prédictions sur l'ensemble de test pour vérification clinique
├── report/
│   ├── report_fr.tex                     # Rapport complet (LaTeX, en français)
│   └── thesis.md                         # Rapport structuré (Markdown)
├── README.md
└── requirements.txt
```

---

## Approche — pourquoi des variables longitudinales ?

Un seul bilan biologique est un faible prédicteur. Un taux de neutrophiles de 2,0 G/L a une signification très différente s'il est **stable** depuis 3 cycles ou s'il a **chuté de 8,0** au cycle précédent.
Ce projet encode cette trajectoire sous forme de variables numériques concrètes :

| Famille de variables | Exemples | Ce qu'elle capture |
|---|---|---|
| Biomarqueurs actuels | `hb`, `neut`, `plq`, `creat`, … | Photo instantanée de l'état du patient |
| Biomarqueurs décalés | `hb_t1`, `neut_t2`, … | Où en était le patient il y a 1–2 cycles |
| Variables delta | `delta_hb`, `delta_plq` | Vitesse et direction du changement |
| Variables de tendance | `trend_neut_3cycles`, `drop_rate_plq`, `variability_neut` | Dynamiques multi-cycles |
| Complications antérieures | `anemia_t1`, `neutropenia_t1`, … | Cette complication est-elle déjà survenue ? |

---

## Comment reproduire

### Prérequis

```bash
pip install -r requirements.txt
```

### Exécuter le pipeline complet

```bash
# 1. Construire le jeu de données longitudinal à partir du CSV brut
jupyter nbconvert --to notebook --execute --inplace notebooks/1_data_preparation.ipynb

# 2. Entraîner et évaluer le modèle
jupyter nbconvert --to notebook --execute --inplace notebooks/2_model_longitudinal.ipynb
```

Chaque notebook est autonome et comprend :
- Des cellules Markdown détaillant chaque étape
- Une analyse exploratoire avec visualisations
- Des décisions de prétraitement documentées
- Une validation croisée sans fuite de données (`GroupKFold` sur les identifiants patients, mise à l'échelle par pli)
- Un ensemble par vote souple des 3 meilleurs modèles (XGBoost + LightGBM + RandomForest)
- Une évaluation sur un ensemble de test isolé de 22 patients jamais vus

---

## Résultats principaux — 22 patients isolés, 141 cycles

| Complication | AUC-ROC | Sensibilité | Spécificité | VPP | F1 |
|---|---|---|---|---|---|
| Anémie | **0,995** | 1,00 | 0,89 | 0,94 | 0,97 |
| Neutropénie | **0,954** | 0,50 | 1,00 | 1,00 | 0,67 |
| Thrombocytopénie | **0,993** | 1,00 | 0,98 | 0,88 | 0,94 |
| Toxicité rénale | **0,999** | 0,91 | 1,00 | 1,00 | 0,95 |
| Toxicité hépatique | **0,972** | 0,95 | 0,99 | 0,98 | 0,97 |
| **Moyenne** | **0,983** | **0,87** | **0,97** | **0,96** | **0,90** |

Les probabilités par patient pour chaque cycle de test sont disponibles dans `outputs/per_patient_predictions.csv`.

---

## Protocole d'utilisation clinique (proposé)

1. **Éligibilité :** le patient doit avoir complété au moins 2 cures précédentes (nécessaires pour calculer l'historique décalé).
2. **Entrées :** données démographiques statiques, comorbidités, stadification TNM, biomarqueurs actuels, dose, historique des cycles récents.
3. **Sortie :** une probabilité par complication, plus un indicateur (O/N) au seuil de 0,50.
4. **Action :** les patients signalés nécessitent une surveillance rapprochée ou un ajustement de dose avant la prochaine cure.

---

## Limites

- **Données monocentriques**, 108 patients seulement — pas de validation externe
- **Étude rétrospective** ; un essai prospectif est nécessaire avant toute utilisation clinique
- La sévérité des complications (grade I–IV) **n'est pas** modélisée — uniquement la présence/absence
- La neutropénie n'a qu'environ 5,1 % de prévalence ⇒ la sensibilité est la métrique la plus fragile
- Le modèle nécessite ≥ 2 cycles d'historique ⇒ ne peut pas être appliqué aux deux premières cures d'un patient
- ~20 lignes avec des valeurs de créatinine suspectes ont été conservées en attente de confirmation clinique

---

## Licence et gouvernance des données

Les données des patients sont anonymisées et utilisées uniquement à des fins académiques dans le cadre de ce projet de fin d'études.
Ne pas redistribuer le CSV brut ni les modèles entraînés sans le consentement explicite de l'encadrant clinique.

---

## Références

- `data/data_dictionary.md` — Documentation colonne par colonne
- `report/report_fr.tex` — Rapport complet en français
- Références du protocole FOLFOX : voir le chapitre 2 du rapport (Contexte clinique)
# Oncolab_ai
