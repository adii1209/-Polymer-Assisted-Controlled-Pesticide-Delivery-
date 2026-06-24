
# 🌱 Polymer-Assisted Controlled Pesticide Delivery
### A Machine Learning Framework for Release Profiling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-success)

A data-driven framework that predicts and optimizes the release behavior of pesticides from polymer-based carrier systems — replacing costly lab experiments with machine learning. Adapted from [Bannigan et al. (2023)](https://www.nature.com/articles/s41467-022-35343-w) (polymeric drug delivery) and applied to precision agriculture.


## What It Does

Conventional pesticide formulations deliver less than 10% of their active ingredient to targets — the rest leaches into soil and water. Polymer-based carriers solve this via controlled, sustained release, but designing them is slow and expensive.

This project trains a **LightGBM (LGBM) model** on physicochemical and formulation features to accurately predict cumulative pesticide release over time, enabling:

- **Fast in-silico screening** of new pesticide–polymer combinations
- **Identification of key formulation drivers** (DLC, SA/V ratio) via SHAP
- **Reproducible benchmarking** of 9 ML models under nested cross-validation

---

## Key Features

| Feature | Detail |
|---|---|
| **Dataset** | 3,600 time-series measurements across 240 unique pesticide–polymer formulations |
| **Models benchmarked** | MLR, Lasso, kNN, PLS, SVR, DT, RF, LGBM, NGBoost |
| **Best model** | LightGBM — lowest MAE, best generalization |
| **Validation strategy** | Nested CV — `GroupShuffleSplit` (outer) + `GroupKFold` (inner) |
| **Interpretability** | SHAP TreeExplainer — feature importance, dependence plots, PCA/t-SNE clustering |
| **Top features found** | Drug Loading Content (DLC) and Surface Area-to-Volume ratio (SA/V) |

---

## Repository Structure

```
.
├── codes/
│   ├── NESTED_CV_.ipynb                          # Core nested CV class
│   ├── Preliminary_model_screening.ipynb         # Step 1 – Screen all 9 models
│   ├── Preliminary_LGBM_model_refinement.ipynb   # Step 2 – Feature selection (17→15)
│   ├── Preliminary_refined_LGBM_model_training.ipynb  # Step 3 – Retrain + tune LGBM
│   ├── LGBM_Predictions.ipynb                    # Step 4 – Predict on new data
│   ├── Figuree1.ipynb                            # Fig 1 – Model MAE boxplots
│   ├── Figuree2.ipynb                            # Fig 2 – Spearman heatmap + dendrogram
│   ├── Figure_3.ipynb                            # Fig 3 – DLC vs SA/V scatter
│   ├── FIGUREE4&5.ipynb                          # Fig 4&5 – SHAP summary + dependence
│   └── FIGURE_6.ipynb                            # Fig 6 – SHAP PCA / t-SNE
│
├── trained_models/
│   ├── 15_feat_LGBM_model.pkl    ← final model  (~900 KB)
│   ├── 17_feat_LGBM_model.pkl                    (~900 KB)
│   ├── 17_feat_RF_model.pkl                      (~33 MB, track with Git LFS)
│   ├── 17_feat_NGB_model.pkl                     (~10 MB, track with Git LFS)
│   └── ...                       (MLR, Lasso, kNN, PLS, SVR, DT)
│
├── expanded_drug_delivery_dataset_with_timeseries.xlsx
├── polymers_report_finall_Aftr.docx
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm ngboost shap \
            matplotlib seaborn scipy openpyxl rdkit
```

> Requires **Python 3.8+**. A virtual environment is recommended.

### Option A — Google Colab (original environment, easiest)

1. Upload the dataset and `trained_models/` to your Google Drive
2. Open any notebook in [Google Colab](https://colab.research.google.com)
3. Mount your Drive at the top of the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update the `datafile` and model paths to point to your Drive location
5. Run all cells (`Runtime → Run all`)

### Option B — Local Jupyter

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/polymer-pesticide-ml.git
   cd polymer-pesticide-ml
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Update file paths** — search for `/content/` in each notebook and replace with your local path, e.g.:
   ```python
   # Before (Colab)
   datafile = "/content/Dataset_17_feat.xlsx"

   # After (local)
   datafile = "expanded_drug_delivery_dataset_with_timeseries.xlsx"
   ```
4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

### Recommended Run Order

Run the notebooks in this sequence to reproduce the full pipeline from scratch:

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `Preliminary_model_screening.ipynb` | Trains and compares all 9 ML models |
| 2 | `Preliminary_LGBM_model_refinement.ipynb` | Reduces features from 17 → 15 via Ward's clustering |
| 3 | `Preliminary_refined_LGBM_model_training.ipynb` | Final LGBM training with hyperparameter search |
| 4 | `LGBM_Predictions.ipynb` | Generates predictions on a new formulation dataset |
| 5–9 | `Figuree1–FIGURE_6.ipynb` | Reproduces all report figures |

> **Skip Steps 1–3** if you just want predictions — load `trained_models/15_feat_LGBM_model.pkl` directly.

### Quick Prediction Example

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('trained_models/15_feat_LGBM_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and prepare your formulation data
df = pd.read_excel('your_new_formulations.xlsx')
X = df.drop(['Experimental_index', 'DP_Group', 'Release'], axis='columns')

# Align features to the model's expected input
X = X.reindex(columns=model.feature_name_, fill_value=0)

# Scale and predict
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
predictions = model.predict(X_scaled)

print(predictions)   # Cumulative % release at each time point
```

---

## Dataset

**`expanded_drug_delivery_dataset_with_timeseries.xlsx`** — 3,600 rows × 23 columns

Each row is one cumulative release measurement at a given time point for one formulation.

| Column Group | Features |
|---|---|
| Identifiers | `Exp_no`, `formulation_id` |
| Pesticide properties | `pesticide_molecular_weight`, `logP`, `pKa`, `T50` |
| Polymer properties | `polymer_molecular_weight`, `pH_sensitivity_flag` |
| Formulation params | `drug_loading_capacity`, `encapsulation_efficiency`, `SA_V`, `burst_fraction`, `particle_size`, `temperature`, `pH` |
| Time-point releases | `release_t0.25`, `release_t0.5`, `release_t1`, `release_t7`, `release_t30` |
| **Target** | `Release` — cumulative % released at `Time` |

Formulations covered include: Atrazine, 2,4-D, Cypermethrin, Thiamethoxam paired with polymers such as Acacia gum, Acrylamide, Gelatin, Xanthan Gum, PVA, CMC, and more (240 unique combinations).

---

## Results Summary

- LGBM and RF significantly outperformed all linear baselines (p < 0.05)
- LGBM achieved the lowest median MAE with the smallest fold-to-fold variance
- **DLC** and **SA/V ratio** are the dominant release drivers (SHAP)
- Strong correlation (r = 0.957) between DLC and SA/V across formulations
- Predicted release curves closely track experimental profiles across all tested formulation types, capturing both the initial burst phase and the Higuchi diffusion-controlled plateau

---

## Large Model Files & Git LFS

Two trained models exceed GitHub's 50 MB recommendation:

| File | Size | Recommendation |
|---|---|---|
| `17_feat_RF_model.pkl` | ~33 MB | Track with Git LFS |
| `17_feat_NGB_model.pkl` | ~10 MB | Track with Git LFS |

To use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

Alternatively, exclude large `.pkl` files from the repo and share them via Google Drive.

---

## Getting Help

- **Project report:** See `polymers_report_finall_Aftr.docx` for full methodology, results, and references
- **Original methodology:** [Bannigan et al. (2023), *Nature Communications*](https://www.nature.com/articles/s41467-022-35343-w)
- **Dataset (Google Sheets):** [View here](https://docs.google.com/spreadsheets/d/1sEM-bcJS9lmGWzJm7YrgAIBRYnSM_dhj/edit?gid=831699553)
- **Issues:** Open a GitHub Issue for bugs or questions

---


## References

1. Bannigan et al. (2023). *Machine learning models to accelerate the design of polymeric long-acting injectables.* Nature Communications, 14, 35.
2. Zhang et al. (2023). *Research progress of a pesticide polymer-controlled release system.* Polymers, 15(13), 2810.
3. Zanino et al. (2023). *Polymers as controlled delivery systems in agriculture.* European Polymer Journal, 203, 112665.


