# Multi-Model Classification Streamlit App

## a. Problem Statement

This project demonstrates five classical machine learning classification models
trained on a single public dataset. A Streamlit web app lets a user upload test data,
select one of the five trained models, and view its evaluation metrics, confusion
matrix, and classification report — all against the same held-out test split used
during training, so results are directly comparable across models.

## b. Dataset Description

**Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Source:** UCI Machine Learning Repository (W.N. Street, W.H. Wolberg, O.L.
  Mangasarian) — https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
  — bundled directly in `scikit-learn` via `sklearn.datasets.load_breast_cancer`.
- **Instances:** 569
- **Features:** 30 numeric, continuous features (mean/standard-error/worst values for
  10 measurements taken from digitized images of a fine needle aspirate of a breast
  mass — e.g. radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension).
- **Target:** Binary — malignant (0) vs. benign (1).
- **Meets FR-1:** 30 features ≥ 12 required; 569 rows ≥ 500 required.

## c. GitHub Repository Link

`<add your repository URL here after pushing to GitHub>`

## d. Comparison Table

Computed on the held-out test split (20%, `random_state=42`, stratified), identical
split and `StandardScaler` preprocessing used for all 5 models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Best performer across every metric on this dataset — accuracy, AUC, and MCC all highest. The 30 features (radius, texture, area, etc.) separate the two classes in a way that's close to linearly separable after scaling, which plays directly to logistic regression's strength. |
| Decision Tree | Weakest performer here. A single tree with `max_depth=6` sacrifices some accuracy to control overfitting; training accuracy is noticeably higher than test accuracy, consistent with the overfitting tendency noted in `MODELS_SOURCE_OF_TRUTH.md`. |
| kNN | Solid, mid-pack performance once features were scaled — confirms scaling was necessary and effective (an unscaled kNN would perform far worse on features with very different ranges, e.g. area vs. smoothness). |
| Naive Bayes | Reasonable AUC (0.9868) but lower accuracy/F1 than the top models — the independence assumption between features is somewhat violated here since many of the 30 features are derived from the same underlying measurements (e.g. radius and area are correlated). |
| Random Forest (Ensemble) | Strong and stable — matches kNN on accuracy/F1 and has the second-highest AUC, improving over the single Decision Tree as expected, though it doesn't surpass Logistic Regression on this particular dataset. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest score on all 6 metrics. Random Forest is typically the strongest all-round performer per general expectations, but on this dataset's near-linearly-separable feature space, a well-scaled linear model edges it out. |

---

## Project Structure

```
multi-model-classifier-app/
├── app.py                  # Streamlit app (upload, model select, metrics, confusion matrix)
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test split (114 rows), same schema as training data
└── model/
    ├── train_models.ipynb  # data prep, training, metrics, artifact export
    ├── scaler.pkl           # fitted StandardScaler (reused, not refit, at inference)
    ├── feature_names.pkl    # exact training-time column order
    ├── metrics.json          # 6 metrics per model, shared source of truth
    ├── comparison_table.csv
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── kNN.pkl
    ├── Naive_Bayes.pkl
    └── Random_Forest_Ensemble.pkl
```

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL, upload `test_data.csv`, and select any of the 5
models from the dropdown to see its metrics, confusion matrix, and classification
report.

## Re-running Training

Open `model/train_models.ipynb` in Jupyter and run all cells top to bottom (or from a
terminal: `jupyter nbconvert --to notebook --execute --inplace model/train_models.ipynb`).
This regenerates all 5 model files, `scaler.pkl`, `feature_names.pkl`, and
`metrics.json` from scratch using the same fixed random seed.
