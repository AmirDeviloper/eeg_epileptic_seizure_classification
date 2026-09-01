# EEG-Based Epileptic Seizure Recognition

A full machine learning pipeline for classifying EEG signal segments into five clinical categories — including active seizure activity — using handcrafted signal features, PCA, mutual-information feature selection, and gradient-boosted trees.

---

## Overview

This project works with the **Epileptic Seizure Recognition** dataset (11,500 samples, 5 balanced classes, 178 raw EEG values per sample) and builds an end-to-end classification pipeline: signal-domain feature engineering, dimensionality reduction, feature selection, model comparison, hyperparameter tuning, and a final held-out test evaluation.

**Classes:**
1. Seizure (active seizure activity)
2. Tumor (signal from a tumor-affected brain region)
3. Healthy (signal from a healthy region in a tumor patient)
4. Eyes Closed (healthy subject, eyes closed)
5. Eyes Open (healthy subject, eyes open)

---

## Pipeline

**1. Data Loading & Inspection**
11,500 samples × 178 raw signal columns, no missing values, no duplicates, perfectly balanced (2,300 samples/class).

**2. Handcrafted Feature Extraction (30 features per signal)**
- **Basic statistics (12):** mean, variance, std, skewness, kurtosis, min, max, range, energy, MAV, IQR, MAD
- **Hjorth parameters (3):** activity, mobility, complexity
- **Spectral band power (10):** integrated power and mean PSD (Welch's method) across delta, theta, alpha, beta, and gamma bands
- **Nonlinear complexity / entropy (5):** spectral centroid, sample entropy, Higuchi fractal dimension, Petrosian fractal dimension, SVD entropy

**3. Statistical & Correlation Analysis**
Pearson correlation heatmap across the 30 features, plus per-class boxplots of key features (variance, skewness, kurtosis, delta power, sample entropy, Hjorth mobility) to visualize class separability.

**4. Train/Test Split**
80/20 stratified split on the *raw* signals (9,200 train / 2,300 test), with features extracted separately per split to avoid data leakage.

**5. PCA on Raw Signals**
`StandardScaler` (fit on train only) + PCA with `n_components=0.97` reduces the 178-dimensional raw signal to **43 principal components**, retaining 97.19% of variance.

**6. Feature Fusion**
The 30 handcrafted features and 43 PCA components are concatenated into a single 73-dimensional feature vector per sample.

**7. Mutual-Information Feature Selection**
`SelectKBest` with `mutual_info_classif` selects the top **50** features out of 73. Beta-band power (`beta_power`, `beta_mean_psd`) and `hjorth_complexity` ranked highest.

**8. Final Scaling**
A second `StandardScaler` (fit on train only) standardizes the selected 50 features for scale-sensitive models.

**9. Model Comparison (5-fold CV)**

| Model | CV Accuracy |
|---|---|
| HistGradientBoosting | 0.8074 ± 0.0045 |
| Random Forest | 0.8005 ± 0.0070 |
| SVM | 0.7382 ± 0.0099 |
| k-NN | 0.7030 ± 0.0038 |

**10. Hyperparameter Tuning**
Grid search (5-fold CV, 18 combinations) on `HistGradientBoostingClassifier` over `learning_rate`, `max_depth`, and `max_iter`. Best config — `learning_rate=0.1, max_depth=10, max_iter=200` — improved CV accuracy to **0.8129**.

**11. Final Test Evaluation**
Evaluated once on the untouched 2,300-sample test set:

| Metric | Score |
|---|---|
| Accuracy | 81.61% |
| Precision (avg) | 81.67% |
| Recall (avg) | 81.61% |
| F1-score (avg) | 81.57% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Seizure | 0.972 | 0.978 | 0.975 |
| Tumor | 0.715 | 0.704 | 0.710 |
| Healthy | 0.738 | 0.704 | 0.721 |
| Eyes Closed | 0.887 | 0.833 | 0.859 |
| Eyes Open | 0.772 | 0.861 | 0.814 |

The model separates the clinically critical **Seizure** class almost perfectly (~98% F1). Most confusion occurs between **Tumor ↔ Healthy** (physiologically similar recordings from the same patients) and, to a lesser extent, **Eyes Closed ↔ Eyes Open**.

---

## Tech Stack

- Python, NumPy, SciPy, pandas
- scikit-learn (PCA, feature selection, model selection, HistGradientBoosting, Random Forest, SVM, k-NN)
- joblib (parallel feature extraction)
- matplotlib, seaborn

---

## Project Structure

```
.
├── epileptic_seizure_reco.py   # Full pipeline: feature extraction → PCA → selection → tuning → evaluation
├── Epileptic Seizure Recognition.csv   # Dataset (not included — see Dataset section)
└── report.pdf / report.docx    # Full written report with figures and tables
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib
   ```
2. Download the `dataset.zip`, then unzip it and place `Epileptic Seizure Recognition.csv` in the project root.
3. Run:
   ```bash
   python epileptic_seizure_reco.py
   ```

---

## Dataset

Epileptic Seizure Recognition UCI Machine Learning Repository.

---

## Contact

Feel free to reach out if you have questions or feedback!  
Telegram: [@AmirDevil](https://t.me/AmirDevil)

---

## License
This project is licensed under the MIT License.
By contributing, you agree that your contributions will be released under the same license.
