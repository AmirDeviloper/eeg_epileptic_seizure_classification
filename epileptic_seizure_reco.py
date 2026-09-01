import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from scipy.stats import skew, kurtosis, iqr, median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

RANDOM_STATE = 87
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. LOAD DATA AND INITIAL INSPECTION
# ============================================================
df = pd.read_csv("Epileptic Seizure Recognition.csv")
df.drop(columns=["Unnamed"], inplace=True)  # Remove the extra unnamed column

# Duplicate and missing value checks
print(f"Shape: {df.shape}")
print("Number of duplicate rows:", df.duplicated().sum())
print("Number of NaN values:", df.isna().sum().sum())

# Show first 5 rows and statistics of the raw dataset
print("\nFirst 5 rows of raw dataset:")
print(df.head())
print("\nRaw dataset description:")
print(df.describe())

X_raw = df.drop(columns=["y"]).values  # (n_samples, 178)
y = df["y"].values

label_map = {
    1: 'Seizure',
    2: 'Tumor',
    3: 'Healthy',
    4: 'Eyes Closed',
    5: 'Eyes Open'
}

print("\nClass distribution (counts & percentages):")
y_mapped = df['y'].map(label_map)

class_counts = y_mapped.value_counts()
class_perc = y_mapped.value_counts(normalize=True) * 100
dist_df = pd.DataFrame({'Count': class_counts, 'Percentage': class_perc.round(2)})
print(dist_df)

fs = 178  # Sampling frequency (Hz)

# ============================================================
# 2. FEATURE EXTRACTION FUNCTIONS (EEG-specific)
# ============================================================
def compute_hjorth(signal):
    """Compute Hjorth parameters: Activity, Mobility, Complexity."""
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / activity) if activity != 0 else 0
    complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility) if mobility != 0 and np.var(diff1) != 0 else 0
    return activity, mobility, complexity

def compute_band_powers(signal, fs):
    """Compute integrated power and mean PSD in five EEG bands using Welch."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal))

    band_powers = {}
    mean_psds = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.any(idx):
            band_powers[band] = np.trapezoid(psd[idx], freqs[idx])
            mean_psds[band] = np.mean(psd[idx])
        else:
            band_powers[band] = 0.0
            mean_psds[band] = 0.0
    return (list(band_powers.values()), list(mean_psds.values()), freqs, psd)

def sample_entropy(signal, m=2, r_factor=0.2):
    """Compute Sample Entropy to measure signal complexity."""
    N = len(signal)
    r = r_factor * np.std(signal)
    if r == 0:
        return 0.0

    def _phi(m_val):
        templates = np.array([signal[i:i+m_val] for i in range(N - m_val + 1)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i] - templates), axis=1)
            count += np.sum(dist < r) - 1  # exclude self-match
        return count

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    return -np.log(phi_m1 / phi_m)

def higuchi_fd(signal, kmax=10):
    """Compute Higuchi Fractal Dimension."""
    L = []
    x = np.array(signal)
    N = len(x)
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(x[idxs]))) * (N - 1) / (((N - m) // k) * k)
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    ln_k = np.log(np.arange(1, kmax+1))
    ln_L = np.log(L)
    fd = np.polyfit(ln_k, ln_L, 1)[0]
    return fd

def petrosian_fd(signal):
    """Compute Petrosian Fractal Dimension."""
    n = len(signal)
    # Number of sign changes in the first derivative (delta)
    diff_signal = np.diff(signal)
    n_delta = np.sum(np.diff(np.sign(diff_signal)) != 0)
    if n_delta == 0:
        return 1.0  # avoid division by zero
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))

def svd_entropy(signal, emb_dim=3, tau=1):
    """Compute SVD entropy of a delay embedding matrix."""
    N = len(signal)
    if N < emb_dim:
        return 0.0
    # Create embedding matrix
    n_vectors = N - (emb_dim - 1) * tau
    if n_vectors <= 0:
        return 0.0
    embedded = np.array([signal[i:i + emb_dim * tau:tau] for i in range(n_vectors)])
    # SVD and normalize singular values
    _, s, _ = np.linalg.svd(embedded, full_matrices=False)
    s_norm = s / np.sum(s)
    # Entropy
    return -np.sum(s_norm * np.log(s_norm + 1e-12))

def spectral_centroid(freqs, psd):
    """Compute spectral centroid (mean frequency weighted by power)."""
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    return np.sum(freqs * psd) / psd_sum

def extract_features(signal, fs):
    """
    Extract all required EEG features from a 178-point signal.
    Returns a numpy array of features.
    """
    features = []
    # Basic statistical features: mean, variance, std, skewness, kurtosis, min, max
    features.append(np.mean(signal))
    features.append(np.var(signal))
    features.append(np.std(signal))
    features.append(skew(signal))
    features.append(kurtosis(signal))
    features.append(np.min(signal))
    features.append(np.max(signal))

    # Range (max - min)
    features.append(np.max(signal) - np.min(signal))

    # Energy (sum of squares)
    features.append(np.sum(signal**2))

    # Mean Absolute Value (MAV)
    features.append(np.mean(np.abs(signal)))

    # Interquartile Range (IQR)
    features.append(iqr(signal))

    # Median Absolute Deviation (MAD)
    features.append(median_abs_deviation(signal))

    # Hjorth parameters
    features.extend(compute_hjorth(signal))

    # Band powers and mean PSDs (also get full spectrum for centroid)
    band_powers, mean_psds, freqs, psd = compute_band_powers(signal, fs)
    features.extend(band_powers)           # delta, theta, alpha, beta, gamma integrated powers
    features.extend(mean_psds)             # mean PSD in each band

    # Spectral centroid
    features.append(spectral_centroid(freqs, psd))

    # Sample Entropy
    features.append(sample_entropy(signal))

    # Higuchi Fractal Dimension
    features.append(higuchi_fd(signal))

    # Petrosian Fractal Dimension
    features.append(petrosian_fd(signal))

    # SVD Entropy
    features.append(svd_entropy(signal))

    return np.array(features)

# ============================================================
# 3. EXTRACT HANDCRAFTED FEATURES FROM ALL RAW DATA (for EDA)
# ============================================================
print("\nExtracting EEG features for EDA...")
feature_list = [extract_features(X_raw[i], fs) for i in range(len(X_raw))]
X_features_all = np.array(feature_list)
print(f"Extracted {X_features_all.shape[1]} features for {X_features_all.shape[0]} samples.")

# Feature names for all handcrafted features (order must match extract_features)
handcrafted_names = [
    'mean', 'variance', 'std', 'skewness', 'kurtosis', 'min', 'max', 'range',
    'energy', 'MAV', 'IQR', 'MAD',
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
    'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
    'delta_mean_psd', 'theta_mean_psd', 'alpha_mean_psd', 'beta_mean_psd', 'gamma_mean_psd',
    'spectral_centroid', 'sample_entropy', 'higuchi_fd', 'petrosian_fd', 'svd_entropy'
]

# ============================================================
# 4. INITIAL STATISTICAL ANALYSIS & CORRELATION HEATMAP
# ============================================================
# Correlation heatmap of handcrafted features
plt.figure(figsize=(16, 14))
corr_matrix = pd.DataFrame(X_features_all, columns=handcrafted_names).corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Extracted Handcrafted Features")
plt.tight_layout()
plt.show()

# Statistical summary of extracted features
print("\nStatistical description of handcrafted features (all data):")
print("\nFirst 5 rows of raw dataset:")
final_df = pd.DataFrame(X_features_all, columns=handcrafted_names)
print(final_df.head())
print(final_df.describe())

# ============================================================
# 5. STRATIFIED TRAIN/TEST SPLIT (keep raw signals for PCA later)
# ============================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Extract handcrafted features for train and test sets separately
# Parallel feature extraction using all CPU cores
print("\nExtracting handcrafted features for train/test splits [parallel]...")
X_train_feat = np.array(Parallel(n_jobs=-1)(
    delayed(extract_features)(sig, fs) for sig in X_train_raw
))
X_test_feat = np.array(Parallel(n_jobs=-1)(
    delayed(extract_features)(sig, fs) for sig in X_test_raw
))


# ============================================================
# 6. PCA ON RAW SIGNALS (10 components) – fit only on training data
# ============================================================
scaler_pca = StandardScaler()
X_train_raw_scaled = scaler_pca.fit_transform(X_train_raw)
X_test_raw_scaled  = scaler_pca.transform(X_test_raw)

pca = PCA(n_components=0.97, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_raw_scaled)
X_test_pca  = pca.transform(X_test_raw_scaled)
print(f"PCA on raw signal: retained variance = {pca.explained_variance_ratio_.sum():.3f}")

# ============================================================
# 7. COMBINE HANDCRAFTED FEATURES AND PCA COMPONENTS
# ============================================================
X_train_combined = np.hstack((X_train_feat, X_train_pca))
X_test_combined  = np.hstack((X_test_feat, X_test_pca))

# Combined feature names
pca_n = X_train_pca.shape[1]
pca_names = [f'pca_{i+1}' for i in range(pca_n)]
all_feature_names = handcrafted_names + pca_names

# Display first 5 rows and statistics of the final training feature matrix
print("\nFirst 5 rows of final combined training features (with labels):")
df_final_show = pd.DataFrame(X_train_combined[:5], columns=all_feature_names)
df_final_show.insert(0, 'Class', y_train[:5])
print(df_final_show)

print("\nStatistical description of final combined training features:")
print(pd.DataFrame(X_train_combined, columns=all_feature_names).describe())

# ============================================================
# 8. FEATURE IMPORTANCE USING MUTUAL INFORMATION (SelectKBest)
# ============================================================
# Use all features for model training; we just compute MI scores for display
k = 50
print(f'we keep {k} top features based on MI Score.')
selector = SelectKBest(mutual_info_classif, k=k)
selector.fit(X_train_combined, y_train)
mi_scores = selector.scores_

importance_df = pd.DataFrame({'Feature': all_feature_names, 'Mutual_Info': mi_scores})
importance_df = importance_df.sort_values('Mutual_Info', ascending=False).reset_index(drop=True)
print("\nFeature importance based on Mutual Information (all features):")
print(importance_df)


# We keep all features (can be no reduction, but we keep top featuers)
X_train_selected = selector.transform(X_train_combined)
X_test_selected  = selector.transform(X_test_combined)

# ============================================================
# 9. STANDARD SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled  = scaler.transform(X_test_selected)

print('Final Data Shape:')

# ============================================================
# 10. MODEL COMPARISON (CROSS-VALIDATION)
# ============================================================
models = {
    "k-NN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(random_state=RANDOM_STATE)
}

print("\nCross-validation Accuracy (mean ± std):")
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(cv_results, key=lambda x: cv_results[x][0])
print(f"\nBest model based on CV accuracy: {best_model_name}")

# ============================================================
# 11. HYPERPARAMETER TUNING FOR THE BEST MODEL
# ============================================================
param_grid_map = {
    "k-NN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "HistGradientBoosting": {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [None, 5, 10],
        'max_iter': [100, 200]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

param_grid = param_grid_map[best_model_name]

grid = GridSearchCV(
    models[best_model_name],
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
print(f"\nBest Parameters for {best_model_name}:")
print(grid.best_params_)

# ============================================================
# 12. FINAL EVALUATION ON TEST SET
# ============================================================
y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map)]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=[label_map[i] for i in sorted(label_map)],
            yticklabels=[label_map[i] for i in sorted(label_map)])
plt.title(f"Confusion Matrix - {best_model_name} (tuned)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


q = input('press enter to exit.')