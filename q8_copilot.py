import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ---------------------------
# Columns
# ---------------------------
CONT = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]
CAT = ["weapon_code", "scene_type", "weather", "vic_gender"]

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("crimes.csv")

train = df[df["split"] == "TRAIN"].reset_index(drop=True)
val = df[df["split"] == "VAL"].reset_index(drop=True)
test = df[df["split"] == "TEST"].reset_index(drop=True)

print(f"TRAIN: {len(train)}, VAL: {len(val)}, TEST: {len(test)}")

# ---------------------------
# Prepare X, y
# ---------------------------
X_train = train[CONT + CAT].copy()
X_val = val[CONT + CAT].copy()
X_test = test[CONT + CAT].copy()

# Χειρισμός NaN
for col in CONT:
    median_val = X_train[col].median()
    X_train[col].fillna(median_val, inplace=True)
    X_val[col].fillna(median_val, inplace=True)
    X_test[col].fillna(median_val, inplace=True)

for col in CAT:
    X_train[col].fillna("UNKNOWN", inplace=True)
    X_val[col].fillna("UNKNOWN", inplace=True)
    X_test[col].fillna("UNKNOWN", inplace=True)

y_train = train["killer_id"].values
y_val = val["killer_id"].values

S = len(np.unique(y_train))  # number of killers
print(f"Number of killers (S): {S}")

# ---------------------------
# Preprocessing (Q8a - part 1)
# ---------------------------
pre = ColumnTransformer([
    ("scale", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

Z_train = pre.fit_transform(X_train)
Z_val = pre.transform(X_val)
Z_test = pre.transform(X_test)

print(f"\nPreprocessed feature dimensions: {Z_train.shape[1]}")

# ---------------------------
# Q8a: PCA projection (use m from Q7)
# ---------------------------
# IMPORTANT: Replace this with your actual m from Q7
# Common choices: m = 10, 15, 20, or based on explained variance
m = 2  # 🔴 REPLACE THIS with your value from Q7!

print(f"\nUsing m = {m} principal components (from Q7)")

pca = PCA(n_components=m)
Z_train_m = pca.fit_transform(Z_train)
Z_val_m = pca.transform(Z_val)
Z_test_m = pca.transform(Z_test)

print(f"Explained variance ratio (first {m} PCs): {pca.explained_variance_ratio_.sum():.4f}")

# ---------------------------
# Q8b: K-means clustering on TRAIN
# ---------------------------
print(f"\nRunning k-means with k = {S} clusters...")
kmeans = KMeans(n_clusters=S, random_state=0, n_init=10)
train_clusters = kmeans.fit_predict(Z_train_m)

print(f"Cluster assignments - unique values: {np.unique(train_clusters)}")
print(f"Cluster sizes: {np.bincount(train_clusters)}")

# ---------------------------
# Q8c: Cluster → Killer mapping (majority vote)
# ---------------------------
cluster_to_killer = {}

for q in range(S):
    # Find all TRAIN incidents in cluster q
    idx_in_cluster_q = np.where(train_clusters == q)[0]

    if len(idx_in_cluster_q) > 0:
        # Get true killer labels for these incidents
        true_killers = y_train[idx_in_cluster_q]

        # Find majority killer
        g_q = np.bincount(true_killers).argmax()
        cluster_to_killer[q] = g_q
    else:
        # Empty cluster (shouldn't happen, but handle it)
        cluster_to_killer[q] = 0

print(f"\nCluster → Killer mapping g(q):")
for q, killer in cluster_to_killer.items():
    count = np.sum(train_clusters == q)
    print(f"  Cluster {q} → Killer {killer} ({count} incidents)")

# ---------------------------
# Q8d: VAL evaluation
# ---------------------------
val_clusters = kmeans.predict(Z_val_m)
val_pred = np.array([cluster_to_killer[c] for c in val_clusters])
val_acc = accuracy_score(y_val, val_pred)

print(f"\n{'=' * 50}")
print(f"Q8d: VAL Accuracy = {val_acc:.4f}")
print(f"{'=' * 50}")

# ---------------------------
# Q8e: TEST predictions
# ---------------------------
test_clusters = kmeans.predict(Z_test_m)
test_pred = np.array([cluster_to_killer[c] for c in test_clusters])

print(f"\nTEST predictions generated: {len(test_pred)}")
print(f"Unique predicted killers: {np.unique(test_pred)}")
print(f"Prediction distribution: {np.bincount(test_pred)}")

# ---------------------------
# Create submission.csv
# ---------------------------
submission = pd.DataFrame({
    "incident_id": test["incident_id"].values,
    "predicted_killer": test_pred
})

# Verification
print(f"\n{'=' * 50}")
print(f"SUBMISSION VERIFICATION:")
print(f"{'=' * 50}")
print(f"Submission shape: {submission.shape}")
print(f"Expected shape: ({len(test)}, 2)")
print(f"Match: {len(submission) == len(test)} ✓" if len(submission) == len(
    test) else f"Match: {len(submission) == len(test)} ✗")

submission.to_csv("submission.csv", index=False)
print(f"\n✓ submission.csv created successfully!")

# Double-check file
reloaded = pd.read_csv("submission.csv")
print(f"Reloaded CSV shape: {reloaded.shape}")

print("\nFirst 5 predictions:")
print(submission.head())

# ---------------------------
# Q8f: Visualization - Scatter plot on PC1 vs PC2
# ---------------------------
print(f"\n{'=' * 50}")
print(f"Q8f: Creating visualization...")
print(f"{'=' * 50}")

# Project TEST onto first 2 PCs for visualization
pca_2d = PCA(n_components=2)
Z_train_2d = pca_2d.fit_transform(Z_train)
Z_test_2d = pca_2d.transform(Z_test)

# Create scatter plot
plt.figure(figsize=(12, 8))

# Use a colormap with enough distinct colors
colors = plt.cm.tab20(np.linspace(0, 1, S))

for killer_id in range(S):
    mask = test_pred == killer_id
    plt.scatter(
        Z_test_2d[mask, 0],
        Z_test_2d[mask, 1],
        c=[colors[killer_id]],
        label=f'Killer {killer_id}',
        alpha=0.6,
        s=30,
        edgecolors='k',
        linewidths=0.5
    )

plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('Q8f: TEST incidents projected onto PC1-PC2\nColored by k-means predicted killer', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('q8f_kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved plot: q8f_kmeans_clusters.png")


# ---------------------------
# Q8f: Commentary
# ---------------------------
print(f"\n{'=' * 50}")
print(f"Q8f COMMENTARY:")
print(f"{'=' * 50}")

print("""
Visual inspection of the scatter plot:

1. CLUSTER SEPARATION:
   - If clusters are well-separated in the PC1-PC2 space, it suggests that
     the k-means assignments align well with the geometric structure.
   - Overlapping regions indicate ambiguity where different killers have
     similar crime patterns.

2. ALIGNMENT WITH K-MEANS:
   - Well-defined, compact clusters → good k-means performance
   - Scattered or intermingled points → k-means struggles to separate

3. DIMENSIONALITY:
   - Remember: we used m={} PCs for k-means, but only show 2 PCs here.
   - Clusters may be better separated in higher dimensions.

4. IMPLICATIONS:
   - Tight visual clusters suggest distinct killer signatures.
   - Diffuse patterns suggest killers with similar MOs are hard to distinguish.

Check the saved plot 'q8f_kmeans_clusters.png' for detailed visual analysis.
""".format(m))

print(f"\n{'=' * 50}")
print(f"Q8 COMPLETE!")
print(f"{'=' * 50}")