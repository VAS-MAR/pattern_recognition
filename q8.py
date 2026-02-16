import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ---------------------------
# Columns (as required)
# ---------------------------
CONT = [
    "hour_float","latitude","longitude","victim_age",
    "temp_c","humidity","dist_precinct_km","pop_density"
]
CAT = ["weapon_code","scene_type","weather","vic_gender"]

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("crimes.csv")

train = df[df["split"] == "TRAIN"]
val   = df[df["split"] == "VAL"]
test  = df[df["split"] == "TEST"]

X_train = train[CONT + CAT]
X_val   = val[CONT + CAT]
X_test  = test[CONT + CAT]

y_train = train["killer_id"].values
y_val   = val["killer_id"].values

S = len(np.unique(y_train))   # number of killers

# ---------------------------
# PCA Preprocessing
# ---------------------------
pre = ColumnTransformer([
    ("scale", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

Z_train = pre.fit_transform(X_train)
Z_val   = pre.transform(X_val)
Z_test  = pre.transform(X_test)

# ---------------------------
# Choose m based on Q7 elbow
# ---------------------------
m = 5   # <-- set this based on your Q7 results
pca_m = PCA(n_components=m)

Z_train_m = pca_m.fit_transform(Z_train)
Z_val_m   = pca_m.transform(Z_val)
Z_test_m  = pca_m.transform(Z_test)

# ---------------------------
# K-means with k = S
# ---------------------------
kmeans = KMeans(n_clusters=S, random_state=0)
train_clusters = kmeans.fit_predict(Z_train_m)

# ---------------------------
# Build cluster -> killer mapping (majority vote)
# ---------------------------
cluster_to_killer = {}

for q in range(S):
    idx = np.where(train_clusters == q)[0]
    killers = y_train[idx]
    majority = np.bincount(killers).argmax()
    cluster_to_killer[q] = majority

# ---------------------------
# VAL Accuracy (optional check)
# ---------------------------
val_clusters = kmeans.predict(Z_val_m)
val_pred = [cluster_to_killer[c] for c in val_clusters]
val_acc = accuracy_score(y_val, val_pred)
print("VAL accuracy (Q8):", val_acc)

# ---------------------------
# Compute soft probabilities for TEST
# ---------------------------

def compute_probabilities(Z, centroids):
    """
    Convert distances to softmax probabilities.
    Z: (N, m) PCA features
    centroids: (S, m) k-means centers
    """
    # Distance matrix: shape (N, S)
    dists = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)

    # Convert distances to negative scores
    scores = -dists

    # Softmax
    exp_scores = np.exp(scores)
    prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return prob

centroids = kmeans.cluster_centers_
prob_test = compute_probabilities(Z_test_m, centroids)

# Hard predictions:
# For each test sample: cluster -> killer ID
test_clusters = kmeans.predict(Z_test_m)
test_pred = [cluster_to_killer[c] for c in test_clusters]

# ---------------------------
# Create submission.csv
# ---------------------------

columns = ["incident_id", "predicted_killer"]

# Add probability columns
for k in range(1, S+1):
    columns.append(f"p_killer_{k}")

rows = []

for i, inc_id in enumerate(test["incident_id"]):
    row = [inc_id, test_pred[i]]

    # probability p_killer_k for k = 1...S
    for k in range(1, S+1):
        row.append(prob_test[i, k-1])

    rows.append(row)

submission = pd.DataFrame(rows, columns=columns)
submission.to_csv("submission.csv", index=False)

print("submission.csv successfully created!")