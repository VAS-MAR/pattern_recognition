import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ---------------------------
# Columns
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

train = df[df["split"] == "TRAIN"].reset_index(drop=True)
val   = df[df["split"] == "VAL"].reset_index(drop=True)
test  = df[df["split"] == "TEST"].reset_index(drop=True)

X_train = train[CONT + CAT]
X_val   = val[CONT + CAT]
X_test  = test[CONT + CAT]

y_train = train["killer_id"].values
y_val   = val["killer_id"].values

S = len(np.unique(y_train))   # number of killers

# ---------------------------
# Preprocessing
# ---------------------------
pre = ColumnTransformer([
    ("scale", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

Z_train = pre.fit_transform(X_train)
Z_val   = pre.transform(X_val)
Z_test  = pre.transform(X_test)

# ---------------------------
# PCA (m from Q7)
# ---------------------------
m = 2
pca = PCA(n_components=m)

Z_train_m = pca.fit_transform(Z_train)
Z_val_m   = pca.transform(Z_val)
Z_test_m  = pca.transform(Z_test)

# ---------------------------
# K-means (k = S)
# ---------------------------
kmeans = KMeans(n_clusters=S, random_state=0)
train_clusters = kmeans.fit_predict(Z_train_m)

# ---------------------------
# Cluster → Killer mapping
# ---------------------------
cluster_to_killer = {}

for c in range(S):
    idx = np.where(train_clusters == c)[0]
    majority = np.bincount(y_train[idx]).argmax()
    cluster_to_killer[c] = majority

# ---------------------------
# VAL accuracy (check)
# ---------------------------
val_clusters = kmeans.predict(Z_val_m)
val_pred = np.array([cluster_to_killer[c] for c in val_clusters])
val_acc = accuracy_score(y_val, val_pred)
print("Q8 VAL accuracy:", val_acc)

# ---------------------------
# TEST predictions
# ---------------------------
test_clusters = kmeans.predict(Z_test_m)
test_pred = np.array([cluster_to_killer[c] for c in test_clusters])

# ---------------------------
# Create submission.csv
# ---------------------------
submission = pd.DataFrame({
    "incident_id": test["incident_id"],
    "predicted_killer": test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")