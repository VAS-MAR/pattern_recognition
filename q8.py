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

# ΕΛΕΓΧΟΣ 1: Μέγεθος splits
print(f"TRAIN: {len(train)}, VAL: {len(val)}, TEST: {len(test)}")
print(f"Total in df: {len(df)}, Sum of splits: {len(train) + len(val) + len(test)}")

X_train = train[CONT + CAT]
X_val   = val[CONT + CAT]
X_test  = test[CONT + CAT]

# ΕΛΕΓΧΟΣ 2: Missing values
print("\nMissing values in TRAIN:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])
print("\nMissing values in TEST:")
print(X_test.isnull().sum()[X_test.isnull().sum() > 0])

# Χειρισμός NaN
for col in CONT:
    X_train[col].fillna(X_train[col].median(), inplace=True)
    X_val[col].fillna(X_val[col].median(), inplace=True)
    X_test[col].fillna(X_test[col].median(), inplace=True)

for col in CAT:
    X_train[col].fillna("UNKNOWN", inplace=True)
    X_val[col].fillna("UNKNOWN", inplace=True)
    X_test[col].fillna("UNKNOWN", inplace=True)

y_train = train["killer_id"].values
y_val   = val["killer_id"].values

S = len(np.unique(y_train))   # number of killers
print(f"\nNumber of killers (S): {S}")

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

print(f"\nTransformed shapes - Train: {Z_train.shape}, Val: {Z_val.shape}, Test: {Z_test.shape}")

# ---------------------------
# PCA (m from Q7)
# ---------------------------
m = 2
pca = PCA(n_components=m)

Z_train_m = pca.fit_transform(Z_train)
Z_val_m   = pca.transform(Z_val)
Z_test_m  = pca.transform(Z_test)

print(f"PCA shapes - Train: {Z_train_m.shape}, Val: {Z_val_m.shape}, Test: {Z_test_m.shape}")

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

print(f"\nCluster to Killer mapping: {cluster_to_killer}")

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

# ΕΛΕΓΧΟΣ 3: Μέγεθος predictions
print(f"\nTest predictions length: {len(test_pred)}")
print(f"Expected (test size): {len(test)}")
print(f"Match: {len(test_pred) == len(test)}")

# ---------------------------
# Create submission.csv
# ---------------------------
submission = pd.DataFrame({
    "incident_id": test["incident_id"],
    "predicted_killer": test_pred
})

# ΕΛΕΓΧΟΣ 4: Final submission check
print(f"\nSubmission shape: {submission.shape}")
print(f"Unique incidents in submission: {submission['incident_id'].nunique()}")
print(f"Duplicate incidents: {submission['incident_id'].duplicated().sum()}")

# Στο τέλος του κώδικα, πριν το to_csv:

# Δημιουργία λίστας με όλα τα expected incidents
expected_incidents = set(test["incident_id"])
submission_incidents = set(submission["incident_id"])

missing = expected_incidents - submission_incidents
extra = submission_incidents - expected_incidents

print(f"\nExpected incidents: {len(expected_incidents)}")
print(f"Incidents in submission: {len(submission_incidents)}")
print(f"Missing incidents: {len(missing)}")
if missing:
    print(f"Missing IDs: {sorted(list(missing))[:10]}")  # Πρώτα 10
print(f"Extra incidents: {len(extra)}")
if extra:
    print(f"Extra IDs: {sorted(list(extra))[:10]}")

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

# ΕΛΕΓΧΟΣ 5: Preview
#print("\nFirst 5 rows of submission:")
print(submission.value_counts())
