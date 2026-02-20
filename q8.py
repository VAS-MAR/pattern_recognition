import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 1. Ορισμός Στηλών (Columns)
CONT = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]
CAT = ["weapon_code", "scene_type", "weather", "vic_gender"]

# 2. Φόρτωση και Διαχωρισμός Δεδομένων (Splitting)
df = pd.read_csv("crimes.csv")

# Διαχωρισμός με βάση τη στήλη split
train = df[df["split"] == "TRAIN"].reset_index(drop=True)
val   = df[df["split"] == "VAL"].reset_index(drop=True)
test  = df[df["split"] == "TEST"].reset_index(drop=True)

X_train = train[CONT + CAT]
X_val   = val[CONT + CAT]
X_test  = test[CONT + CAT]

y_train = train["killer_id"].values
y_val   = val["killer_id"].values

# S: Ο αριθμός των δολοφόνων
S = len(np.unique(y_train))

# 3. Προεπεξεργασία (Preprocessing)
# Δημιουργία του transformer: scale τις συνεχείς, one-hot τις κατηγορικές
pre = ColumnTransformer([
    ("scaler", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

# Fit μόνο στο TRAIN, transform σε όλα
Z_train = pre.fit_transform(X_train)
Z_val   = pre.transform(X_val)
Z_test  = pre.transform(X_test)

# 4. PCA (από το Q7) - Μείωση Διαστάσεων
# Επιλέγουμε m=2 για την οπτικοποίηση και την ομαδοποίηση
m = 2
pca_m = PCA(n_components=m)

Z_train_m = pca_m.fit_transform(Z_train)
Z_val_m   = pca_m.transform(Z_val)
Z_test_m  = pca_m.transform(Z_test)

# 5. Q8: K-means Clustering (Unsupervised Learning)
# Εκπαίδευση του k-means με k = S clusters στον PCA χώρο
kmeans = KMeans(n_clusters=S, random_state=42, n_init=50)
train_clusters = kmeans.fit_predict(Z_train_m)

# 6. Αντιστοίχιση Clusters σε Killer IDs (Majority Vote)
cluster_to_killer = {}
for q in range(S):
    # Βρίσκουμε ποια εγκλήματα του TRAIN μπήκαν στο cluster q
    indices = np.where(train_clusters == q)[0]
    if len(indices) > 0:
        # Ποιος δολοφόνος εμφανίζεται συχνότερα σε αυτό το cluster;
        majority_label = np.bincount(y_train[indices]).argmax()
        cluster_to_killer[q] = majority_label
    else:
        cluster_to_killer[q] = -1

# 7. Αξιολόγηση στο VAL Split
val_clusters = kmeans.predict(Z_val_m)
val_pred = np.array([cluster_to_killer[c] for c in val_clusters])
val_acc = accuracy_score(y_val, val_pred)
print(f"Q8: VAL Accuracy = {val_acc:.4f}")

# 8. Προβλέψεις TEST και Δημιουργία submission.csv
test_clusters = kmeans.predict(Z_test_m)
test_pred = np.array([cluster_to_killer[c] for c in test_clusters])

distances = kmeans.transform(Z_test_m)

# 2. Μετατροπή αποστάσεων σε πιθανότητες (Softmax approach)
# Χρησιμοποιούμε exp(-dist) ώστε οι μικρές αποστάσεις να δίνουν μεγάλες πιθανότητες
inv_distances = np.exp(-distances)
# Κανονικοποίηση ώστε το άθροισμα κάθε γραμμής να είναι 1 [cite: 91]
probs_clusters = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

# 3. Δημιουργία DataFrame
submission = pd.DataFrame({"incident_id": test["incident_id"], "predicted_killer": test_pred})

# 4. Υπολογισμός πιθανοτήτων για κάθε killer_id (1 έως S)
for k in range(1, S + 1):
    # Βρίσκουμε ποια clusters (q) αντιστοιχούν στον δολοφόνο k [cite: 218]
    matching_clusters = [q for q, killer in cluster_to_killer.items() if killer == k]

    if matching_clusters:
        # Αθροίζουμε τις πιθανότητες των clusters που ανήκουν στον δολοφόνο k
        submission[f"p_killer_{k}"] = probs_clusters[:, matching_clusters].sum(axis=1)
    else:
        # Αν ο δολοφόνος k δεν κέρδισε κανένα cluster, του δίνουμε μια πολύ μικρή
        # βασική πιθανότητα (π.χ. από το πλησιέστερο cluster) για να μην είναι 0
        submission[f"p_killer_{k}"] = 1e-5

    # Επανεξισορρόπηση (Normalization) για να αθροίζουν ακριβώς στο 1 [cite: 91]
prob_cols = [f"p_killer_{k}" for k in range(1, S + 1)]
submission[prob_cols] = submission[prob_cols].div(submission[prob_cols].sum(axis=1), axis=0)

submission.to_csv("submission.csv", index=False)

print("Q8: submission.csv created with decimal probabilities!")

# 9. Οπτικοποίηση (Scatter Plot)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(Z_test_m[:, 0], Z_test_m[:, 1],
                      c=test_pred, cmap="tab10", s=20, alpha=0.7)
plt.title("Q8: K-means Clusters on TEST Split (PCA space)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Predicted Killer ID")
plt.grid(True)
plt.savefig("q8_clusters_plot.png")
