import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from q5 import model_svm

# Columns (from assignment PDF)
CONT = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]

CAT = ["weapon_code", "scene_type", "weather", "vic_gender"]

# Load dataset
df = pd.read_csv("crimes.csv")

train = df[df["split"] == "TRAIN"]
val   = df[df["split"] == "VAL"]

X_train = train[CONT + CAT]
X_val = val[CONT + CAT]

# Preprocess: scale + onehot
pre = ColumnTransformer([
    ("scaler", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

# Fit on TRAIN only
Z_train = pre.fit_transform(X_train)

# Transform VAL using same fitted preprocessor
Z_val = pre.transform(X_val)

# Q7a — Run PCA on TRAIN
pca = PCA()
pca.fit(Z_train)

# Q7b — Plot eigenvalues
plt.figure(figsize=(8,4))
plt.plot(np.arange(1, len(pca.explained_variance_)+1),
         pca.explained_variance_, marker='o')
plt.xlabel("Component index j")
plt.ylabel("Eigenvalue λ_j")
plt.title("Q7 – PCA Eigenvalues (Scree Plot)")
plt.grid(True)
plt.savefig("q7_scree_plot.png", dpi=150)

# Choose m = 2 for visualization
pca2 = PCA(n_components=2)
Z_train_2D = pca2.fit_transform(Z_train)
Z_val_2D = pca2.transform(Z_val)

# Q7c — Colour VAL points by SVM predictions
# (Assumes you trained model_svm in Q5)

svm_pred_val = model_svm.predict(X_val)

plt.figure(figsize=(7,6))
scatter = plt.scatter(Z_val_2D[:,0], Z_val_2D[:,1],
                      c=svm_pred_val, cmap="tab20", s=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Q7 – PCA Projection of VAL (coloured by SVM predictions)")
plt.colorbar(scatter, label="Predicted Killer (SVM)")
plt.grid(True)
plt.savefig("q7_pca2_svm_colours.png", dpi=150)
