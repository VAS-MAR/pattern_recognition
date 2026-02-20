import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Column definitions
CONT = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]
CAT = ["weapon_code", "scene_type", "weather", "vic_gender"]

# Load data
df = pd.read_csv("crimes.csv")

train = df[df["split"] == "TRAIN"]
val   = df[df["split"] == "VAL"]

X_train = train[CONT + CAT]
y_train = train["killer_id"].values

X_val = val[CONT + CAT]
y_val = val["killer_id"].values

# Preprocess (scale + onehot)
pre = ColumnTransformer([
    ("scaler", StandardScaler(), CONT),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)
])

# Build MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 hidden layers
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=0,
    early_stopping=True
)

model = Pipeline([
    ("pre", pre),
    ("mlp", mlp)
])

# Train
print("Training MLP...")
model.fit(X_train, y_train)
# Evaluate on VAL

pred_val = model.predict(X_val)
acc = accuracy_score(y_val, pred_val)
print(f"VAL Accuracy: {acc}")

# Permutation Feature Importance

# Baseline accuracy
A_base = acc

# Get feature names after one-hot encoding
ohe = model.named_steps["pre"].named_transformers_["onehot"]
ohe_names = ohe.get_feature_names_out(CAT)
all_feat = CONT + list(ohe_names)

# Transform VAL once
Z_val = model.named_steps["pre"].transform(X_val)

importances = []
rng = np.random.default_rng(0)

for j in range(Z_val.shape[1]):
    Zp = Z_val.copy()
    Zp[:, j] = rng.permutation(Zp[:, j])  # permute one feature
    pred = model.named_steps["mlp"].predict(Zp)
    Aj = accuracy_score(y_val, pred)
    importances.append((all_feat[j], A_base - Aj))

# Sort by importance
importances.sort(key=lambda x: x[1], reverse=True)
top5 = importances[:5]

print("\nTop 5 Most Important Features:")
for name, imp in top5:
    print(f"{name:35s} ΔA = {imp}")

# Plot top 5
names = [x[0] for x in top5]
values = [x[1] for x in top5]

plt.figure(figsize=(7,4))
plt.barh(names, values, color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 5 Most Important Features (MLP)")
plt.xlabel("Drop in Accuracy (ΔA)")
plt.tight_layout()
plt.savefig("q6_top5_features.png", dpi=150)
