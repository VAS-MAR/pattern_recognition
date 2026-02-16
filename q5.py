import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- USER CONFIG ---------------------------
CSV_PATH = "crimes.csv"          # Path to your dataset
KERNEL = "rbf"                   # "rbf" or "poly"
FIGDIR = "figures"               # Where to save figures

# Initial/default hyperparameters (a candidate that will be tried)
C_0 = 1.0
GAMMA_0 = "scale"
DEGREE_0 = 2                     # Only used for poly
COEF0_0 = 0.0                    # Only used for poly

# Small manual search on VAL (set to True to try a tiny grid and pick best on VAL)
RUN_SMALL_SEARCH = True

# Column names as per assignment PDF: continuous + raw categorical codes + meta  [1](https://2cq1sj-my.sharepoint.com/personal/kouklaki_serrenity_onmicrosoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/Pattern_Recognition_Course_Assignment.pdf)
CONT_COLS = ["hour_float","latitude", "longitude", "victim_age", "temp_c","humidity","dist_precinct_km","pop_density"]
CAT_COLS = ["weapon_code", "scene_type", "weather", "vic_gender"]
META_COLS = ["incident_id", "split", "killer_id"]


def load_and_split(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = set(CONT_COLS + CAT_COLS + META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    train_df = df[df["split"] == "TRAIN"].copy()
    val_df = df[df["split"] == "VAL"].copy()
    test_df = df[df["split"] == "TEST"].copy()
    return train_df, val_df, test_df


def make_preprocessor():
    cont_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("cont", cont_pipe, CONT_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return pre


def small_search_space(kernel: str):
    if kernel == "rbf":
        Cs = [0.3, 1, 3]
        gammas = ["scale", 0.1, 0.03]
        return [(c, g, None, None) for c in Cs for g in gammas]
    else:
        Cs = [0.3, 1, 3]
        gammas = ["scale", 0.1]
        degrees = [2]
        coef0s = [0, 1]
        return [(c, g, d, c0) for c in Cs for g in gammas for d in degrees for c0 in coef0s]

def fit_and_eval(model: Pipeline, X_tr: pd.DataFrame, y_tr: np.ndarray,X_val: pd.DataFrame, y_val: np.ndarray, kernel_tag: str, out_dir: str):
    model.fit(X_tr, y_tr)
    pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, pred_val)
    cm = confusion_matrix(y_val, pred_val)

    print(f"VAL accuracy ({kernel_tag}): {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Q5 – SVM ({kernel_tag}) – VAL Confusion Matrix")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"q5_confusion_matrix_{kernel_tag}.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    return acc, cm

def plot_pca_and_support_vectors (full_df: pd.DataFrame, y: np.ndarray,
                                  split_mask: np.ndarray,pipeline_pre: ColumnTransformer,
                                  clf_ovr: OneVsRestClassifier,kernel_tag: str,out_dir: str):

    # Preprocess all (TRAIN+VAL here) with the fitted preprocessor
    X_all = full_df[CONT_COLS + CAT_COLS]
    Z_all = pipeline_pre.transform(X_all)  # numpy array in preprocessed space

    # PCA(2) fit on TRAIN only
    pca2 = PCA(n_components=2, random_state=0)
    Z_train = Z_all[split_mask]
    pca2.fit(Z_train)

    # Transform for plotting
    Z2_all = pca2.transform(Z_all)

    # Build a meshgrid in PCA2 and map back via inverse_transform (approximate)
    x_min, x_max = Z2_all[:, 0].min() - 0.5, Z2_all[:, 0].max() + 0.5
    y_min, y_max = Z2_all[:, 1].min() - 0.5, Z2_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_pre = pca2.inverse_transform(grid_2d)
    grid_pred = clf_ovr.predict(grid_pre)

    # Decision regions + TRAIN scatter
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.25, cmap="tab20")
    sc = ax.scatter(Z2_all[split_mask, 0], Z2_all[split_mask, 1],
                    c=y[split_mask], cmap="tab20", s=12,
                    edgecolors="k", linewidths=0.2, alpha=0.9, label="TRAIN")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Killer ID")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Q5 – SVM ({kernel_tag}) – Decision regions in PCA(2)")
    fig.savefig(os.path.join(out_dir, f"q5_decision_regions_pca2_{kernel_tag}.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Support vectors in PCA(2)
    sv_list = []
    for est in clf_ovr.estimators_:
        sv = est.support_vectors_
        sv_list.append(sv)
    if len(sv_list) > 0:
        SV = np.vstack(sv_list)
        SV2 = pca2.transform(SV)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(Z2_all[split_mask, 0], Z2_all[split_mask, 1],
                   c=y[split_mask], cmap="tab20", s=10, alpha=0.4)
        ax.scatter(SV2[:, 0], SV2[:, 1], c="red", s=18,
                   edgecolors="white", linewidths=0.3, label="Support vectors")
        ax.legend(loc="best")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Q5 – SVM ({kernel_tag}) – Support vectors in PCA(2)")
        fig.savefig(os.path.join(out_dir, f"q5_support_vectors_pca2_{kernel_tag}.png"),
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

# --------------------------- TOP-LEVEL EXECUTION (no main) ---------------------------

# 1) Load splits (TRAIN/VAL/TEST) as defined in the PDF. [1](https://2cq1sj-my.sharepoint.com/personal/kouklaki_serrenity_onmicrosoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/Pattern_Recognition_Course_Assignment.pdf)
train_df, val_df, _ = load_and_split(CSV_PATH)

# 2) Prepare features & labels
X_train = train_df[CONT_COLS + CAT_COLS]
y_train = train_df["killer_id"].values
X_val = val_df[CONT_COLS + CAT_COLS]
y_val = val_df["killer_id"].values

# 3) Build preprocessor and OvR SVM model
pre = make_preprocessor()

def build_ovr_svc(kernel: str, C, gamma, degree=None, coef0=None):
    if kernel == "rbf":
        base = SVC(kernel="rbf", C=C, gamma=gamma)
    else:
        base = SVC(kernel="poly", C=C, gamma=gamma,
                   degree=(degree or 2), coef0=(coef0 or 0.0))
    return OneVsRestClassifier(base)

# 4) Hyperparameter search on VAL (tiny manual grid, as per spec’s tuning request). [1](https://2cq1sj-my.sharepoint.com/personal/kouklaki_serrenity_onmicrosoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/Pattern_Recognition_Course_Assignment.pdf)
candidates = [(C_0, GAMMA_0, DEGREE_0 if KERNEL == "poly" else None,
               COEF0_0 if KERNEL == "poly" else None)]
if RUN_SMALL_SEARCH:
    for c in small_search_space(KERNEL):
        if c not in candidates:
            candidates.append(c)

best_model = None
best_tuple = None
best_acc = -np.inf

for (C, gamma, degree, coef0) in candidates:
    clf = build_ovr_svc(KERNEL, C, gamma, degree, coef0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    acc = accuracy_score(y_val, pred_val)
    if acc > best_acc:
        best_acc = acc
        best_model = pipe
        best_tuple = (C, gamma, degree, coef0)

print("Selected hyperparameters:")
if KERNEL == "rbf":
    print({"C": best_tuple[0], "gamma": best_tuple[1]})
else:
    print({"C": best_tuple[0], "gamma": best_tuple[1],
           "degree": best_tuple[2], "coef0": best_tuple[3]})

# 5) Final evaluation (VAL accuracy + confusion matrix)
acc, cm = fit_and_eval(best_model, X_train, y_train, X_val, y_val, KERNEL, FIGDIR)

# 6) Visualisations in PCA(2): decision regions + support vectors (requested by Q5). [1](https://2cq1sj-my.sharepoint.com/personal/kouklaki_serrenity_onmicrosoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/Pattern_Recognition_Course_Assignment.pdf)
# Fit preprocessor on TRAIN only (Pipeline did this internally; we refit pre for stand-alone transforms)
pre.fit(X_train)
clf_ovr = best_model.named_steps["clf"]

tv_df = pd.concat([train_df, val_df], ignore_index=True)
y_tv = tv_df["killer_id"].values
split_mask = tv_df["split"].values == "TRAIN"

plot_pca_and_support_vectors(tv_df, y_tv, split_mask, pre, clf_ovr, KERNEL, FIGDIR)

print("Saved figures to:", os.path.abspath(FIGDIR))