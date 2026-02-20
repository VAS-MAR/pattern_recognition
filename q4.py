# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import q3

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0) Φόρτωση & προετοιμασία
df = pd.read_csv("crimes.csv")

# Συνεχή (Q3)
continuous_cols = [
    "hour_float","latitude","longitude","victim_age",
    "temp_c","humidity","dist_precinct_km","pop_density"
]
categorical_cols = ["weapon_code","scene_type","weather","vic_gender"]

# One-hot για Q4 (full features)

df_cat_full = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)
df_cat = df_cat_full[[c for c in df_cat_full.columns if any(c.startswith(p + "_") for p in categorical_cols)]]

full_cols = continuous_cols + list(df_cat.columns)

# Train/Val διαχωρισμός
train = df[df["split"]=="TRAIN"].copy()
val   = df[df["split"]=="VAL"].copy()

Xc_train = train[continuous_cols].values              # μόνο συνεχόμενα (Q3)
Xc_val   = val[continuous_cols].values

Xfull_train = np.hstack([Xc_train, df_cat.loc[train.index].values])  # όλα (Q4)
Xfull_val   = np.hstack([Xc_val,   df_cat.loc[val.index].values])

ytrain = train["killer_id"].values
yval   = val["killer_id"].values

killers = sorted(np.unique(ytrain))
S = len(killers)
label_to_idx = {k:i for i,k in enumerate(killers)}
ytrain_idx = np.array([label_to_idx[k] for k in ytrain])
yval_idx   = np.array([label_to_idx[k] for k in yval])

# 2) Q4: Linear Classifier (γραμμικό)
# Επιλογή 1 (σύμφωνη με εκφώνηση): PyTorch MSE + one-hot
# Αν δεν έχεις PyTorch διαθέσιμο, δες ακριβώς πιο κάτω "Επιλογή 2".

Xtr_t = torch.tensor(Xfull_train, dtype=torch.float32).to(device)  # (N, d_full)
Xva_t = torch.tensor(Xfull_val,   dtype=torch.float32).to(device)
ytr_1h = torch.eye(S, dtype=torch.float32)[torch.tensor(ytrain_idx)].to(device)

d_full = Xfull_train.shape[1]
linear = nn.Linear(d_full, S).to(device)
criterion = nn.MSELoss()
opt = optim.Adam(linear.parameters(), lr=1e-3)

linear.train()
for epoch in range(1500):
    opt.zero_grad()
    out = linear(Xtr_t)             # logits (S)
    loss = criterion(out, ytr_1h)   # SSE με one-hot
    loss.backward()
    opt.step()
    if epoch % 300 == 0:
        print(f"[Q4] epoch {epoch}, loss={loss.item():.4f}")

linear.eval()
with torch.no_grad():
    tr_logits = linear(Xtr_t).cpu().numpy()
    va_logits = linear(Xva_t).cpu().numpy()
tr_pred_idx = tr_logits.argmax(1)
va_pred_idx = va_logits.argmax(1)
print("Q4 TRAIN acc:", accuracy_score(ytrain_idx, tr_pred_idx))
print("Q4 VAL acc:",   accuracy_score(yval_idx,   va_pred_idx))
print("Q4 VAL confusion:\n", confusion_matrix(yval_idx, va_pred_idx))

def predict_class_linear(x_full):
    # x_full: (d_full,) numpy
    with torch.no_grad():
        t = torch.tensor(x_full[None, :], dtype=torch.float32).to(device)
        logits = linear(t).cpu().numpy()[0]
    return killers[np.argmax(logits)]

# 3) PCA(2) όπως στο Q3 (ΜΟΝΟ στα συνεχόμενα) — reuse “σκηνικό”
scaler = StandardScaler()
Xc_train_scaled = scaler.fit_transform(Xc_train)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xc_train_scaled)  # (N,2)

# grid στον 2D PCA χώρο
margin = 1.0
x_min, x_max = Z[:,0].min()-margin, Z[:,0].max()+margin
y_min, y_max = Z[:,1].min()-margin, Z[:,1].max()+margin
grid_res = 200
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, grid_res),
    np.linspace(y_min, y_max, grid_res)
)
grid_2d = np.c_[xx.ravel(), yy.ravel()]

# inverse PCA -> scaled continuous -> original continuous
Xgrid_cont_scaled = pca.inverse_transform(grid_2d)
Xgrid_cont = scaler.inverse_transform(Xgrid_cont_scaled)   # (G, 8)

# baseline για το κατηγορικό τμήμα του grid (Q4):
# μέσος όρος των one-hot στηλών στο TRAIN = συχνότητες κατηγοριών
cat_train = df_cat.loc[train.index].values  # (N, d_cat_onehot)
cat_baseline = cat_train.mean(axis=0)       # (d_cat_onehot,) - π.χ. [0.12, 0.30, ...]
# -> έτσι φτιάχνουμε πλήρες grid για Q4:
Xgrid_full = np.hstack([Xgrid_cont, np.tile(cat_baseline, (Xgrid_cont.shape[0], 1))])

# 4) Προβλέψεις στο grid για τα δύο μοντέλα
# Bayes (από συνεχόμενα)
preds_bayes = np.array([q3.predict_class(xc) for xc in Xgrid_cont])

# Linear (από full vector – continuous + cat_baseline)
preds_linear = np.array([predict_class_linear(xf) for xf in Xgrid_full])

# Για συνεπές colormap
preds_bayes_idx  = np.vectorize(label_to_idx.get)(preds_bayes)
preds_linear_idx = np.vectorize(label_to_idx.get)(preds_linear)
ytrain_idx_m     = ytrain_idx  # ήδη 0..S-1

cmap = plt.get_cmap('tab20', S)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, S+0.5, 1), ncolors=S)

# 5) Σχεδίαση overlay + ασυμφωνίες
fig, ax = plt.subplots(figsize=(8,6))

# Φόντο: Gaussian Bayes decision regions (όπως Q3)
Zgrid_bayes = preds_bayes_idx.reshape(xx.shape)
ax.contourf(xx, yy, Zgrid_bayes,
            levels=np.arange(-0.5, S+0.5, 1),
            cmap=cmap, norm=norm, alpha=0.30, zorder=1)

# Overlay: γραμμικά όρια απόφασης (Q4)
# Θα “τραβήξουμε” τις καμπύλες όπου αλλάζει η κλάση του linear
Zgrid_linear = preds_linear_idx.reshape(xx.shape)
# Σχεδιάζουμε τις ισο-γραμμές ορίου με λεπτές γραμμές:
cs = ax.contour(xx, yy, Zgrid_linear, levels=np.arange(-0.5, S+0.5, 1),
                colors='k', linewidths=0.8, alpha=0.9, zorder=3)
ax.clabel(cs, inline=1, fontsize=8, fmt=lambda v: "")

# Σημεία TRAIN (πραγματικές ετικέτες)
ax.scatter(Z[:,0], Z[:,1], c=ytrain_idx_m, cmap=cmap, norm=norm,
           s=14, edgecolor='k', linewidths=0.25, alpha=0.95, zorder=4)

ax.set_title("Q3 vs Q4: Overlay στο ίδιο 2D PCA")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

# Legend
handles = [Patch(facecolor=cmap(norm(i)), edgecolor='k', label=f"Killer {k}")
           for i,k in enumerate(killers)]
leg = ax.legend(handles=handles, title="Χρώμα → κλάση", loc='lower right',
                frameon=True, fontsize=9, title_fontsize=10)
leg.get_frame().set_alpha(0.9)

# Info box
info = (
    "Φόντο: Gaussian Bayes (Q3) από συνεχόμενα\n"
    "Μαύρες γραμμές: Linear (Q4) σε full vector (cont + one-hot baseline)\n"
    "Σημεία: TRAIN (πραγματικές ετικέτες)\n"

)
anch = AnchoredText(info, loc='upper left', prop={'size':9}, frameon=True)
anch.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
anch.patch.set_alpha(0.85)
ax.add_artist(anch)

plt.tight_layout()
plt.savefig("q4_overlay_on_q3_pca.png", dpi=160)



