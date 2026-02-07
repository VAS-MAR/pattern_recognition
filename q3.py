import numpy as np
import pandas as pd
from numpy.linalg import inv, det
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("crimes.csv")

# Continuous features
continuous_cols = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]

# TRAIN + VAL
train = df[df["split"]=="TRAIN"].copy()
val   = df[df["split"]=="VAL"].copy()

Xtrain = train[continuous_cols].values
ytrain = train["killer_id"].values

Xval   = val[continuous_cols].values
yval   = val["killer_id"].values

killers = sorted(np.unique(ytrain))
S = len(killers)

# Compute priors π_k
pi = {}
Nk_total = len(train)
for k in killers:
    Nk = np.sum(ytrain == k)
    pi[k] = Nk / Nk_total

# Use mu_dict, cov_dict from Q2
# If not already computed:
mu_dict = {}
cov_dict = {}
for k in killers:
    Xk = Xtrain[ytrain == k]
    mu_k = Xk.mean(axis=0)
    diff = Xk - mu_k
    Sigma_k = (diff.T @ diff) / Xk.shape[0]
    mu_dict[k] = mu_k
    cov_dict[k] = Sigma_k 

# Precompute inverses & log-determinants
invcov = {}
logdet = {}
for k in killers:
    Sigma = cov_dict[k]
    invcov[k] = inv(Sigma)
    logdet[k] = np.log(det(Sigma))

def log_gaussian(x, mu, Sigma_inv, logdet):
    diff = x - mu
    return -0.5 * (diff @ Sigma_inv @ diff) - 0.5 * logdet

def predict_proba(x):
    # returns vector of posterior probabilities for one x
    log_scores = []
    for k in killers:
        lg = np.log(pi[k]) + log_gaussian(
            x,
            mu_dict[k],
            inv_cov[k],
            logdet[k]
        )
        log_scores.append(lg)
    log_scores = np.array(log_scores)
    # softmax
    exp_scores = np.exp(log_scores - np.max(log_scores))
    return exp_scores / exp_scores.sum()

def predict_class(x):
    pp = predict_proba(x)
    return killers[np.argmax(pp)]

# Predict on TRAIN
ytrain_pred = np.array([predict_class(x) for x in Xtrain])
train_acc = accuracy_score(ytrain, ytrain_pred)

# Predict on VAL
yval_pred = np.array([predict_class(x) for x in Xval])
val_acc = accuracy_score(yval, yval_pred)

print("TRAIN accuracy:", train_acc)
print("VAL accuracy:", val_acc)
print("VAL confusion matrix:")
print(confusion_matrix(yval, yval_pred))

# ---------------------------------------
# Decision regions using PCA (2 components)
# ---------------------------------------
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)

pca = PCA(n_components=2)
Z = pca.fit_transform(Xtrain_scaled)

# Generate grid
x_min, x_max = Z[:,0].min()-1, Z[:,0].max()+1
y_min, y_max = Z[:,1].min()-1, Z[:,1].max()+1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Inverse PCA → original continuous space
grid_points = np.c_[xx.ravel(), yy.ravel()]
Xorig = pca.inverse_transform(grid_points)
Xorig = scaler.inverse_transform(Xorig)

preds = np.array([predict_class(x) for x in Xorig])

plt.figure(figsize=(7,6))
plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap="tab20")
plt.scatter(Z[:,0], Z[:,1], c=ytrain, s=12, cmap="tab20")
plt.title("Gaussian Bayes decision regions (2D PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()