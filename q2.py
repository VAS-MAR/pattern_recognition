import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv

csv_path = "crimes.csv"
df = pd.read_csv(csv_path)

train = df[df["split"].isin(["TRAIN"])].copy()

continuous_col = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density"
]

xc = train[continuous_col]
y = train["killer_id"]

killers = sorted(y.unique())
S = len(killers)

mu_dict = {}
cov_dict = {}

for k in killers:
    xk = xc[y == k].values
    Nk = xk.shape[0] # ari8mos deigmatwn

    # mean
    mu_k = xk.mean(axis=0)

    diff = xk - mu_k
    Sk = (diff.T @ diff)/Nk

    mu_dict[k] = mu_k
    cov_dict[k] = Sk

for k in killers:
    plt.figure(figsize=(7, 5))
    sns.heatmap(cov_dict[k], annot=False, cmap="coolwarm",
                xticklabels=continuous_col, yticklabels=continuous_col)
    plt.title(f"Covariance matrix Σ_k for killer {k}")
    plt.tight_layout()
    plt.savefig(f"q2_covariance_k{k}.png", dpi=150)
    plt.close()


feat1, feat2 = "hour_float", "longitude"

for k in killers:
    data = train[y == k][[feat1, feat2]].dropna().values
    mu2 = data.mean(axis=0)
    diff2 = data - mu2
    Sigma2 = (diff2.T @ diff2) / data.shape[0]

    # Mahalanobis distances
    invSigma2 = inv(Sigma2)
    D2 = np.array([d.T @ invSigma2 @ d for d in diff2])
    c_k = D2.max()

    # Grid for ellipse
    theta = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])  # unit circle

    # Transform unit circle to ellipse:
    # ellipse = μ + A * circle, where A satisfies A A^T = c_k Σ
    # Use Cholesky or sqrtm; here Cholesky if PD
    try:
        A = np.linalg.cholesky(c_k * Sigma2)
    except:
        # fallback
        from scipy.linalg import sqrtm
        A = sqrtm(c_k * Sigma2)

    ellipse = (mu2.reshape(2,1) + A @ circle)

    # 2D
    plt.figure(figsize=(6.5, 6))
    plt.scatter(data[:,0], data[:,1], s=10, alpha=0.4)
    plt.plot(ellipse[0], ellipse[1], 'r', lw=2)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(f"Killer {k}: ellipse in ({feat1}, {feat2})")
    plt.tight_layout()
    plt.savefig(f"q2_ellipse_{feat1}_{feat2}_k{k}.png", dpi=150)
    plt.close()

