import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# -----------------------------
# 1) Φόρτωση & φιλτράρισμα
# -----------------------------
csv_path = "crimes.csv"
df = pd.read_csv(csv_path)

# Κρατάμε μόνο TRAIN+VAL
df = df[df["split"].isin(["TRAIN", "VAL"])].copy()

# Επιβεβαίωση ότι έχουμε όλες τις απαιτούμενες στήλες
required_cols = ["hour_float", "victim_age", "latitude", "longitude"]
missing = [c for c in required_cols if c not in df.columns]
assert not missing, f"Λείπουν στήλες: {missing}"

# -----------------------------
# 2) Συνάρτηση για ιστογράμματα
# -----------------------------
def plot_hist_with_stats(series, title, bins=30, kde=False, xlim=None):
    s = series.dropna().values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(s, bins=bins, stat="density", kde=kde, edgecolor="white", color="#4C78A8", ax=ax)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(title)
    ax.set_ylabel("Πυκνότητα")
    # Περιγραφικά στατιστικά
    mu, sigma = np.mean(s), np.std(s, ddof=1)
    ax.axvline(mu, color="crimson", linestyle="--", label=f"μ={mu:.2f}")
    ax.legend()
    plt.tight_layout()
    return fig, ax, mu, sigma

# -----------------------------
# 3) Ιστόγραμμα για τις 4 μεταβλητές
# -----------------------------
# hour_float: bins κάθε μισή ώρα
fig1, ax1, mu_h, sigma_h = plot_hist_with_stats(
    df["hour_float"], "hour_float", bins=np.linspace(0, 24, 49), xlim=(0, 24)
)
fig1.savefig("q1_hist_hour_float.png", dpi=150)

fig2, ax2, mu_a, sigma_a = plot_hist_with_stats(df["victim_age"], "victim_age", bins=30)
fig2.savefig("q1_hist_victim_age.png", dpi=150)

fig3, ax3, mu_lat, sigma_lat = plot_hist_with_stats(df["latitude"], "latitude", bins=30)
fig3.savefig("q1_hist_latitude.png", dpi=150)

fig4, ax4, mu_lon, sigma_lon = plot_hist_with_stats(df["longitude"], "longitude", bins=30)
fig4.savefig("q1_hist_longitude.png", dpi=150)

# -----------------------------
# 4) Μονο-Γκαουσιανή & GMM(3) στο hour_float
# -----------------------------
h = df["hour_float"].dropna().values.reshape(-1, 1)
# Μονο-Γκαουσιανή με MLE: χρήση δειγματικού μ & σ
mu = h.mean()
sigma = h.std(ddof=1)

# Πλέγμα για καμπύλες
grid = np.linspace(0, 24, 1000)
pdf_single = norm.pdf(grid, loc=mu, scale=sigma)

# GMM με 3 συστατικά
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(h)

# Συνολική πυκνότητα GMM στο grid
# Προσοχή: score_samples δίνει log p(x); exp -> p(x)
from sklearn.utils.validation import check_array
grid_col = grid.reshape(-1, 1)
logprob = gmm.score_samples(grid_col)
pdf_gmm = np.exp(logprob)

# (Προαιρετικά) επιμέρους καμπύλες των components
weights = gmm.weights_
means = gmm.means_.flatten()        # σχήμα (3,)
covars = gmm.covariances_.flatten() # αν full 1D, παίρνουμε 1 στοιχείο ανά component

component_pdfs = []
for w, m, c in zip(weights, means, covars):
    component_pdfs.append(w * norm.pdf(grid, loc=m, scale=np.sqrt(c)))

# Plot: ιστογράμματα + καμπύλες
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(h.flatten(), bins=np.linspace(0,24,49), stat="density", edgecolor="white",
             color="#4C78A8", alpha=0.6, ax=ax)
ax.plot(grid, pdf_single, color="crimson", lw=2, label=f"Μονο-Γκαουσιανή N(μ={mu:.2f}, σ={sigma:.2f})")
ax.plot(grid, pdf_gmm, color="#2CA02C", lw=2, label="GMM (3 συστατικά)")
times = ["Πρωινές ώρες", "Απογευματινές ώρες", "Βραδινές ώρες"]
# (Προαιρετικά) δείξε και τις επιμέρους καμπύλες
for j, comp_pdf in enumerate(component_pdfs):
    ax.plot(grid, comp_pdf, lw=1.5, linestyle="--", label=times[j])

ax.set_xlim(0, 24)
ax.set_xlabel("hour_float")
ax.set_ylabel("Πυκνότητα")
ax.set_title("hour_float: Ιστόγραμμα + Μονο-Γκαουσιανή + GMM(3)")
ax.legend(ncol=2)
plt.tight_layout()
fig.savefig("q1_hour_float_single_vs_gmm.png", dpi=150)

# -----------------------------
# 5) 2D scatter: hour_float vs longitude (χωρίς labels)
# -----------------------------
tmp = df[["hour_float", "longitude"]].dropna()
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(tmp["hour_float"], tmp["longitude"], s=10, alpha=0.3, color="#4C78A8")
ax.set_xlabel("hour_float")
ax.set_ylabel("longitude")
ax.set_xlim(0, 24)
ax.set_title("2D scatter: hour_float vs longitude (TRAIN+VAL, χωρίς labels)")
plt.tight_layout()
fig.savefig("q1_scatter_hour_vs_longitude.png", dpi=150)

# Επιβεβαίωση ότι τρέχει
print("ΟΚ: Δημιουργήθηκαν τα σχήματα Q1.")