"""
analysis.py
-----------
Analysis pipeline for the rodent species summary.

This file is organized into numbered sections. Each new analysis
step gets appended as its own section below, so the whole analysis
runs top-to-bottom in one go. It reads the checkpoint CSV produced
by species_summary.py and works from there.

  Section 1 : Load data
  Section 2 : Z-score standardize trait features
  Section 3 : K-means clustering (k=3)
  Section 4 : Cluster scatter plot
  Section 5 : Interpret clusters with common names
  Section 6 : Elbow plot (inertia vs. k)
  Section 7 : Three-feature clustering (adds n_plots)
  Section 8 : Compare 2-feature vs 3-feature clustering
  Section 9 : Table of species that changed grouping
  Section 10: KNN size-class classification (leave-one-out CV)
  Section 11: KNN from hindfoot length only (non-circular test)
  Section 12: KNN decision boundary plot
  Section 13: Final branded summary table
  (more sections will be added as the analysis grows)

Input  : rodent_species_summary.csv
Output : cluster_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================

INPUT_CSV = "rodent_species_summary.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} species.\n")

# Display options so wide tables aren't truncated in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# =============================================================================
# SECTION 2 — Z-SCORE STANDARDIZE TRAIT FEATURES
# =============================================================================
#
# z = (value - mean) / standard_deviation
#
# After standardizing, each column has mean 0 and std 1, putting both
# traits on a comparable scale for clustering, PCA, or distance methods.

FEATURES = ["mean_weight", "mean_hindfoot_length"]

# New dataframe; keep species name as a label column for reference
features_scaled = pd.DataFrame()
features_scaled["species_name"] = df["species_name"]

print("Standardizing features:")
for col in FEATURES:
    mean = df[col].mean()
    std  = df[col].std()            # pandas sample std (ddof=1) by default
    features_scaled[col] = (df[col] - mean) / std
    print(f"  {col}: original mean = {mean:.3f}, original std = {std:.3f}")

# Verify: each scaled column should have ~0 mean and ~1 std
print("\nAfter standardization (should be ~0 mean, ~1 std):")
for col in FEATURES:
    print(f"  {col}: mean = {features_scaled[col].mean():.6f}, "
          f"std = {features_scaled[col].std():.6f}")

print("\nFirst 5 rows of features_scaled:")
print("-" * 60)
print(features_scaled.head().to_string(index=False))


# =============================================================================
# SECTION 3 — K-MEANS CLUSTERING (k=3)
# =============================================================================
#
# Group species into 3 clusters based on their standardized traits.
# random_state=42 makes the result reproducible across runs.

K = 3

# Use only the scaled numeric trait columns (not the species_name label)
X = features_scaled[FEATURES]

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Record cluster assignment on the main species summary dataframe
df["kmeans_cluster"] = cluster_labels

print(f"\n\nK-means clustering with k={K}:")
print("=" * 60)

# Print the species belonging to each cluster
for cluster_id in range(K):
    members = df[df["kmeans_cluster"] == cluster_id]
    print(f"\nCluster {cluster_id}  ({len(members)} species):")
    print("-" * 60)
    for _, row in members.iterrows():
        print(f"  {row['species_name']:<30}"
              f"weight={row['mean_weight']:>7.2f} g   "
              f"hindfoot={row['mean_hindfoot_length']:>6.2f} mm")


# =============================================================================
# SECTION 4 — CLUSTER SCATTER PLOT
# =============================================================================
#
# Scatter of the two standardized traits, each species colored by its
# K-means cluster and labeled with its species_id code.

OUTPUT_PNG = "cluster_scatter.png"

fig, ax = plt.subplots(figsize=(9, 7))

# A distinct color per cluster
colors = ["tomato", "steelblue", "mediumseagreen", "goldenrod", "orchid"]

for cluster_id in range(K):
    members = df[df["kmeans_cluster"] == cluster_id]
    ax.scatter(
        features_scaled.loc[members.index, "mean_weight"],
        features_scaled.loc[members.index, "mean_hindfoot_length"],
        s=80,
        color=colors[cluster_id % len(colors)],
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
        label=f"Cluster {cluster_id}",
    )

# Label each point with its species_id code
for idx, row in df.iterrows():
    ax.annotate(
        row["species_id"],
        xy=(features_scaled.loc[idx, "mean_weight"],
            features_scaled.loc[idx, "mean_hindfoot_length"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8,
        color="black",
    )

ax.set_xlabel("Standardized Mean Weight (z-score)", fontsize=12)
ax.set_ylabel("Standardized Mean Hindfoot Length (z-score)", fontsize=12)
ax.set_title("Rodent Species Clustered by Body Traits\n(K-means, k=3)", fontsize=13)
ax.legend(title="Cluster", fontsize=10)

# Reference lines at 0 (the mean of each standardized trait)
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"\n\nCluster scatter plot saved: {OUTPUT_PNG}")
plt.show()


# =============================================================================
# SECTION 5 — INTERPRET CLUSTERS WITH COMMON NAMES
# =============================================================================
#
# The Portal dataset stores only scientific names, so common names are
# supplied here as a lookup keyed on species_id. Any code not in the
# dictionary falls back to a placeholder so nothing breaks.
#
# Note: these are standard common names for these North American rodents.
# Double-check any that look off for your purposes — taxonomy and common
# usage can vary slightly between sources.

COMMON_NAMES = {
    "BA": "Northern pygmy mouse",
    "DM": "Merriam's kangaroo rat",
    "DO": "Ord's kangaroo rat",
    "DS": "Banner-tailed kangaroo rat",
    "NL": "White-throated woodrat",
    "OL": "Northern grasshopper mouse",
    "OT": "Southern grasshopper mouse",
    "PB": "Bailey's pocket mouse",
    "PE": "Cactus mouse",
    "PF": "Silky pocket mouse",
    "PH": "Hispid pocket mouse",
    "PI": "Rock pocket mouse",
    "PL": "White-footed mouse",
    "PM": "Deer mouse",
    "PP": "Desert pocket mouse",
    "RF": "Fulvous harvest mouse",
    "RM": "Western harvest mouse",
    "RO": "Plains harvest mouse",
    "SF": "Tawny-bellied cotton rat",
    "SH": "Hispid cotton rat",
    "SO": "Yellow-nosed cotton rat",
    "SS": "Spotted ground squirrel",
    "ST": "Round-tailed ground squirrel",
}

# Attach common names; unknown codes get a clear placeholder
df["common_name"] = df["species_id"].map(COMMON_NAMES).fillna("(common name not listed)")

# Reference table, grouped by cluster
print("\n\nCluster reference table:")
print("=" * 78)
for cluster_id in range(K):
    members = df[df["kmeans_cluster"] == cluster_id]

    # A plain-language trait profile for the cluster
    avg_w = members["mean_weight"].mean()
    avg_h = members["mean_hindfoot_length"].mean()

    print(f"\nCluster {cluster_id}  —  avg weight {avg_w:.1f} g, "
          f"avg hindfoot {avg_h:.1f} mm  ({len(members)} species)")
    print("-" * 78)
    print(f"  {'ID':<4}{'Scientific name':<28}{'Common name':<28}")
    print("  " + "-" * 60)
    for _, row in members.iterrows():
        print(f"  {row['species_id']:<4}{row['species_name']:<28}{row['common_name']:<28}")


# =============================================================================
# SECTION 6 — ELBOW PLOT (INERTIA vs. k)
# =============================================================================
#
# Run K-means for several values of k and record the inertia (the sum of
# squared distances from each point to its assigned cluster centre).
# Inertia always falls as k rises, so we look for the "elbow" — the point
# where adding another cluster stops buying much improvement.

K_VALUES = [2, 3, 4, 5]
inertias = []

print("\n\nElbow analysis:")
print("-" * 30)
for k in K_VALUES:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)                       # X = scaled trait columns, from Section 3
    inertias.append(km.inertia_)
    print(f"  k = {k}: inertia = {km.inertia_:.2f}")

# Line plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(K_VALUES, inertias, marker="o", markersize=8,
        color="steelblue", linewidth=1.5)

ax.set_xlabel("Number of clusters (k)", fontsize=12)
ax.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=12)
ax.set_title("Elbow Plot for K-means Clustering\n(Rodent Body Traits)", fontsize=13)
ax.set_xticks(K_VALUES)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150)
print("\nElbow plot saved: elbow_plot.png")
plt.show()


# =============================================================================
# SECTION 7 — THREE-FEATURE CLUSTERING (adds n_plots)
# =============================================================================
#
# Same idea as Section 2-3, but now with a third feature: n_plots, the
# number of distinct plots a species was observed in (a measure of how
# widespread it is). All three features are standardized so each carries
# equal weight, then we cluster again with k=3.

FEATURES_3 = ["mean_weight", "mean_hindfoot_length", "n_plots"]

# Build a fresh standardized table for the three features
features_scaled_3 = pd.DataFrame()
features_scaled_3["species_name"] = df["species_name"]

print("\n\nStandardizing three features:")
for col in FEATURES_3:
    mean = df[col].mean()
    std  = df[col].std()            # pandas sample std (ddof=1)
    features_scaled_3[col] = (df[col] - mean) / std
    print(f"  {col}: original mean = {mean:.3f}, original std = {std:.3f}")

# Cluster on the three standardized features
X3 = features_scaled_3[FEATURES_3]
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df["kmeans_cluster_3features"] = kmeans_3.fit_predict(X3)

# Print species in each new cluster, with common names for interpretation
print(f"\nThree-feature K-means clustering (k=3):")
print("=" * 72)
for cluster_id in range(3):
    members = df[df["kmeans_cluster_3features"] == cluster_id]
    avg_w = members["mean_weight"].mean()
    avg_h = members["mean_hindfoot_length"].mean()
    avg_p = members["n_plots"].mean()
    print(f"\nCluster {cluster_id}  —  avg weight {avg_w:.1f} g, "
          f"hindfoot {avg_h:.1f} mm, plots {avg_p:.1f}  ({len(members)} species)")
    print("-" * 72)
    for _, row in members.iterrows():
        print(f"  {row['species_id']:<4}{row['species_name']:<28}"
              f"{row['common_name']:<26}plots={row['n_plots']}")


# =============================================================================
# SECTION 8 — COMPARE 2-FEATURE vs 3-FEATURE CLUSTERING
# =============================================================================
#
# Two panels share the same axes (standardized weight vs hindfoot length).
#   Left  : points colored by the original 2-feature clustering
#   Right : points colored by the 3-feature clustering, with point SIZE
#           scaled to n_plots so you can see the dimension that was added.
# Species that change color between the panels are the ones that regrouped.

colors = ["tomato", "steelblue", "mediumseagreen", "goldenrod", "orchid"]

# Shared coordinates (weight & hindfoot standardize identically in both)
xs = features_scaled_3["mean_weight"]
ys = features_scaled_3["mean_hindfoot_length"]

# Point sizes for the right panel, scaled from n_plots
sizes = 40 + (df["n_plots"] / df["n_plots"].max()) * 240

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)

# -- Left panel: 2-feature clustering -----------------------------------------
for cluster_id in sorted(df["kmeans_cluster"].unique()):
    m = df[df["kmeans_cluster"] == cluster_id]
    ax_left.scatter(xs.loc[m.index], ys.loc[m.index],
                    s=80, color=colors[cluster_id % len(colors)],
                    alpha=0.8, edgecolors="black", linewidths=0.5,
                    label=f"Cluster {cluster_id}")
ax_left.set_title("2 features\n(weight + hindfoot)", fontsize=12)

# -- Right panel: 3-feature clustering, sized by n_plots ----------------------
for cluster_id in sorted(df["kmeans_cluster_3features"].unique()):
    m = df[df["kmeans_cluster_3features"] == cluster_id]
    ax_right.scatter(xs.loc[m.index], ys.loc[m.index],
                     s=sizes.loc[m.index], color=colors[cluster_id % len(colors)],
                     alpha=0.8, edgecolors="black", linewidths=0.5,
                     label=f"Cluster {cluster_id}")
ax_right.set_title("3 features\n(weight + hindfoot + n_plots)\npoint size = n_plots", fontsize=12)

# Label every point with its species_id in both panels, and shared styling
for ax in (ax_left, ax_right):
    for idx, row in df.iterrows():
        ax.annotate(row["species_id"], xy=(xs.loc[idx], ys.loc[idx]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Standardized Mean Weight (z-score)", fontsize=11)
    ax.legend(title="Cluster", fontsize=9)

ax_left.set_ylabel("Standardized Mean Hindfoot Length (z-score)", fontsize=11)

fig.suptitle("How Adding 'Number of Plots' Changes the Clustering", fontsize=14)
plt.tight_layout()
plt.savefig("cluster_comparison.png", dpi=150)
print("\n\nComparison plot saved: cluster_comparison.png")
plt.show()


# =============================================================================
# SECTION 9 — TABLE OF SPECIES THAT CHANGED GROUPING
# =============================================================================
#
# Cluster ID numbers are arbitrary between the two runs, so we can't just
# compare cluster labels. Instead, for each species we compare its set of
# CLUSTER-MATES (the other species sharing its cluster) before and after.
# If that set changed, the species' grouping genuinely changed.

def mate_sets(cluster_col):
    """Map each species_id -> set of its cluster-mates (excluding itself)."""
    mates = {}
    for cid in df[cluster_col].unique():
        members = set(df[df[cluster_col] == cid]["species_id"])
        for sid in members:
            mates[sid] = members - {sid}
    return mates

mates_before = mate_sets("kmeans_cluster")            # 2 features
mates_after  = mate_sets("kmeans_cluster_3features")  # 3 features

# Find species whose cluster-mate set changed
changed = []
for _, row in df.iterrows():
    sid    = row["species_id"]
    before = mates_before[sid]
    after  = mates_after[sid]
    if before != after:
        left   = before - after   # mates it no longer shares a cluster with
        joined = after - before    # mates it now shares a cluster with
        changed.append((sid, row["common_name"], left, joined))

print("\n\nSpecies that changed grouping when n_plots was added:")
print("=" * 84)

if not changed:
    print("  No species changed grouping — the two clusterings are identical.")
else:
    print(f"  {len(changed)} of {len(df)} species regrouped.\n")
    print(f"  {'ID':<4}{'Common name':<26}{'Left behind':<24}{'Newly grouped with':<24}")
    print("  " + "-" * 76)
    for sid, common, left, joined in changed:
        left_str   = ", ".join(sorted(left))   if left   else "—"
        joined_str = ", ".join(sorted(joined)) if joined else "—"
        print(f"  {sid:<4}{common:<26}{left_str:<24}{joined_str:<24}")


# =============================================================================
# SECTION 10 — KNN SIZE-CLASS CLASSIFICATION (LEAVE-ONE-OUT CV)
# =============================================================================
#
# Label each species "small" (below median weight) or "large" (at/above
# median), then test how well KNN (k=5) predicts that label from the two
# standardized traits, using leave-one-out cross-validation.
#
# Standardization is wrapped in a Pipeline so it is re-fit on the training
# species inside each CV fold — this avoids letting the held-out species
# influence the scaling (a subtle but real form of information leakage).

# -- Create the size_class label ----------------------------------------------
median_weight = df["mean_weight"].median()
df["size_class"] = (df["mean_weight"] >= median_weight).map({True: "large", False: "small"})
print(f"\n\nMedian weight = {median_weight:.2f} g")
print(f"  small (< median) : {(df['size_class'] == 'small').sum()} species")
print(f"  large (>= median): {(df['size_class'] == 'large').sum()} species")

# -- Features and target ------------------------------------------------------
X_knn = df[FEATURES]          # FEATURES = ["mean_weight", "mean_hindfoot_length"]
y_knn = df["size_class"]

# -- KNN with leave-one-out cross-validation ----------------------------------
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
loo  = LeaveOneOut()

# cross_val_predict returns each species' prediction from the fold where it
# was the held-out test point
y_pred = cross_val_predict(pipe, X_knn, y_knn, cv=loo)

accuracy = accuracy_score(y_knn, y_pred)
n_correct = (y_knn.values == y_pred).sum()

print(f"\nKNN (k=5) leave-one-out cross-validation:")
print("=" * 70)
print(f"Overall accuracy: {accuracy:.3f}  ({n_correct}/{len(df)} correct)")

# -- Predicted vs actual for each species -------------------------------------
# Sort by weight so the boundary region (near the median) is easy to see
order = df["mean_weight"].sort_values().index

print(f"\n  {'ID':<4}{'Common name':<26}{'Weight':>8}  {'Actual':<8}{'Predicted':<10}{'Result'}")
print("  " + "-" * 66)
for idx in order:
    actual    = y_knn.loc[idx]
    predicted = y_pred[df.index.get_loc(idx)]
    result    = "correct" if actual == predicted else "*** MISS ***"
    print(f"  {df.loc[idx, 'species_id']:<4}{df.loc[idx, 'common_name']:<26}"
          f"{df.loc[idx, 'mean_weight']:>8.1f}  {actual:<8}{predicted:<10}{result}")


# =============================================================================
# SECTION 11 — KNN FROM HINDFOOT LENGTH ONLY (NON-CIRCULAR TEST)
# =============================================================================
#
# size_class is defined from weight, so using weight as a predictor is
# partly circular. Here we predict size_class from hindfoot length ALONE.
# This asks a real question: is foot size a good proxy for body size?

acc_both_features = accuracy   # captured from Section 10 (weight + hindfoot)

X_hf = df[["mean_hindfoot_length"]]   # single feature
y_hf = df["size_class"]

pipe_hf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
y_pred_hf = cross_val_predict(pipe_hf, X_hf, y_hf, cv=LeaveOneOut())

acc_hf = accuracy_score(y_hf, y_pred_hf)
n_correct_hf = (y_hf.values == y_pred_hf).sum()

print(f"\n\nKNN (k=5) from HINDFOOT LENGTH ONLY, leave-one-out CV:")
print("=" * 70)
print(f"Overall accuracy: {acc_hf:.3f}  ({n_correct_hf}/{len(df)} correct)")
print(f"\nComparison of the two classifiers:")
print(f"  weight + hindfoot : {acc_both_features:.3f}")
print(f"  hindfoot only     : {acc_hf:.3f}")

order = df["mean_hindfoot_length"].sort_values().index
print(f"\n  {'ID':<4}{'Common name':<26}{'Hindfoot':>9}  {'Actual':<8}{'Predicted':<10}{'Result'}")
print("  " + "-" * 68)
for idx in order:
    actual    = y_hf.loc[idx]
    predicted = y_pred_hf[df.index.get_loc(idx)]
    result    = "correct" if actual == predicted else "*** MISS ***"
    print(f"  {df.loc[idx, 'species_id']:<4}{df.loc[idx, 'common_name']:<26}"
          f"{df.loc[idx, 'mean_hindfoot_length']:>9.1f}  {actual:<8}{predicted:<10}{result}")


# =============================================================================
# SECTION 12 — KNN DECISION BOUNDARY PLOT
# =============================================================================
#
# Shade the trait space by which size_class a k=5 KNN predicts, then overlay
# the actual species colored by their true label. This classifier is fit on
# all species (not cross-validated) purely to draw the boundary — it shows
# what the model "sees", not an unbiased accuracy estimate.

# Standardized coordinates (reuse the 2-feature scaled table from Section 2)
Xs = features_scaled[["mean_weight", "mean_hindfoot_length"]].values
y_true = df["size_class"].values

# Fit a KNN on the full standardized data for the boundary
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(Xs, y_true)

# Build a mesh grid covering the trait space
pad = 0.5
x_min, x_max = Xs[:, 0].min() - pad, Xs[:, 0].max() + pad
y_min, y_max = Xs[:, 1].min() - pad, Xs[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predict the class for every grid point, then map to 0/1 for shading
grid_pred = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (grid_pred == "large").astype(int).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9, 7))

# Background regions: light blue = small, light red = large
bg_cmap = ListedColormap(["#cfe3f7", "#f7d4cc"])
ax.contourf(xx, yy, Z, alpha=0.5, cmap=bg_cmap, levels=[-0.5, 0.5, 1.5])

# Actual species points colored by true label
for label, color in [("small", "steelblue"), ("large", "tomato")]:
    m = (df["size_class"] == label).values
    ax.scatter(Xs[m, 0], Xs[m, 1], c=color, s=90,
               edgecolors="black", linewidths=0.5, label=f"actual: {label}")

# Label each point with species_id
for i, idx in enumerate(df.index):
    ax.annotate(df.loc[idx, "species_id"], xy=(Xs[i, 0], Xs[i, 1]),
                xytext=(5, 4), textcoords="offset points", fontsize=8)

ax.set_xlabel("Standardized Mean Weight (z-score)", fontsize=12)
ax.set_ylabel("Standardized Mean Hindfoot Length (z-score)", fontsize=12)
ax.set_title("KNN Decision Boundary for Size Class (k=5)\n"
             "Shaded = predicted region, points = true label", fontsize=13)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig("knn_decision_boundary.png", dpi=150)
print("\n\nDecision boundary plot saved: knn_decision_boundary.png")
plt.show()


# =============================================================================
# SECTION 13 — FINAL BRANDED SUMMARY TABLE
# =============================================================================
#
# Pull together species_id, k-means cluster, true size_class, and the KNN
# predicted label (from the weight + hindfoot model in Section 10) into one
# table, sorted by cluster. Printed in branded colors (maroon / gold / slate)
# and also saved as a styled image.

# Build the summary dataframe (common_name included for easy reading)
summary_final = df[["species_id", "common_name", "kmeans_cluster", "size_class"]].copy()
summary_final["knn_predicted"] = y_pred          # Section 10 predictions, in df order
summary_final = summary_final.sort_values("kmeans_cluster").reset_index(drop=True)

# -- Terminal print in branded colors (24-bit ANSI) ---------------------------
# These show up colored in PyCharm's Run console and most modern terminals.
MAROON = "\033[38;2;122;0;25m"
GOLD   = "\033[38;2;184;134;11m"   # dark goldenrod, readable on light or dark
SLATE  = "\033[38;2;112;128;144m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

print(f"\n\n{BOLD}{MAROON}Rodent Classification Summary (sorted by k-means cluster){RESET}")
print(f"{BOLD}{MAROON}{'ID':<5}{'Common name':<26}{'Cluster':<9}{'Size':<8}{'KNN pred':<10}{RESET}")
print(f"{SLATE}{'-' * 58}{RESET}")
for _, row in summary_final.iterrows():
    print(f"{row['species_id']:<5}{row['common_name']:<26}"
          f"{SLATE}{str(row['kmeans_cluster']):<9}{RESET}"
          f"{GOLD}{row['size_class']:<8}{RESET}"
          f"{MAROON}{row['knn_predicted']:<10}{RESET}")

# -- Styled branded table image -----------------------------------------------
BRAND_MAROON = "#7a0019"
BRAND_GOLD   = "#ffc72c"
BRAND_SLATE  = "#708090"
LIGHT_SLATE  = "#e7eaed"

fig, ax = plt.subplots(figsize=(9, 0.42 * len(summary_final) + 1.2))
ax.axis("off")

col_labels = ["species_id", "common_name", "kmeans_cluster", "size_class", "knn_predicted"]
table = ax.table(cellText=summary_final.values, colLabels=col_labels,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(BRAND_SLATE)
    if r == 0:                                  # header row
        cell.set_facecolor(BRAND_MAROON)
        cell.set_text_props(color=BRAND_GOLD, fontweight="bold")
    else:                                       # data rows: slate striping
        cell.set_facecolor("white" if r % 2 else LIGHT_SLATE)

ax.set_title("Rodent Classification Summary",
             color=BRAND_MAROON, fontsize=15, fontweight="bold", pad=18)

plt.tight_layout()
plt.savefig("summary_table.png", dpi=150, bbox_inches="tight")
print(f"\nBranded summary table saved: summary_table.png")
plt.show()


# =============================================================================
# (NEXT SECTION WILL GO HERE)
# =============================================================================
