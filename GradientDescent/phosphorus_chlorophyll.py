"""
phosphorus_chlorophyll.py
-------------------------
The same gradient-descent demo as gradient_descent.py, but on a new
ecological scenario:

    x = phosphorus concentration   (0.1 to 5.0)
    y = chlorophyll concentration  (true law: chlorophyll = 2.0*P + 1.0)

We simulate noisy data, then use gradient descent to recover the line
from a random starting guess, reproducing the regression plot and the
loss landscape.

Note on the learning rate: this predictor (0.1-5.0) is on a smaller scale
than the species-area example (1-20). With the old rate of 0.001 the
intercept barely moves; a larger rate of 0.01 converges cleanly here.
Different feature scale -> different workable learning rate, which is one
more reason standardizing features is so handy.

Sections:
  1. Simulate noisy phosphorus-chlorophyll data and plot vs the true line
  2. compute_mse() and the two gradient functions
  3. Gradient descent loop (random start)
  4. Regression plot (data, true fit, gradient-descent fit) + loss over time
  5. MSE loss landscape with true (star) and fitted (X) parameters
  6. Standardizing phosphorus: tilted valley vs round bowl (with paths)
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)   # reproducible noise and starting guess

# =============================================================================
# SECTION 1 — SIMULATE DATA AND PLOT AGAINST THE TRUE LINE
# =============================================================================

TRUE_SLOPE     = 2.0
TRUE_INTERCEPT = 1.0

# Phosphorus concentration from 0.1 to 5.0 (the predictor, x)
x = np.linspace(0.1, 5.0, 40)

# Chlorophyll = true line + random ecological noise
noise = np.random.normal(0, 1.0, size=x.shape)
y = TRUE_SLOPE * x + TRUE_INTERCEPT + noise

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="seagreen", alpha=0.7, label="Simulated data (noisy)")
plt.plot(x, TRUE_SLOPE * x + TRUE_INTERCEPT, color="black",
         linewidth=2, label="True relationship")
plt.xlabel("Phosphorus Concentration", fontsize=12)
plt.ylabel("Chlorophyll Concentration", fontsize=12)
plt.title("Simulated Phosphorus-Chlorophyll Data\n"
          "(true line: chlorophyll = 2.0 × phosphorus + 1.0)", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("chlorophyll_data.png", dpi=150)
print("Saved: chlorophyll_data.png")


# =============================================================================
# SECTION 2 — LOSS FUNCTION AND GRADIENTS
# =============================================================================

def compute_mse(slope, intercept, x, y):
    """Mean Squared Error: (1/n) * sum((predicted - actual)^2)."""
    y_pred = slope * x + intercept
    return np.mean((y_pred - y) ** 2)


def gradient_slope(slope, intercept, x, y):
    """d(MSE)/d(slope) = (2/n) * sum((pred - y) * x)."""
    y_pred = slope * x + intercept
    return (2 / len(x)) * np.sum((y_pred - y) * x)


def gradient_intercept(slope, intercept, x, y):
    """d(MSE)/d(intercept) = (2/n) * sum(pred - y)."""
    y_pred = slope * x + intercept
    return (2 / len(x)) * np.sum(y_pred - y)


# =============================================================================
# SECTION 3 — GRADIENT DESCENT LOOP
# =============================================================================

LEARNING_RATE = 0.01    # larger than the species-area example (see header note)
ITERATIONS    = 1000

slope     = np.random.randn()
intercept = np.random.randn()
print(f"\nRandom starting guess: slope = {slope:.3f}, intercept = {intercept:.3f}")

mse_history = []
for i in range(ITERATIONS):
    g_slope     = gradient_slope(slope, intercept, x, y)
    g_intercept = gradient_intercept(slope, intercept, x, y)
    slope     -= LEARNING_RATE * g_slope
    intercept -= LEARNING_RATE * g_intercept
    mse_history.append(compute_mse(slope, intercept, x, y))

print(f"After {ITERATIONS} iterations:")
print(f"  fitted slope     = {slope:.3f}   (true = {TRUE_SLOPE})")
print(f"  fitted intercept = {intercept:.3f}   (true = {TRUE_INTERCEPT})")
print(f"  final MSE        = {mse_history[-1]:.3f}")


# =============================================================================
# SECTION 4 — REGRESSION PLOT + LOSS OVER TIME
# =============================================================================

fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

ax_fit.scatter(x, y, color="seagreen", alpha=0.7, label="Data")
ax_fit.plot(x, TRUE_SLOPE * x + TRUE_INTERCEPT, color="black",
            linewidth=2, label="True fit")
ax_fit.plot(x, slope * x + intercept, color="tomato",
            linewidth=2, linestyle="--", label="Gradient descent fit")
ax_fit.set_xlabel("Phosphorus Concentration", fontsize=12)
ax_fit.set_ylabel("Chlorophyll Concentration", fontsize=12)
ax_fit.set_title("Fitted Line vs. True Line", fontsize=13)
ax_fit.legend(fontsize=10)
ax_fit.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

ax_loss.plot(range(ITERATIONS), mse_history, color="mediumseagreen", linewidth=1.5)
ax_loss.set_xlabel("Iteration", fontsize=12)
ax_loss.set_ylabel("Mean Squared Error", fontsize=12)
ax_loss.set_title("Loss over time", fontsize=13)
ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("chlorophyll_descent_result.png", dpi=150)
print("\nSaved: chlorophyll_descent_result.png")


# =============================================================================
# SECTION 5 — MSE LOSS LANDSCAPE
# =============================================================================

slope_values     = np.linspace(-1, 5, 200)
intercept_values = np.linspace(-3, 5, 200)
S, I = np.meshgrid(slope_values, intercept_values)

MSE_surface = np.zeros_like(S)
for row in range(S.shape[0]):
    for col in range(S.shape[1]):
        MSE_surface[row, col] = compute_mse(S[row, col], I[row, col], x, y)

plt.figure(figsize=(9, 7))
contour = plt.contourf(S, I, MSE_surface, levels=30, cmap="viridis_r")
plt.colorbar(contour, label="Mean Squared Error")

plt.scatter([TRUE_SLOPE], [TRUE_INTERCEPT], marker="*", s=350,
            color="limegreen", edgecolors="black", linewidths=1.0,
            label="True parameters", zorder=5)
plt.scatter([slope], [intercept], marker="X", s=160,
            color="blue", edgecolors="white", linewidths=1.0,
            label="Gradient descent result", zorder=5)

plt.xlabel("Slope", fontsize=12)
plt.ylabel("Intercept", fontsize=12)
plt.title("MSE Loss Landscape: Phosphorus -> Chlorophyll\n"
          "(yellow = low error, purple = high error)", fontsize=13)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("chlorophyll_loss_surface.png", dpi=150)
print("Saved: chlorophyll_loss_surface.png")
plt.show()


# =============================================================================
# SECTION 6 — STANDARDIZING PHOSPHORUS: TILTED VALLEY vs ROUND BOWL
# =============================================================================
#
# Same lesson as the species-area example, confirmed on a second dataset.
# Standardizing the predictor (z-score) turns the tilted, elongated valley
# into a round bowl, and a round bowl lets a larger learning rate send the
# descent straight to the minimum.

def run_descent(xv, yv, lr, iterations, seed):
    np.random.seed(seed)
    m = np.random.randn()
    b = np.random.randn()
    m_path, b_path = [m], [b]
    for _ in range(iterations):
        m -= lr * gradient_slope(m, b, xv, yv)
        b -= lr * gradient_intercept(m, b, xv, yv)
        m_path.append(m)
        b_path.append(b)
    return m, b, np.array(m_path), np.array(b_path)

# Run 1: RAW phosphorus, the working rate for this scale (0.01)
m_raw, b_raw, mpath_raw, bpath_raw = run_descent(x, y, lr=0.01, iterations=1000, seed=1)

# Run 2: STANDARDIZED phosphorus, larger now-safe rate (0.1)
x_mean = x.mean()
x_std  = x.std()
x_z    = (x - x_mean) / x_std
m_z, b_z, mpath_z, bpath_z = run_descent(x_z, y, lr=0.1, iterations=1000, seed=1)

# Convert standardized fit back to original phosphorus scale
slope_recovered     = m_z / x_std
intercept_recovered = b_z - m_z * x_mean / x_std
print("\nStandardized-x fit, converted back to original scale:")
print(f"  recovered slope     = {slope_recovered:.3f}   (true = {TRUE_SLOPE})")
print(f"  recovered intercept = {intercept_recovered:.3f}   (true = {TRUE_INTERCEPT})")

# True parameters expressed in standardized space
slope_z_true     = TRUE_SLOPE * x_std
intercept_z_true = TRUE_SLOPE * x_mean + TRUE_INTERCEPT

def mse_grid(xv, yv, slope_vals, intercept_vals):
    Sg, Ig = np.meshgrid(slope_vals, intercept_vals)
    Z = np.zeros_like(Sg)
    for r in range(Sg.shape[0]):
        for c in range(Sg.shape[1]):
            Z[r, c] = compute_mse(Sg[r, c], Ig[r, c], xv, yv)
    return Sg, Ig, Z

# Raw surface (same window as Section 5)
S_raw, I_raw, Z_raw = mse_grid(
    x, y, np.linspace(-1, 5, 200), np.linspace(-3, 5, 200))

# Standardized surface, squared around its minimum so it reads round
half = 9
S_z, I_z, Z_z = mse_grid(
    x_z, y,
    np.linspace(slope_z_true - half, slope_z_true + half, 200),
    np.linspace(intercept_z_true - half, intercept_z_true + half, 200))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

# Left: raw, tilted valley
axL.contourf(S_raw, I_raw, Z_raw, levels=30, cmap="viridis_r")
axL.plot(mpath_raw, bpath_raw, color="white", linewidth=1.5, alpha=0.9)
axL.scatter([mpath_raw[0]], [bpath_raw[0]], color="white", s=60,
            edgecolors="black", zorder=5, label="start")
axL.scatter([TRUE_SLOPE], [TRUE_INTERCEPT], marker="*", s=350,
            color="limegreen", edgecolors="black", zorder=6, label="True parameters")
axL.scatter([m_raw], [b_raw], marker="X", s=160, color="blue",
            edgecolors="white", zorder=6, label="Descent result")
axL.set_title("RAW phosphorus  (learning rate 0.01)\ntilted valley", fontsize=12)
axL.set_xlabel("Slope", fontsize=11)
axL.set_ylabel("Intercept", fontsize=11)
axL.legend(fontsize=9, loc="upper right")

# Right: standardized, round bowl
axR.contourf(S_z, I_z, Z_z, levels=30, cmap="viridis_r")
axR.plot(mpath_z, bpath_z, color="white", linewidth=1.5, alpha=0.9)
axR.scatter([mpath_z[0]], [bpath_z[0]], color="white", s=60,
            edgecolors="black", zorder=5, label="start")
axR.scatter([slope_z_true], [intercept_z_true], marker="*", s=350,
            color="limegreen", edgecolors="black", zorder=6, label="True parameters")
axR.scatter([m_z], [b_z], marker="X", s=160, color="blue",
            edgecolors="white", zorder=6, label="Descent result")
axR.set_aspect("equal")
axR.set_title("STANDARDIZED phosphorus  (learning rate 0.1)\nround bowl", fontsize=12)
axR.set_xlabel("Slope (standardized space)", fontsize=11)
axR.set_ylabel("Intercept (standardized space)", fontsize=11)
axR.legend(fontsize=9, loc="upper right")

fig.suptitle("Standardizing the Predictor — Confirmed on the Chlorophyll Data", fontsize=14)
plt.tight_layout()
plt.savefig("chlorophyll_standardize_comparison.png", dpi=150)
print("\nSaved: chlorophyll_standardize_comparison.png")
plt.show()
