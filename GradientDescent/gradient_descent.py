"""
gradient_descent.py
-------------------
A from-scratch look at gradient descent, using a species-area
relationship as the example.

We simulate noisy ecological data around a known true line
(species richness = 1.5 * area + 3), then use gradient descent to
*recover* that line starting from a random guess — watching the
Mean Squared Error (MSE) shrink as the fit improves.

Sections:
  1. Simulate noisy species-area data and plot it against the true line
  2. compute_mse()            — the loss function
  3. gradient_slope() / gradient_intercept() — the two partial derivatives
  4. Gradient descent loop    — random start, 1000 updates
  5. Side-by-side result: fitted vs true line, and loss over time
  6. MSE loss surface (contour) with true vs fitted parameters
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed so the noise and random start are reproducible.
# Remove or change this to see different runs.
np.random.seed(42)

# =============================================================================
# SECTION 1 — SIMULATE DATA AND PLOT AGAINST THE TRUE LINE
# =============================================================================

TRUE_SLOPE     = 1.5
TRUE_INTERCEPT = 3.0

# Habitat area from 1 to 20 (the predictor, x)
x = np.linspace(1, 20, 40)

# Species richness = true line + random ecological "noise"
noise = np.random.normal(0, 3.0, size=x.shape)   # mean 0, std 3
y = TRUE_SLOPE * x + TRUE_INTERCEPT + noise

# Plot the noisy data and the true underlying line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Simulated data (noisy)")
plt.plot(x, TRUE_SLOPE * x + TRUE_INTERCEPT, color="black",
         linewidth=2, label="True relationship")
plt.xlabel("Habitat Area", fontsize=12)
plt.ylabel("Species Richness", fontsize=12)
plt.title("Simulated Species-Area Data\n(true line: richness = 1.5 × area + 3)",
          fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("species_area_data.png", dpi=150)
print("Saved: species_area_data.png")


# =============================================================================
# SECTION 2 — THE LOSS FUNCTION: MEAN SQUARED ERROR
# =============================================================================

def compute_mse(slope, intercept, x, y):
    """Mean Squared Error between the line's predictions and the data.

    MSE = (1/n) * sum( (predicted - actual)^2 )
    """
    y_pred = slope * x + intercept
    return np.mean((y_pred - y) ** 2)


# =============================================================================
# SECTION 3 — THE GRADIENTS (PARTIAL DERIVATIVES OF MSE)
# =============================================================================
#
# Starting from MSE = (1/n) sum( (m*x + b - y)^2 ), calculus gives:
#
#   d(MSE)/d(slope)     = (2/n) * sum( (m*x + b - y) * x )
#   d(MSE)/d(intercept) = (2/n) * sum(  m*x + b - y       )
#
# Each gradient points "uphill"; gradient descent steps the opposite way.

def gradient_slope(slope, intercept, x, y):
    """Partial derivative of MSE with respect to the slope."""
    y_pred = slope * x + intercept
    return (2 / len(x)) * np.sum((y_pred - y) * x)


def gradient_intercept(slope, intercept, x, y):
    """Partial derivative of MSE with respect to the intercept."""
    y_pred = slope * x + intercept
    return (2 / len(x)) * np.sum(y_pred - y)


# =============================================================================
# SECTION 4 — GRADIENT DESCENT LOOP
# =============================================================================

LEARNING_RATE = 0.001
ITERATIONS    = 1000

# Start from a random guess for both parameters
slope     = np.random.randn()
intercept = np.random.randn()
print(f"\nRandom starting guess: slope = {slope:.3f}, intercept = {intercept:.3f}")

mse_history = []   # store the MSE after each update

for i in range(ITERATIONS):
    # Compute both gradients at the current slope/intercept
    g_slope     = gradient_slope(slope, intercept, x, y)
    g_intercept = gradient_intercept(slope, intercept, x, y)

    # Step downhill: move opposite the gradient, scaled by the learning rate
    slope     -= LEARNING_RATE * g_slope
    intercept -= LEARNING_RATE * g_intercept

    # Record the loss at this step
    mse_history.append(compute_mse(slope, intercept, x, y))

print(f"After {ITERATIONS} iterations:")
print(f"  fitted slope     = {slope:.3f}   (true = {TRUE_SLOPE})")
print(f"  fitted intercept = {intercept:.3f}   (true = {TRUE_INTERCEPT})")
print(f"  final MSE        = {mse_history[-1]:.3f}")


# =============================================================================
# SECTION 5 — SIDE-BY-SIDE RESULT: FIT vs TRUE, AND LOSS OVER TIME
# =============================================================================

fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

# -- Left: data, true line, and the gradient-descent fit ----------------------
ax_fit.scatter(x, y, color="steelblue", alpha=0.7, label="Data")
ax_fit.plot(x, TRUE_SLOPE * x + TRUE_INTERCEPT, color="black",
            linewidth=2, label="True fit")
ax_fit.plot(x, slope * x + intercept, color="tomato",
            linewidth=2, linestyle="--", label="Gradient descent fit")
ax_fit.set_xlabel("Habitat Area", fontsize=12)
ax_fit.set_ylabel("Species Richness", fontsize=12)
ax_fit.set_title("Fitted Line vs. True Line", fontsize=13)
ax_fit.legend(fontsize=10)
ax_fit.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# -- Right: MSE vs iteration --------------------------------------------------
ax_loss.plot(range(ITERATIONS), mse_history, color="mediumseagreen", linewidth=1.5)
ax_loss.set_xlabel("Iteration", fontsize=12)
ax_loss.set_ylabel("Mean Squared Error", fontsize=12)
ax_loss.set_title("Loss over time", fontsize=13)
ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("gradient_descent_result.png", dpi=150)
print("\nSaved: gradient_descent_result.png")
plt.show()


# =============================================================================
# SECTION 6 — MSE LOSS SURFACE (CONTOUR)
# =============================================================================
#
# Instead of watching MSE drop over iterations, here we map out the ENTIRE
# landscape: the MSE for every combination of slope and intercept on a grid.
# Gradient descent is just a ball rolling downhill on this surface. The true
# parameters sit at the lowest point (the minimum).

# Grid of candidate slope and intercept values
slope_values     = np.linspace(-1, 4, 200)
intercept_values = np.linspace(-2, 8, 200)

# meshgrid: S varies along the x-axis, I along the y-axis
S, I = np.meshgrid(slope_values, intercept_values)

# Compute MSE at every (slope, intercept) point on the grid
MSE_surface = np.zeros_like(S)
for row in range(S.shape[0]):
    for col in range(S.shape[1]):
        MSE_surface[row, col] = compute_mse(S[row, col], I[row, col], x, y)

# Filled contour: low MSE = yellow, high MSE = purple (viridis reversed)
plt.figure(figsize=(9, 7))
contour = plt.contourf(S, I, MSE_surface, levels=30, cmap="viridis_r")
plt.colorbar(contour, label="Mean Squared Error")

# True parameters: green star
plt.scatter([TRUE_SLOPE], [TRUE_INTERCEPT], marker="*", s=350,
            color="limegreen", edgecolors="black", linewidths=1.0,
            label="True parameters", zorder=5)

# Gradient descent result: blue X
plt.scatter([slope], [intercept], marker="X", s=160,
            color="blue", edgecolors="white", linewidths=1.0,
            label="Gradient descent result", zorder=5)

plt.xlabel("Slope", fontsize=12)
plt.ylabel("Intercept", fontsize=12)
plt.title("MSE Loss Surface\n(yellow = low error, purple = high error)", fontsize=13)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("mse_loss_surface.png", dpi=150)
print("Saved: mse_loss_surface.png")
plt.show()
