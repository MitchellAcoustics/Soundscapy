"""
Demonstration notebook for the Multivariate Skew-Normal Distribution.

This script can be converted to a Jupyter Notebook (.ipynb) using tools like jupytext:
`jupytext --to notebook MultivariateSkewNormalDemo.py`
Or opened directly in VS Code.
"""
# %%
# %% [markdown]
# # Multivariate Skew-Normal Distribution Demonstration
#
# This notebook demonstrates the usage of the `multivariate_skew_normal` distribution implemented in `soundscapy.spi.multivariate_skewnorm`.
#
# We will cover:
# 1. Importing and creating an instance.
# 2. Evaluating PDF, logPDF, CDF, and logCDF.
# 3. Generating random samples.
# 4. Calculating moments (mean, covariance, variance).
# 5. Parameter conversions (DP <-> CP).
# 6. Fitting the distribution to data using Maximum Likelihood Estimation (MLE).
# 7. Comparison with the standard Multivariate Normal distribution.

# %% [code]
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal  # For comparison

from soundscapy.spi.multivariate_skewnorm_commit import multivariate_skewnormal

# Set random seed for reproducibility
RNG = np.random.default_rng(12345)

# %% [markdown]
# ## 1. Creating a Multivariate Skew-Normal Instance
#
# We define the Direct Parameters (DP):
# - `loc` (ξ): Location vector
# - `scale` (Ω): Scale matrix (symmetric positive semi-definite)
# - `shape` (α): Shape vector

# %% [code]
# Define parameters (DP)
xi = np.array([0.06, 0.597])  # Location
Omega = np.array([[0.15, -0.058], [-0.058, 0.093]])  # Scale matrix
alpha = np.array([0.868, -0.561])  # Shape vector

# Create a frozen instance of the distribution
msn_frozen = multivariate_skewnormal(loc=xi, cov=Omega, skew=alpha)

print(f"Distribution Dimension: {msn_frozen.dim}")
print(f"Location (xi): {msn_frozen.loc}")
print(f"Covariance Matrix (Omega):\n{msn_frozen.cov}")
print(f"Shape (alpha): {msn_frozen.skew}")

# %% [markdown]
# ## 2. Evaluating PDF and CDF
#
# We can evaluate the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) at specific points. We can also compute their logarithms (`logpdf`, `logcdf`).

# %% [code]
# Point(s) to evaluate
x_point = np.array([0.5, 0.0])
x_multiple = np.array([[0.5, 0.0], [0.3, -0.5], [-0.2, 0.2]])

# Using the frozen instance
pdf_val = msn_frozen.pdf(x_point)
logpdf_val = msn_frozen.logpdf(x_point)
cdf_val = msn_frozen.cdf(x_point)  # CDF calculation can be slow
logcdf_val = msn_frozen.logcdf(x_point)

print(f"--- Evaluation at {x_point} ---")
print(f"PDF: {pdf_val:.5f}")
print(f"logPDF: {logpdf_val:.5f}")
print(f"CDF: {cdf_val:.5f}")
print(f"logCDF: {logcdf_val:.5f}")

# Using the generator directly with parameters
pdf_multi = multivariate_skewnormal.pdf(x_multiple, loc=xi, cov=Omega, skew=alpha)
print("\n--- PDF Evaluation at multiple points ---")
print(pdf_multi)

# %% [markdown]
# ## 3. Generating Random Samples
#
# Use the `rvs` method to draw random samples from the distribution.

# %% [code]
# Generate random samples
n_samples = 1000
samples = msn_frozen.rvs(size=n_samples, random_state=RNG)

print(f"Shape of generated samples: {samples.shape}")
print(f"First 5 samples:\n{samples[:5]}")

# Plot the samples
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Blues", fill=True, alpha=0.8)
plt.title(f"{n_samples} Samples from Multivariate Skew-Normal")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, linestyle="--", alpha=0.6)
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(0, color="grey", lw=0.5)
plt.axis("equal")  # Helps visualize the correlation/skew
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.show()

# %% [markdown]
# ## 5. Parameter Conversions (DP <-> CP)
#
# The distribution can be parameterized using Direct Parameters (DP: ξ, Ω, α) or Centered Parameters (CP: μ, Σ, γ₁), where γ₁ is the vector of component-wise skewness.
#
# We can convert between these parameterizations using `dp2cp` and `cp2dp`.

# %% [code]
# Convert original DP (xi, Omega, alpha) to CP
# mu_cp, Sigma_cp, gamma1_cp = multivariate_skew_normal.dp2cp(xi, Omega, alpha)

# print("--- DP to CP Conversion ---")
# print(f"Calculated CP Mean (μ): {mu_cp}")
# print(f"Calculated CP Covariance (Σ):\n{Sigma_cp}")
# print(f"Calculated CP Skewness (γ₁): {gamma1_cp}")

# # Verify consistency with calculated moments
# print(f"\nCP Mean matches .mean(): {np.allclose(mu_cp, mean_vec)}")
# print(f"CP Covariance matches .cov(): {np.allclose(Sigma_cp, cov_mat)}")

# # Convert CP back to DP
# xi_dp_rec, Omega_dp_rec, alpha_dp_rec = multivariate_skew_normal.cp2dp(
#     mu_cp, Sigma_cp, gamma1_cp
# )

# print("\n--- CP to DP Conversion ---")
# print(f"Reconstructed DP Location (ξ): {xi_dp_rec}")
# print(f"Reconstructed DP Scale (Ω):\n{Omega_dp_rec}")
# print(f"Reconstructed DP Shape (α): {alpha_dp_rec}")

# # Check if the reconstructed DP match the original DP
# print(f"\nReconstructed ξ matches original: {np.allclose(xi, xi_dp_rec)}")
# print(f"Reconstructed Ω matches original: {np.allclose(Omega, Omega_dp_rec)}")
# print(f"Reconstructed α matches original: {np.allclose(alpha, alpha_dp_rec)}")

# %% [markdown]
# ## 6. Fitting the Distribution to Data (MLE)
#
# We can estimate the DP parameters (ξ, Ω, α) from a dataset using Maximum Likelihood Estimation (MLE) via the `fit` method.

# %% [code]
# Use the samples generated earlier
print(f"Fitting distribution to {n_samples} samples...")

# Perform MLE fit
try:
    fitted_loc, fitted_scale, fitted_shape = multivariate_skewnormal.fit(samples)

    print("\n--- MLE Fit Results ---")
    print(f"Original Location (ξ): {xi}")
    print(f"Fitted Location (ξ):   {fitted_loc}")
    print(f"\nOriginal Scale (Ω):\n{Omega}")
    print(f"Fitted Scale (Ω):\n{fitted_scale}")
    print(f"\nOriginal Shape (α): {alpha}")
    print(f"Fitted Shape (α):   {fitted_shape}")

    # Check closeness (MLE estimates might not be exact, especially with limited data)
    print(f"\nFitted ξ close to original: {np.allclose(xi, fitted_loc, atol=0.15)}")
    print(f"Fitted Ω close to original: {np.allclose(Omega, fitted_scale, atol=0.2)}")
    print(
        f"Fitted α close to original: {np.allclose(alpha, fitted_shape, atol=0.3)}"
    )  # Shape can be harder to estimate

except Exception as e:
    print(f"MLE fitting failed: {e}")
    print(
        "Fitting can sometimes be challenging, especially if the true shape parameter is large or data is limited."
    )

# %% [markdown]
# ### Fitting with Fixed Parameters
#
# The `fit` method also allows fixing some parameters while estimating others.

# %% [code]
# Example: Fit scale and shape, keeping location fixed at the true value
print("\n--- Fitting with Fixed Location ---")
try:
    fit_loc_f, fit_scale_f, fit_shape_f = multivariate_skew_normal.fit(
        samples, f_loc=xi
    )

    print(f"Fixed Location (ξ):   {fit_loc_f}")
    print(f"Fitted Scale (Ω):\n{fit_scale_f}")
    print(f"Fitted Shape (α):   {fit_shape_f}")
    assert np.allclose(fit_loc_f, xi)  # Check if location remained fixed

except Exception as e:
    print(f"MLE fitting with fixed parameters failed: {e}")

# %% [markdown]
# ## 7. Comparison with Multivariate Normal
#
# When the shape parameter `alpha` is zero, the multivariate skew-normal distribution reduces to the standard multivariate normal distribution.

# %% [code]
# Parameters for a standard normal equivalent
xi_norm = np.array([0.0, 0.0])
Omega_norm = np.array([[1.0, 0.3], [0.3, 1.0]])
alpha_zero = np.array([0.0, 0.0])

# Create MSN with zero shape
msn_zero_shape = multivariate_skew_normal(
    loc=xi_norm, scale=Omega_norm, shape=alpha_zero
)

# Create equivalent MVN
mvn_equiv = multivariate_normal(mean=xi_norm, cov=Omega_norm)

# Compare PDF and moments
x_test = np.array([0.5, 0.5])

pdf_msn_zero = msn_zero_shape.pdf(x_test)
pdf_mvn = mvn_equiv.pdf(x_test)

mean_msn_zero = msn_zero_shape.mean()
mean_mvn = mvn_equiv.mean

cov_msn_zero = msn_zero_shape.cov()
cov_mvn = mvn_equiv.cov

print("--- Comparison with Multivariate Normal (alpha=0) ---")
print(f"Test point: {x_test}")
print(f"MSN PDF (α=0): {pdf_msn_zero:.6f}")
print(f"MVN PDF:       {pdf_mvn:.6f}")
print(f"PDFs match: {np.allclose(pdf_msn_zero, pdf_mvn)}")

print(f"MSN Mean (α=0): {mean_msn_zero}")
print(f"MVN Mean:       {mean_mvn}")
print(f"Means match: {np.allclose(mean_msn_zero, mean_mvn)}")

print(f"MSN Cov (α=0):\n{cov_msn_zero}")
print(f"MVN Cov:\n{cov_mvn}")
print(f"Covariances match: {np.allclose(cov_msn_zero, cov_mvn)}")

# %%
