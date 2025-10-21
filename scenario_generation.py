"""
Scenario generation for PV production, load demand, and grid carbon intensity.

This script constructs synthetic daily scenarios starting from real operational data
using Principal Component Analysis (PCA). It generates statistically consistent 
samples for use in stochastic optimization models.

"""

import pandas as pd
import numpy as np

np.random.seed(0)

# =============================================================================
# USER AND SYSTEM PARAMETERS
# =============================================================================
T = 96  # Number of 15-min intervals in one day
eta_RT = 0.96
user = (5, 13.5, eta_RT, 6.4)  # (Power [kW], Capacity [kWh], efficiency, PV size [kWp])
avg_pv_power = 6.4  # Average photovoltaic system capacity [kWp]
K = 5               # Number of principal components retained
N = 10000           # Number of synthetic scenarios to generate

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def load_data(filepath):
    """
    Load a matrix of daily profiles (rows = days, columns = 96 time steps).
    The file should contain numeric values with each row corresponding to a day.
    """
    # Example expected format: CSV with 96 columns, each representing 15-minute intervals
    return pd.read_csv(filepath, header=None).to_numpy()


def generate_scenarios(data_matrix, cap=None, is_log=False):
    """
    Generate synthetic scenarios using PCA.

    Parameters
    ----------
    data_matrix : ndarray
        Historical daily data matrix (n_days × T).
    cap : float, optional
        Maximum allowable value for the generated data (e.g., PV capacity).
    is_log : bool, optional
        Whether to apply logarithmic transformation before PCA (for αₜ).

    Returns
    -------
    scenarios : ndarray
        Generated synthetic scenarios (N × T).
    """
    # Apply logarithmic transformation if required (e.g., for carbon intensity)
    if is_log:
        data_matrix = np.log(data_matrix)

    # Center the data
    mean_profile = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_profile

    # Perform Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(centered_data, full_matrices=False)
    V = Vt.T

    # Retain first K principal components
    V_k = V[:, :K]

    # Compute eigenvalues (variance of each component)
    eigenvalues = (Sigma ** 2) / (len(data_matrix) - 1)

    # Generate random coefficients using N(0, λ_k)
    coeffs = np.random.normal(loc=0, scale=np.sqrt(eigenvalues[:K]), size=(N, K))

    # Reconstruct synthetic scenarios
    synthetic = coeffs @ V_k.T + mean_profile

    # Apply exponential transformation if log-transformed
    if is_log:
        synthetic = np.exp(synthetic)

    # Apply physical bounds if applicable
    if cap is not None:
        synthetic = np.clip(synthetic, a_min=0, a_max=cap)
    else:
        synthetic = np.clip(synthetic, a_min=0, a_max=None)

    return synthetic


# =============================================================================
# LOAD REAL DATA
# =============================================================================

# Replace these file paths with your actual data sources
# Each file should contain one daily profile per row and 96 columns for 15-min intervals
solar_data_path = "data/solar_profiles.csv"
load_data_path = "data/load_profiles.csv"
alpha_data_path = "data/alpha_profiles.csv"

# Load the input data
S = load_data(solar_data_path)
L = load_data(load_data_path)
A = load_data(alpha_data_path)

# =============================================================================
# SCENARIO GENERATION
# =============================================================================

# --- Photovoltaic generation ---
# Scale by user's PV capacity relative to average system
domestic_solar = S
n_s = 1_600_000  # Estimated number of domestic PV installations
avg_solar = (1 / n_s) * 1000 * domestic_solar  # Normalized production
U_s = (user[3] / avg_pv_power) * avg_solar

solar_scenarios = generate_scenarios(U_s, cap=user[3], is_log=False)

# --- Load demand ---
domestic_loads = L * 0.22  # Representative domestic consumption share
n = 25_690_057  # Estimated number of domestic consumers
avg_load = (1 / n) * 1000 * domestic_loads
delta = 2.65
U_l = delta * (user[3] / avg_pv_power) * avg_load

load_scenarios = generate_scenarios(U_l, cap=None, is_log=False)

# --- Carbon intensity (αₜ) ---
alpha_scenarios = generate_scenarios(A, cap=None, is_log=True)

# =============================================================================
# SAVE GENERATED SCENARIOS
# =============================================================================

np.save("scenarios_solar.npy", solar_scenarios)
np.save("scenarios_load.npy", load_scenarios)
np.save("scenarios_alpha.npy", alpha_scenarios)
