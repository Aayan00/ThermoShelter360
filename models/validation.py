"""Validation module for comparing ThermoShelter360 simulation with ANSYS CFD/thermal results.

Provides statistical error metrics (MAE, RMSE, Max Error) and side-by-side
comparison dataframes for model verification.
"""

import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compare_with_ansys(
    python_df: pd.DataFrame, ansys_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compare Python simulation indoor temperature results with imported ANSYS data.

    Calculates:
        - absolute_error: Point-by-point temperature difference |T_python - T_ansys| (C)
        - mean_absolute_error (MAE): Mean Absolute Error (C)
        - root_mean_square_error (RMSE): Root Mean Square Error (C)
        - max_absolute_error: Peak absolute error (C)
        - comparison_df: Aligned DataFrame with columns:
            [time, python_temp, ansys_temp, ansys_min_temp, ansys_max_temp, absolute_error]

    Args:
        python_df: Results DataFrame from simulate_indoor_temperature containing 'indoor_temperature'.
        ansys_df: DataFrame loaded from ANSYS CSV export. Expected columns:
            'time' (or 'timestamp'), 'indoor_temperature', optionally 'maximum_temperature',
            'minimum_temperature', 'total_heat_flow'.

    Returns:
        Dictionary containing statistical metrics and the comparison DataFrame.

    Raises:
        ValueError: If either DataFrame is empty or missing required temperature columns.
    """
    if python_df.empty or ansys_df.empty:
        raise ValueError("Both Python simulation results and ANSYS dataframes must be non-empty.")

    # Find Python temperature column
    py_temp_col = None
    for col in ["indoor_temperature", "Tin", "indoor_temp", "temp"]:
        if col in python_df.columns:
            py_temp_col = col
            break
    if py_temp_col is None:
        raise ValueError(
            f"Python simulation DataFrame must have an 'indoor_temperature' column. Found: {list(python_df.columns)}"
        )

    # Standardize ANSYS column names (lowercase, stripped)
    ansys_clean = ansys_df.copy()
    ansys_clean.columns = [c.strip().lower().replace(" ", "_") for c in ansys_clean.columns]

    ansys_temp_col = None
    for col in ["indoor_temperature", "indoor_temp", "temp", "temperature"]:
        if col in ansys_clean.columns:
            ansys_temp_col = col
            break
    if ansys_temp_col is None:
        raise ValueError(
            f"ANSYS DataFrame must contain an 'indoor_temperature' column. Found: {list(ansys_clean.columns)}"
        )

    # Determine time alignment
    has_mismatch = False
    mismatch_msg = ""

    len_py = len(python_df)
    len_ansys = len(ansys_clean)

    if len_py != len_ansys:
        has_mismatch = True
        mismatch_msg = (
            f"Length mismatch: Python simulation has {len_py} steps, while ANSYS dataset has {len_ansys} steps. "
            f"Aligning to shortest common duration ({min(len_py, len_ansys)} steps)."
        )
        warnings.warn(mismatch_msg, UserWarning)

    n_common = min(len_py, len_ansys)
    py_subset = python_df.iloc[:n_common].reset_index(drop=True)
    ansys_subset = ansys_clean.iloc[:n_common].reset_index(drop=True)

    time_vals = (
        ansys_subset["time"].tolist()
        if "time" in ansys_subset.columns
        else (
            py_subset["timestamp"].tolist()
            if "timestamp" in py_subset.columns
            else list(range(n_common))
        )
    )

    t_py = py_subset[py_temp_col].astype(float).to_numpy()
    t_ansys = ansys_subset[ansys_temp_col].astype(float).to_numpy()

    abs_error = np.abs(t_py - t_ansys)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(abs_error**2)))
    max_err = float(np.max(abs_error))

    comp_dict = {
        "time": time_vals,
        "python_temp": np.round(t_py, 2),
        "ansys_temp": np.round(t_ansys, 2),
        "absolute_error": np.round(abs_error, 2),
    }

    if "minimum_temperature" in ansys_subset.columns:
        comp_dict["ansys_min_temp"] = np.round(ansys_subset["minimum_temperature"].astype(float).to_numpy(), 2)
    if "maximum_temperature" in ansys_subset.columns:
        comp_dict["ansys_max_temp"] = np.round(ansys_subset["maximum_temperature"].astype(float).to_numpy(), 2)
    if "total_heat_flow" in ansys_subset.columns:
        comp_dict["ansys_heat_flow"] = np.round(ansys_subset["total_heat_flow"].astype(float).to_numpy(), 2)

    comp_df = pd.DataFrame(comp_dict)

    return {
        "absolute_error": comp_df["absolute_error"],
        "mean_absolute_error": round(mae, 3),
        "root_mean_square_error": round(rmse, 3),
        "max_absolute_error": round(max_err, 3),
        "comparison_df": comp_df,
        "has_mismatch": has_mismatch,
        "mismatch_warning": mismatch_msg,
    }
