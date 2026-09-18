"""Thermal comfort evaluation module for ThermoShelter360.

Evaluates indoor thermal comfort based on adaptive thermal comfort guidelines
suitable for high-altitude cold environments (18 C to 24 C standard comfort range).
"""

from typing import Dict, Sequence, Union
import numpy as np
import pandas as pd


COMFORT_MIN_TEMP = 18.0  # C
COMFORT_MAX_TEMP = 24.0  # C


def thermal_comfort_category(temp: float) -> str:
    """Classify a single temperature value into a comfort category.

    Categories:
        - "Cold": Below 18 C
        - "Comfortable": 18 C to 24 C (inclusive)
        - "Hot": Above 24 C

    Args:
        temp: Indoor dry-bulb temperature in degrees Celsius (C).

    Returns:
        String classification: "Cold", "Comfortable", or "Hot".
    """
    if temp < COMFORT_MIN_TEMP:
        return "Cold"
    elif temp <= COMFORT_MAX_TEMP:
        return "Comfortable"
    else:
        return "Hot"


def calculate_comfort_metrics(
    indoor_temp_series: Union[pd.Series, np.ndarray, Sequence[float]],
    step_hours: float = 1.0,
) -> Dict[str, Union[float, int]]:
    """Calculate aggregate thermal comfort metrics over a time series.

    Metrics returned:
        - avg_temp (C): Arithmetic mean indoor temperature.
        - min_temp (C): Minimum recorded indoor temperature.
        - max_temp (C): Maximum recorded indoor temperature.
        - hours_in_comfort_range (hrs): Total hours within [18 C, 24 C].
        - percent_comfortable (%): Percentage of total simulation time in comfort band.

    Args:
        indoor_temp_series: Series or array of indoor temperatures in degrees Celsius.
        step_hours: Duration of each data point in hours (default: 1.0 hr).

    Returns:
        Dictionary containing aggregate comfort statistics.

    Raises:
        ValueError: If input series is empty or step_hours <= 0.
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be strictly positive.")

    temps = np.asarray(indoor_temp_series, dtype=float)
    if temps.size == 0:
        raise ValueError("Cannot calculate comfort metrics on an empty series.")

    avg_temp = float(np.mean(temps))
    min_temp = float(np.min(temps))
    max_temp = float(np.max(temps))

    in_comfort_mask = (temps >= COMFORT_MIN_TEMP) & (temps <= COMFORT_MAX_TEMP)
    comfort_count = int(np.sum(in_comfort_mask))
    total_count = int(temps.size)

    hours_in_comfort_range = float(comfort_count * step_hours)
    percent_comfortable = float((comfort_count / total_count) * 100.0) if total_count > 0 else 0.0

    return {
        "avg_temp": round(avg_temp, 2),
        "min_temp": round(min_temp, 2),
        "max_temp": round(max_temp, 2),
        "hours_in_comfort_range": round(hours_in_comfort_range, 2),
        "percent_comfortable": round(percent_comfortable, 2),
    }
