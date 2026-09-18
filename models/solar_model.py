"""Solar radiation and solar heat gain model for ThermoShelter360.

Calculates direct and global solar gains incident on building surfaces
and transmitted through passive solar apertures.
"""

from typing import List, Sequence, Union
import numpy as np
import pandas as pd


def solar_gain(
    absorptivity: float, solar_irradiance: float, exposed_area: float
) -> float:
    """Calculate instantaneous solar thermal gain.

    Formula: Q = alpha * G * A

    Args:
        absorptivity: Surface solar absorptance coefficient (alpha) between 0.0 and 1.0.
        solar_irradiance: Global horizontal or surface solar irradiance (G) in W/m2.
        exposed_area: Effective solar aperture / surface area (A) in m2.

    Returns:
        Solar heat gain in Watts (W).

    Raises:
        ValueError: If absorptivity is outside [0.0, 1.0], or irradiance/area is negative.
    """
    if absorptivity < 0.0 or absorptivity > 1.0:
        raise ValueError(
            f"Solar absorptivity must be between 0.0 and 1.0. Received {absorptivity}."
        )
    if solar_irradiance < 0.0:
        raise ValueError(
            f"Solar irradiance cannot be negative. Received {solar_irradiance}."
        )
    if exposed_area < 0.0:
        raise ValueError(
            f"Exposed area cannot be negative. Received {exposed_area}."
        )

    return absorptivity * solar_irradiance * exposed_area


def solar_gain_time_series(
    absorptivity: float,
    exposed_area: float,
    irradiance_array: Union[Sequence[float], np.ndarray, pd.Series],
) -> np.ndarray:
    """Calculate solar heat gain time series across an array or series of solar irradiance values.

    Args:
        absorptivity: Surface solar absorptivity (0.0 to 1.0).
        exposed_area: Solar exposed surface area in m2.
        irradiance_array: Sequence, NumPy array, or Pandas Series of irradiance values in W/m2.

    Returns:
        NumPy array of solar heat gains in Watts (W).

    Raises:
        ValueError: If absorptivity, exposed_area, or any irradiance value is invalid.
    """
    if absorptivity < 0.0 or absorptivity > 1.0:
        raise ValueError(f"Solar absorptivity must be between 0.0 and 1.0. Got {absorptivity}.")
    if exposed_area < 0.0:
        raise ValueError(f"Exposed area cannot be negative. Got {exposed_area}.")

    arr = np.asarray(irradiance_array, dtype=float)
    if (arr < 0.0).any():
        raise ValueError("Solar irradiance values cannot be negative.")

    return absorptivity * arr * exposed_area
