"""Thermal simulation engine for ThermoShelter360.

Simulates dynamic transient indoor temperatures, component heat losses,
solar gains, and thermal mass storage for high-altitude passive shelters.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from models.heat_loss import (
    calculate_areas,
    composite_wall_heat_loss,
    door_heat_loss,
    roof_heat_loss,
    validate_geometry,
    ventilation_heat_loss,
    wall_heat_loss,
    window_heat_loss,
)
from models.solar_model import solar_gain


def simulate_indoor_temperature(
    climate_df: pd.DataFrame,
    geometry: Dict[str, float],
    material: Dict[str, Any],
    thermal_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Simulate transient indoor temperature and heat balance over time.

    Energy balance differential equation:
        C_th * (dT_in / dt) = Q_solar + Q_internal - Q_loss
        T_in(t + dt) = T_in(t) + (dt / C_th) * (Q_solar + Q_internal - Q_loss)

    Args:
        climate_df: DataFrame with climate time series.
            Must contain columns for outdoor temperature (e.g. 'outdoor_temperature' or 'outdoor_temperature_c')
            and solar irradiance (e.g. 'solar_irradiance' or 'solar_irradiance_w_m2').
            Optionally includes 'timestamp'.
        geometry: Dictionary containing:
            - length (m)
            - width (m)
            - height (m)
            - wall_thickness (m)
            - window_area (m2)
            - door_area (m2)
            - roof_u_value (W/m2K)
            - window_u_value (W/m2K)
            - door_u_value (W/m2K)
            - Optional solar_exposed_area (m2)
        material: Dictionary describing the wall material:
            - thermal_conductivity (W/m*K)
            - absorptivity (0.0 to 1.0)
            - Optional layers (list of dicts with 'thickness' and 'k')
            - Optional density (kg/m3) and specific_heat (J/kg*K)
        thermal_params: Dictionary with thermal parameters:
            - thermal_capacity (J/K, default calculated or 2.0e6 J/K)
            - internal_heat_gain (W, default 150.0 W)
            - ventilation_rate (m3/s, default 0.005 m3/s)
            - air_density (kg/m3, default 1.05 kg/m3 for ~3500m Ladakh altitude)
            - air_specific_heat (J/kg*K, default 1005.0 J/kg*K)
            - initial_indoor_temp (C, default 15.0 C)
            - time_step_seconds (s, default inferred or 3600.0 s)

    Returns:
        DataFrame with columns:
            - timestamp
            - outdoor_temperature
            - solar_irradiance
            - indoor_temperature
            - wall_heat_loss
            - roof_heat_loss
            - window_heat_loss
            - door_heat_loss
            - ventilation_heat_loss
            - total_heat_loss
            - solar_gain
            - net_heat_gain

    Raises:
        ValueError: If inputs are unphysical or required data columns are missing.
    """
    if climate_df.empty:
        raise ValueError("Climate DataFrame cannot be empty.")

    # Validate geometry
    geom_err = validate_geometry(geometry)
    if geom_err:
        raise ValueError(f"Invalid geometry: {geom_err}")

    length = float(geometry.get("length", 0.0))
    width = float(geometry.get("width", 0.0))
    height = float(geometry.get("height", 0.0))
    wall_thickness = float(geometry.get("wall_thickness", 0.0))
    window_area = float(geometry.get("window_area", 0.0))
    door_area = float(geometry.get("door_area", 0.0))

    gross_wall_area, roof_area, floor_area, volume = calculate_areas(length, width, height)
    net_wall_area = max(0.0, gross_wall_area - (window_area + door_area))

    roof_u_value = float(geometry.get("roof_u_value", 0.35))
    window_u_value = float(geometry.get("window_u_value", 1.8))
    door_u_value = float(geometry.get("door_u_value", 2.0))

    # Identify climate columns
    temp_col = None
    for col in ["outdoor_temperature_c", "outdoor_temperature", "temp", "tout"]:
        if col in climate_df.columns:
            temp_col = col
            break
    if temp_col is None:
        raise ValueError(
            f"Climate data must contain an outdoor temperature column. Available: {list(climate_df.columns)}"
        )

    solar_col = None
    for col in ["solar_irradiance_w_m2", "solar_irradiance", "irradiance", "ghi", "solar"]:
        if col in climate_df.columns:
            solar_col = col
            break
    if solar_col is None:
        raise ValueError(
            f"Climate data must contain a solar irradiance column. Available: {list(climate_df.columns)}"
        )

    # Solar exposed aperture / wall area
    solar_exposed_area = float(
        geometry.get("solar_exposed_area", window_area + 0.3 * gross_wall_area)
    )
    absorptivity = float(material.get("absorptivity", 0.65))
    if absorptivity < 0.0 or absorptivity > 1.0:
        raise ValueError(f"Material absorptivity must be between 0.0 and 1.0. Got {absorptivity}.")

    # Material thermal properties
    layers = material.get("layers", None)
    k_wall = float(material.get("thermal_conductivity", material.get("k", 0.72)))

    # Thermal parameters
    params = thermal_params or {}
    thermal_capacity = float(params.get("thermal_capacity", 2.5e6))
    if thermal_capacity <= 0:
        raise ValueError("Thermal capacity must be strictly positive.")

    internal_heat_gain = float(params.get("internal_heat_gain", 150.0))
    ventilation_rate = float(params.get("ventilation_rate", 0.005))
    air_density = float(params.get("air_density", 1.05))
    air_specific_heat = float(params.get("air_specific_heat", 1005.0))
    initial_indoor_temp = float(params.get("initial_indoor_temp", 15.0))

    dt = float(params.get("time_step_seconds", 3600.0))
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    timestamps = (
        climate_df["timestamp"].tolist()
        if "timestamp" in climate_df.columns
        else list(range(len(climate_df)))
    )
    out_temps = climate_df[temp_col].astype(float).to_numpy()
    solars = climate_df[solar_col].astype(float).to_numpy()

    n_steps = len(climate_df)

    indoor_temps = np.zeros(n_steps, dtype=float)
    q_wall = np.zeros(n_steps, dtype=float)
    q_roof = np.zeros(n_steps, dtype=float)
    q_window = np.zeros(n_steps, dtype=float)
    q_door = np.zeros(n_steps, dtype=float)
    q_vent = np.zeros(n_steps, dtype=float)
    q_total_loss = np.zeros(n_steps, dtype=float)
    q_solar = np.zeros(n_steps, dtype=float)
    q_net = np.zeros(n_steps, dtype=float)

    current_tin = initial_indoor_temp

    for i in range(n_steps):
        tout = out_temps[i]
        g_solar = max(0.0, solars[i])

        # Wall heat loss (composite or single layer)
        if layers and len(layers) > 0:
            qw = composite_wall_heat_loss(layers, net_wall_area, current_tin, tout)
        else:
            qw = wall_heat_loss(k_wall, net_wall_area, current_tin, tout, wall_thickness)

        qr = roof_heat_loss(roof_u_value, roof_area, current_tin, tout)
        qwin = window_heat_loss(window_u_value, window_area, current_tin, tout)
        qd = door_heat_loss(door_u_value, door_area, current_tin, tout)
        qv = ventilation_heat_loss(air_density, air_specific_heat, ventilation_rate, current_tin, tout)

        tot_loss = qw + qr + qwin + qd + qv
        sol_gain = solar_gain(absorptivity, g_solar, solar_exposed_area)
        net_gain = sol_gain + internal_heat_gain - tot_loss

        indoor_temps[i] = current_tin
        q_wall[i] = qw
        q_roof[i] = qr
        q_window[i] = qwin
        q_door[i] = qd
        q_vent[i] = qv
        q_total_loss[i] = tot_loss
        q_solar[i] = sol_gain
        q_net[i] = net_gain

        # Euler forward integration for next step
        tin_next = current_tin + (dt / thermal_capacity) * net_gain
        current_tin = tin_next

    results_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "outdoor_temperature": np.round(out_temps, 2),
            "solar_irradiance": np.round(solars, 2),
            "indoor_temperature": np.round(indoor_temps, 2),
            "wall_heat_loss": np.round(q_wall, 2),
            "roof_heat_loss": np.round(q_roof, 2),
            "window_heat_loss": np.round(q_window, 2),
            "door_heat_loss": np.round(q_door, 2),
            "ventilation_heat_loss": np.round(q_vent, 2),
            "total_heat_loss": np.round(q_total_loss, 2),
            "solar_gain": np.round(q_solar, 2),
            "internal_heat_gain": np.round(np.full(n_steps, internal_heat_gain), 2),
            "net_heat_gain": np.round(q_net, 2),
        }
    )

    return results_df


def compare_materials(
    climate_df: pd.DataFrame,
    geometry: Dict[str, float],
    materials_list: List[Dict[str, Any]],
    thermal_params: Optional[Dict[str, Any]] = None,
    target_comfort_temp: float = 18.0,
) -> pd.DataFrame:
    """Run simulations across a list of materials/wall-assemblies and compare metrics.

    Args:
        climate_df: Climate input DataFrame.
        geometry: Shelter geometry dictionary.
        materials_list: List of material dictionaries, each containing:
            - 'material' or 'name': str
            - 'thermal_conductivity': float
            - 'absorptivity': float
            - Optional 'layers': list of composite layers
        thermal_params: Optional thermal parameters dictionary.
        target_comfort_temp: Benchmark comfort temperature for heating demand (C, default: 18.0 C).

    Returns:
        Comparison DataFrame with columns:
            - material: Material / assembly name
            - avg_indoor_temp: Mean indoor temperature (C)
            - min_indoor_temp: Minimum indoor temperature (C)
            - max_indoor_temp: Maximum indoor temperature (C)
            - total_heat_loss: Total cumulative heat loss (kWh)
            - solar_gain: Total cumulative solar gain (kWh)
            - heating_requirement: Estimated supplemental heating demand to maintain target comfort temp (kWh)
    """
    if not materials_list:
        raise ValueError("materials_list cannot be empty.")

    dt = float((thermal_params or {}).get("time_step_seconds", 3600.0))
    dt_hours = dt / 3600.0

    comparison_records = []

    for mat in materials_list:
        mat_name = mat.get("name", mat.get("material", "Unknown Material"))
        sim_df = simulate_indoor_temperature(
            climate_df=climate_df,
            geometry=geometry,
            material=mat,
            thermal_params=thermal_params,
        )

        avg_tin = float(sim_df["indoor_temperature"].mean())
        min_tin = float(sim_df["indoor_temperature"].min())
        max_tin = float(sim_df["indoor_temperature"].max())

        # Cumulative heat loss and solar gain in kWh (Watts * hours / 1000)
        tot_loss_kwh = float((sim_df["total_heat_loss"] * dt_hours / 1000.0).sum())
        tot_solar_kwh = float((sim_df["solar_gain"] * dt_hours / 1000.0).sum())

        # Heating requirement: energy needed to lift any hour where Tin < target_comfort_temp
        # Supplemental power (W) needed: max(0, Q_loss_at_target - Q_solar - Q_internal)
        # Or degree-hours thermal demand:
        deficits = np.maximum(0.0, target_comfort_temp - sim_df["indoor_temperature"].to_numpy())
        c_th = float((thermal_params or {}).get("thermal_capacity", 2.5e6))
        # Supplemental energy in kWh = sum(deficits * C_th / (3.6e6))
        # Alternatively based on heat loss difference:
        heating_kwh = float(np.sum(deficits * c_th / 3.6e6))

        comparison_records.append(
            {
                "material": mat_name,
                "avg_indoor_temp": round(avg_tin, 2),
                "min_indoor_temp": round(min_tin, 2),
                "max_indoor_temp": round(max_tin, 2),
                "total_heat_loss": round(tot_loss_kwh, 2),
                "solar_gain": round(tot_solar_kwh, 2),
                "heating_requirement": round(heating_kwh, 2),
            }
        )

    return pd.DataFrame(comparison_records)
