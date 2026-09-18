"""Heat loss calculation module for ThermoShelter360.

Provides analytical thermal loss formulas for building envelopes including
walls, roofs, fenestrations, doors, and air exchange/ventilation in SI units.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def calculate_areas(
    length: float, width: float, height: float
) -> Tuple[float, float, float, float]:
    """Calculate building surface areas and interior volume.

    Args:
        length: Building internal/external length in meters (m).
        width: Building internal/external width in meters (m).
        height: Building internal/external height in meters (m).

    Returns:
        Tuple containing:
            - wall_area (m2): Total gross exterior wall area (2 * (L + W) * H).
            - roof_area (m2): Projected roof area (L * W).
            - floor_area (m2): Ground floor area (L * W).
            - volume (m3): Interior air volume (L * W * H).

    Raises:
        ValueError: If length, width, or height are non-positive.
    """
    if length <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Building dimensions must be strictly positive. Received length={length}, width={width}, height={height}"
        )

    wall_area = 2.0 * (length + width) * height
    roof_area = length * width
    floor_area = length * width
    volume = length * width * height

    return wall_area, roof_area, floor_area, volume


def validate_geometry(
    geometry: Union[Dict[str, Any], float, None] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    wall_thickness: Optional[float] = None,
    window_area: Optional[float] = 0.0,
    door_area: Optional[float] = 0.0,
    length: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Validate shelter geometry. Returns error message or None if valid.

    Args:
        geometry: Dictionary containing geometry parameters or length (m).
        width: Building width (m).
        height: Building height (m).
        wall_thickness: Wall thickness (m).
        window_area: Window area (m2).
        door_area: Door area (m2).
        length: Building length (m) if passed as keyword argument.

    Returns:
        String with bullet-separated error messages, or None if geometry is valid.
    """
    if isinstance(geometry, dict):
        l_val = float(geometry.get("length", 0.0))
        w_val = float(geometry.get("width", 0.0))
        h_val = float(geometry.get("height", 0.0))
        wt_val = float(geometry.get("wall_thickness", 0.0))
        wa_val = float(geometry.get("window_area", 0.0))
        da_val = float(geometry.get("door_area", 0.0))
    else:
        l_val = float(length if length is not None else (geometry if geometry is not None else 0.0))
        w_val = float(width if width is not None else 0.0)
        h_val = float(height if height is not None else 0.0)
        wt_val = float(wall_thickness if wall_thickness is not None else 0.0)
        wa_val = float(window_area if window_area is not None else 0.0)
        da_val = float(door_area if door_area is not None else 0.0)

    errors = []
    if l_val <= 0:
        errors.append("Length must be > 0")
    if w_val <= 0:
        errors.append("Width must be > 0")
    if h_val <= 0:
        errors.append("Height must be > 0")
    if wt_val <= 0:
        errors.append("Wall thickness must be > 0")
    if wa_val < 0:
        errors.append("Window area cannot be negative")
    if da_val < 0:
        errors.append("Door area cannot be negative")

    # Openings cannot exceed 90% of gross wall area
    gross_wall = 2.0 * (l_val + w_val) * h_val
    if gross_wall > 0 and (wa_val + da_val) > gross_wall * 0.9:
        errors.append(f"Openings ({wa_val + da_val:.1f} m²) exceed 90% of gross wall area ({gross_wall:.1f} m²)")
    elif gross_wall <= 0 and (wa_val + da_val) > 0:
        errors.append("Openings exceed gross wall area")

    return " • ".join(errors) if errors else None


def validate_material(material: Union[Dict[str, Any], Any]) -> Optional[str]:
    """Validate material properties. Returns error message or None if valid."""
    if not isinstance(material, dict):
        return "Material parameters must be a dictionary"

    k = float(material.get("thermal_conductivity", 0.0))
    rho = float(material.get("density", 0.0))
    cp = float(material.get("specific_heat", 0.0))
    alpha = float(material.get("absorptivity", 0.65))

    errors = []
    if k <= 0:
        errors.append("Thermal conductivity must be > 0")
    if rho <= 0:
        errors.append("Density must be > 0")
    if cp <= 0:
        errors.append("Specific heat must be > 0")
    if not (0.0 <= alpha <= 1.0):
        errors.append("Absorptivity must be between 0.0 and 1.0")

    return " • ".join(errors) if errors else None


def validate_thermal_params(params: Union[Dict[str, Any], Any]) -> Optional[str]:
    """Validate thermal solver parameters. Returns error message or None if valid."""
    if not isinstance(params, dict):
        return "Thermal parameters must be a dictionary"

    c_mass = float(params.get("thermal_capacity", 0.0))
    vent_rate = float(params.get("ventilation_rate", 0.0))
    internal_gain = float(params.get("internal_heat_gain", 0.0))

    errors = []
    if c_mass <= 0:
        errors.append("Thermal capacity must be > 0")
    if vent_rate < 0:
        errors.append("Ventilation rate cannot be negative")
    if internal_gain < 0:
        errors.append("Internal heat gain cannot be negative")

    return " • ".join(errors) if errors else None


def wall_heat_loss(
    k: float, wall_area: float, Tin: float, Tout: float, thickness: float
) -> float:
    """Calculate steady-state conduction heat loss through a single-layer wall.

    Formula: Q = k * A * (Tin - Tout) / thickness

    Args:
        k: Thermal conductivity of the material in W/(m*K).
        wall_area: Effective net surface area of the wall in m2.
        Tin: Indoor air/surface temperature in degrees Celsius (C).
        Tout: Outdoor ambient temperature in degrees Celsius (C).
        thickness: Total wall thickness in meters (m).

    Returns:
        Heat loss rate in Watts (W). Positive indicates heat flowing from inside to outside.

    Raises:
        ValueError: If thickness <= 0, k < 0, or wall_area < 0.
    """
    if thickness <= 0:
        raise ValueError("Wall thickness must be strictly positive to avoid division by zero.")
    if k < 0:
        raise ValueError("Thermal conductivity cannot be negative.")
    if wall_area < 0:
        raise ValueError("Wall area cannot be negative.")

    return (k * wall_area * (Tin - Tout)) / thickness


def composite_wall_heat_loss(
    layers: List[Dict[str, float]],
    wall_area: float,
    Tin: float,
    Tout: float,
) -> float:
    """Calculate heat loss through a multi-layered composite wall.

    Formula:
        R_total = sum(d_i / k_i)
        U = 1 / R_total
        Q = U * A * (Tin - Tout)

    Args:
        layers: List of layer dictionaries with keys 'thickness' (m) and 'k' (W/m*K).
        wall_area: Effective wall area in m2.
        Tin: Indoor temperature in C.
        Tout: Outdoor temperature in C.

    Returns:
        Heat loss rate in Watts (W).

    Raises:
        ValueError: If layers is empty, wall_area < 0, or any layer has invalid thickness/k.
    """
    if not layers:
        raise ValueError("Layers list cannot be empty.")
    if wall_area < 0:
        raise ValueError("Wall area cannot be negative.")

    r_total = 0.0
    for idx, layer in enumerate(layers):
        thickness = layer.get("thickness", 0.0)
        k = layer.get("k", layer.get("thermal_conductivity", 0.0))

        if thickness <= 0:
            raise ValueError(f"Layer {idx} thickness must be strictly positive (got {thickness}).")
        if k <= 0:
            raise ValueError(f"Layer {idx} thermal conductivity must be strictly positive (got {k}).")

        r_total += thickness / k

    if r_total <= 0:
        raise ValueError("Total thermal resistance R_total must be positive.")

    u_value = 1.0 / r_total
    return u_value * wall_area * (Tin - Tout)


def roof_heat_loss(U_roof: float, roof_area: float, Tin: float, Tout: float) -> float:
    """Calculate heat loss through the roof assembly.

    Formula: Q = U * A * (Tin - Tout)

    Args:
        U_roof: Overall heat transfer coefficient (U-value) in W/(m2*K).
        roof_area: Roof surface area in m2.
        Tin: Indoor temperature in C.
        Tout: Outdoor temperature in C.

    Returns:
        Roof heat loss rate in Watts (W).

    Raises:
        ValueError: If U_roof < 0 or roof_area < 0.
    """
    if U_roof < 0 or roof_area < 0:
        raise ValueError("Roof U-value and roof area must be non-negative.")

    return U_roof * roof_area * (Tin - Tout)


def window_heat_loss(
    U_window: float, window_area: float, Tin: float, Tout: float
) -> float:
    """Calculate heat loss through fenestrations (windows/glazing).

    Formula: Q = U * A * (Tin - Tout)

    Args:
        U_window: Window overall heat transfer coefficient in W/(m2*K).
        window_area: Total window glazed area in m2.
        Tin: Indoor temperature in C.
        Tout: Outdoor temperature in C.

    Returns:
        Window heat loss rate in Watts (W).

    Raises:
        ValueError: If U_window < 0 or window_area < 0.
    """
    if U_window < 0 or window_area < 0:
        raise ValueError("Window U-value and window area must be non-negative.")

    return U_window * window_area * (Tin - Tout)


def door_heat_loss(U_door: float, door_area: float, Tin: float, Tout: float) -> float:
    """Calculate heat loss through external doors.

    Formula: Q = U * A * (Tin - Tout)

    Args:
        U_door: Door overall heat transfer coefficient in W/(m2*K).
        door_area: Total door surface area in m2.
        Tin: Indoor temperature in C.
        Tout: Outdoor temperature in C.

    Returns:
        Door heat loss rate in Watts (W).

    Raises:
        ValueError: If U_door < 0 or door_area < 0.
    """
    if U_door < 0 or door_area < 0:
        raise ValueError("Door U-value and door area must be non-negative.")

    return U_door * door_area * (Tin - Tout)


def ventilation_heat_loss(
    air_density: float,
    air_specific_heat: float,
    ventilation_rate: float,
    Tin: float,
    Tout: float,
) -> float:
    """Calculate sensible heat loss due to ventilation and natural air infiltration.

    Formula: Q = rho * cp * V_rate * (Tin - Tout)

    Args:
        air_density: Air density (rho) in kg/m3 (typically ~1.0 - 1.2 kg/m3 depending on altitude).
        air_specific_heat: Specific heat capacity of air (cp) in J/(kg*K) (approx 1005 J/kg*K).
        ventilation_rate: Volumetric air flow rate (V_dot) in m3/s.
        Tin: Indoor temperature in C.
        Tout: Outdoor temperature in C.

    Returns:
        Ventilation heat loss rate in Watts (W).

    Raises:
        ValueError: If air_density < 0, air_specific_heat < 0, or ventilation_rate < 0.
    """
    if air_density < 0 or air_specific_heat < 0 or ventilation_rate < 0:
        raise ValueError("Air density, specific heat, and ventilation rate must be non-negative.")

    return air_density * air_specific_heat * ventilation_rate * (Tin - Tout)
