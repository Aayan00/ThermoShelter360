"""ThermoShelter360 Models Package.

Exports core engineering calculations for heat loss, solar gains,
dynamic thermal simulation, thermal comfort classification, and numerical validation.
"""

from models.comfort import (
    calculate_comfort_metrics,
    thermal_comfort_category,
)
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
from models.shelter_3d import (
    create_3d_shelter_figure,
)
from models.solar_model import (
    solar_gain,
    solar_gain_time_series,
)
from models.thermal_model import (
    compare_materials,
    simulate_indoor_temperature,
)
from models.validation import (
    compare_with_ansys,
)

__all__ = [
    "calculate_areas",
    "validate_geometry",
    "wall_heat_loss",
    "composite_wall_heat_loss",
    "roof_heat_loss",
    "window_heat_loss",
    "door_heat_loss",
    "ventilation_heat_loss",
    "create_3d_shelter_figure",
    "solar_gain",
    "solar_gain_time_series",
    "simulate_indoor_temperature",
    "compare_materials",
    "calculate_comfort_metrics",
    "thermal_comfort_category",
    "compare_with_ansys",
]
