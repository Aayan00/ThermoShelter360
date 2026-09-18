"""Comprehensive automated test suite for ThermoShelter360 physics and models."""

import os
import numpy as np
import pandas as pd
import pytest

from database.db import (
    delete_simulation,
    get_saved_simulations,
    get_simulation_by_id,
    initialize_database,
    save_simulation,
)
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


# ==============================================================================
# 1. Equal temperatures (Tin = Tout = 10 C) -> Conduction heat loss is zero
# ==============================================================================
def test_equal_temperatures_zero_conduction():
    """Test that heat loss across all mechanisms is exactly zero when Tin == Tout."""
    tin = 10.0
    tout = 10.0
    wall_area = 45.0
    k = 0.72
    thickness = 0.25

    # Wall conduction
    q_wall = wall_heat_loss(k=k, wall_area=wall_area, Tin=tin, Tout=tout, thickness=thickness)
    assert q_wall == pytest.approx(0.0)

    # Composite wall conduction
    layers = [{"thickness": 0.25, "k": 0.72}, {"thickness": 0.05, "k": 0.035}]
    q_comp = composite_wall_heat_loss(layers=layers, wall_area=wall_area, Tin=tin, Tout=tout)
    assert q_comp == pytest.approx(0.0)

    # Roof, window, door, ventilation
    assert roof_heat_loss(U_roof=0.35, roof_area=25.0, Tin=tin, Tout=tout) == pytest.approx(0.0)
    assert window_heat_loss(U_window=1.8, window_area=4.0, Tin=tin, Tout=tout) == pytest.approx(0.0)
    assert door_heat_loss(U_door=2.0, door_area=2.0, Tin=tin, Tout=tout) == pytest.approx(0.0)
    assert ventilation_heat_loss(air_density=1.05, air_specific_heat=1005.0, ventilation_rate=0.01, Tin=tin, Tout=tout) == pytest.approx(0.0)


# ==============================================================================
# 2. Cold outside (Tin = 20, Tout = -10) -> Heat loss should be positive
# ==============================================================================
def test_cold_outside_positive_heat_loss():
    """Test that positive temperature differential produces positive heat loss."""
    tin = 20.0
    tout = -10.0
    delta_t = tin - tout  # 30.0 C
    wall_area = 50.0
    k = 0.72
    thickness = 0.25

    expected_q_wall = (k * wall_area * delta_t) / thickness
    q_wall = wall_heat_loss(k=k, wall_area=wall_area, Tin=tin, Tout=tout, thickness=thickness)

    assert q_wall > 0.0
    assert q_wall == pytest.approx(expected_q_wall)

    # Roof & window
    q_roof = roof_heat_loss(U_roof=0.4, roof_area=30.0, Tin=tin, Tout=tout)
    assert q_roof > 0.0
    assert q_roof == pytest.approx(0.4 * 30.0 * 30.0)


# ==============================================================================
# 3. No sunlight (irradiance = 0) -> Solar gain should be zero
# ==============================================================================
def test_no_sunlight_zero_solar_gain():
    """Test that solar gain is zero when solar irradiance is 0 W/m2."""
    absorptivity = 0.65
    area = 15.0
    q_solar = solar_gain(absorptivity=absorptivity, solar_irradiance=0.0, exposed_area=area)
    assert q_solar == pytest.approx(0.0)

    # Time series test
    irradiance_series = [0.0, 0.0, 0.0]
    gains = solar_gain_time_series(absorptivity, area, irradiance_series)
    assert np.all(gains == 0.0)


# ==============================================================================
# 4. Insulation comparison (Brick vs Brick + EPS) -> Insulated has lower heat loss
# ==============================================================================
def test_insulation_comparison():
    """Test that adding insulation significantly reduces envelope heat loss."""
    tin = 20.0
    tout = -15.0
    wall_area = 60.0
    brick_thick = 0.25
    brick_k = 0.72
    eps_thick = 0.05
    eps_k = 0.035

    # Uninsulated brick wall
    q_uninsulated = wall_heat_loss(
        k=brick_k,
        wall_area=wall_area,
        Tin=tin,
        Tout=tout,
        thickness=brick_thick,
    )

    # Brick + EPS composite
    layers = [
        {"thickness": brick_thick, "k": brick_k},
        {"thickness": eps_thick, "k": eps_k},
    ]
    q_insulated = composite_wall_heat_loss(
        layers=layers,
        wall_area=wall_area,
        Tin=tin,
        Tout=tout,
    )

    assert q_insulated > 0.0
    assert q_insulated < q_uninsulated
    # EPS layer adds significant resistance, reducing heat loss by > 70%
    reduction_pct = (q_uninsulated - q_insulated) / q_uninsulated * 100.0
    assert reduction_pct > 70.0


# ==============================================================================
# 5. Invalid inputs -> Should raise ValueError
# ==============================================================================
def test_invalid_inputs():
    """Test that invalid geometric and physical inputs raise descriptive ValueErrors."""
    # Zero wall thickness
    with pytest.raises(ValueError, match="thickness"):
        wall_heat_loss(k=0.72, wall_area=50.0, Tin=20.0, Tout=-10.0, thickness=0.0)

    # Negative wall thickness
    with pytest.raises(ValueError, match="thickness"):
        wall_heat_loss(k=0.72, wall_area=50.0, Tin=20.0, Tout=-10.0, thickness=-0.1)

    # Negative length in geometry
    with pytest.raises(ValueError, match="dimensions"):
        calculate_areas(length=-2.0, width=4.0, height=2.8)

    # Window area + door area exceeding total gross wall area
    # 5m x 4m x 3m -> Gross wall area = 2 * (5+4) * 3 = 54 m2
    invalid_geom = {
        "length": 5.0,
        "width": 4.0,
        "height": 3.0,
        "wall_thickness": 0.25,
        "window_area": 40.0,
        "door_area": 20.0,  # 40 + 20 = 60 > 54
    }
    err = validate_geometry(invalid_geom)
    assert err is not None
    assert "exceed" in err.lower()

    # simulate_indoor_temperature should raise ValueError on invalid geometry
    sample_climate = pd.DataFrame({
        "timestamp": ["00:00", "01:00"],
        "outdoor_temperature_c": [-10.0, -12.0],
        "solar_irradiance_w_m2": [0.0, 0.0],
    })
    with pytest.raises(ValueError, match="Invalid geometry"):
        simulate_indoor_temperature(
            climate_df=sample_climate,
            geometry=invalid_geom,
            material={"material": "Adobe", "thermal_conductivity": 0.72, "density": 1800, "specific_heat": 920, "absorptivity": 0.65},
            thermal_params={"thermal_capacity": 5000000.0, "internal_heat_gain": 100.0, "ventilation_rate": 0.01},
        )

    # Invalid absorptivity (> 1.0 or < 0.0)
    with pytest.raises(ValueError, match="absorptivity"):
        solar_gain(absorptivity=1.2, solar_irradiance=500.0, exposed_area=10.0)


# ==============================================================================
# 6. Indoor temperature update -> Tin_next changes in correct direction
# ==============================================================================
def test_indoor_temp_update_direction():
    """Test that indoor temperature evolves physically under net loss and net gain."""
    geometry = {
        "length": 6.0,
        "width": 4.0,
        "height": 2.8,
        "wall_thickness": 0.25,
        "window_area": 3.0,
        "door_area": 1.8,
        "roof_u_value": 0.35,
        "window_u_value": 1.8,
        "door_u_value": 2.0,
    }
    material = {
        "material": "Brick",
        "thermal_conductivity": 0.72,
        "absorptivity": 0.65,
    }

    # Case A: Freezing night, zero solar, low internal gain -> Indoor temp must decrease
    cold_climate = pd.DataFrame(
        {
            "timestamp": ["2026-01-15 00:00", "2026-01-15 01:00", "2026-01-15 02:00"],
            "outdoor_temperature_c": [-20.0, -20.0, -20.0],
            "solar_irradiance_w_m2": [0.0, 0.0, 0.0],
        }
    )
    cooling_sim = simulate_indoor_temperature(
        climate_df=cold_climate,
        geometry=geometry,
        material=material,
        thermal_params={"initial_indoor_temp": 15.0, "internal_heat_gain": 50.0},
    )
    temps = cooling_sim["indoor_temperature"].tolist()
    assert temps[0] > temps[1] >= temps[2]

    # Case B: Strong solar irradiance (900 W/m2) and moderate outdoor temp -> Indoor temp must rise
    sunny_climate = pd.DataFrame(
        {
            "timestamp": ["2026-01-15 11:00", "2026-01-15 12:00", "2026-01-15 13:00"],
            "outdoor_temperature_c": [5.0, 6.0, 6.0],
            "solar_irradiance_w_m2": [900.0, 950.0, 900.0],
        }
    )
    warming_sim = simulate_indoor_temperature(
        climate_df=sunny_climate,
        geometry=geometry,
        material=material,
        thermal_params={"initial_indoor_temp": 12.0, "internal_heat_gain": 200.0},
    )
    w_temps = warming_sim["indoor_temperature"].tolist()
    assert w_temps[1] > w_temps[0]


# ==============================================================================
# 7. Material comparison -> returns DataFrame with correct columns
# ==============================================================================
def test_material_comparison():
    """Test that compare_materials produces the exact expected structure and valid data."""
    climate_df = pd.DataFrame(
        {
            "timestamp": [f"2026-01-15 {h:02d}:00" for h in range(6)],
            "outdoor_temperature_c": [-15.0, -16.0, -17.0, -18.0, -15.0, -12.0],
            "solar_irradiance_w_m2": [0.0, 0.0, 0.0, 50.0, 200.0, 400.0],
        }
    )
    geometry = {
        "length": 6.0,
        "width": 4.0,
        "height": 2.8,
        "wall_thickness": 0.25,
        "window_area": 3.0,
        "door_area": 1.8,
    }
    materials_list = [
        {"name": "Brick", "thermal_conductivity": 0.72, "absorptivity": 0.65},
        {"name": "Stone", "thermal_conductivity": 2.00, "absorptivity": 0.70},
        {"name": "Adobe/Earth", "thermal_conductivity": 0.60, "absorptivity": 0.70},
        {
            "name": "Brick + EPS",
            "absorptivity": 0.65,
            "layers": [
                {"thickness": 0.25, "k": 0.72},
                {"thickness": 0.05, "k": 0.035},
            ],
        },
    ]

    comp_df = compare_materials(
        climate_df=climate_df,
        geometry=geometry,
        materials_list=materials_list,
    )

    expected_cols = [
        "material",
        "avg_indoor_temp",
        "min_indoor_temp",
        "max_indoor_temp",
        "total_heat_loss",
        "solar_gain",
        "heating_requirement",
    ]

    assert isinstance(comp_df, pd.DataFrame)
    assert list(comp_df.columns) == expected_cols
    assert len(comp_df) == len(materials_list)
    assert not comp_df.isnull().any().any()

    # Stone has highest conductivity, so it should lose more heat than Adobe/Earth
    stone_loss = comp_df.loc[comp_df["material"] == "Stone", "total_heat_loss"].values[0]
    adobe_loss = comp_df.loc[comp_df["material"] == "Adobe/Earth", "total_heat_loss"].values[0]
    assert stone_loss > adobe_loss


# ==============================================================================
# 8. Comfort metrics and categorization
# ==============================================================================
def test_comfort_metrics_and_categories():
    """Test thermal comfort categories and aggregate metrics."""
    assert thermal_comfort_category(15.0) == "Cold"
    assert thermal_comfort_category(18.0) == "Comfortable"
    assert thermal_comfort_category(22.5) == "Comfortable"
    assert thermal_comfort_category(24.0) == "Comfortable"
    assert thermal_comfort_category(26.0) == "Hot"

    temps = [15.0, 16.0, 18.5, 20.0, 22.0, 25.0]
    metrics = calculate_comfort_metrics(temps, step_hours=1.0)

    assert metrics["avg_temp"] == pytest.approx(19.42, abs=0.01)
    assert metrics["min_temp"] == 15.0
    assert metrics["max_temp"] == 25.0
    # 18.5, 20.0, 22.0 are comfortable (3 out of 6 -> 50%)
    assert metrics["hours_in_comfort_range"] == 3.0
    assert metrics["percent_comfortable"] == 50.0


# ==============================================================================
# 9. ANSYS validation error calculations
# ==============================================================================
def test_ansys_validation():
    """Test ANSYS statistical comparison function."""
    py_df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "indoor_temperature": [20.0, 19.5, 19.0, 18.5],
        }
    )
    ansys_df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "indoor_temperature": [20.0, 19.0, 18.8, 18.0],
            "maximum_temperature": [20.5, 19.5, 19.2, 18.5],
            "minimum_temperature": [19.5, 18.5, 18.0, 17.5],
            "total_heat_flow": [800, 750, 700, 650],
        }
    )

    val_res = compare_with_ansys(py_df, ansys_df)
    assert "mean_absolute_error" in val_res
    assert "root_mean_square_error" in val_res
    assert "comparison_df" in val_res
    # Point errors: |20-20|=0, |19.5-19.0|=0.5, |19.0-18.8|=0.2, |18.5-18.0|=0.5
    # MAE = (0 + 0.5 + 0.2 + 0.5)/4 = 0.30
    assert val_res["mean_absolute_error"] == pytest.approx(0.3, abs=0.01)


# ==============================================================================
# 10. Database Persistence Tests
# ==============================================================================
def test_database_crud(tmp_path):
    """Test SQLite database operations (CRUD)."""
    db_file = str(tmp_path / "test_shelter.db")
    initialize_database(db_file)

    sim_id = save_simulation(
        project_name="Leh Passive Shelter",
        location="Leh, Ladakh",
        date="2026-01-15",
        climate_inputs={"tout_avg": -15.0},
        geometry_inputs={"length": 6.0, "width": 4.0},
        material="Adobe/Earth",
        duration=24.0,
        avg_temp=16.8,
        min_temp=12.2,
        max_temp=21.4,
        total_heat_loss=45.2,
        solar_gain=38.6,
        db_path=db_file,
    )

    assert sim_id > 0

    records = get_saved_simulations(db_file)
    assert len(records) == 1
    assert records[0]["project_name"] == "Leh Passive Shelter"

    fetched = get_simulation_by_id(sim_id, db_file)
    assert fetched is not None
    assert fetched["location"] == "Leh, Ladakh"
    assert fetched["avg_temp"] == 16.8

    del_success = delete_simulation(sim_id, db_file)
    assert del_success is True
    assert len(get_saved_simulations(db_file)) == 0


# ==============================================================================
# 11. Report Generation Tests (Excel and PDF)
# ==============================================================================
def test_report_generation():
    """Test programmatic generation of Excel and PDF reports."""
    from reports.report_generator import generate_excel_report, generate_pdf_report

    summary_data = {
        "Project": "Leh Passive Shelter",
        "avg_indoor_temp": 17.5,
        "min_indoor_temp": 12.0,
        "max_indoor_temp": 22.0,
        "total_heat_loss": "45.00",
        "solar_gain": "40.00",
        "heating_requirement": "12.50",
    }
    timeseries_df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "outdoor_temperature": [-18.0, -19.0, -20.0],
            "indoor_temperature": [15.0, 14.5, 14.0],
            "total_heat_loss": [500, 520, 540],
            "solar_gain": [0, 0, 0],
            "net_heat_gain": [-500, -520, -540],
        }
    )
    comp_df = pd.DataFrame(
        [
            {"material": "Brick", "avg_indoor_temp": 14.5, "total_heat_loss": 50.0},
            {"material": "Brick + EPS", "avg_indoor_temp": 17.8, "total_heat_loss": 18.0},
        ]
    )

    excel_bytes = generate_excel_report(summary_data, timeseries_df, comp_df)
    assert len(excel_bytes) > 0
    # Check Excel magic bytes (PK zip header)
    assert excel_bytes.startswith(b"PK")

    pdf_bytes = generate_pdf_report(
        project_name="Leh Passive Shelter",
        location="Leh, Ladakh",
        summary_data=summary_data,
        comfort_metrics={"hours_in_comfort_range": 12.0, "percent_comfortable": 50.0},
        geometry={"length": 6.0, "width": 4.0, "height": 2.8, "wall_thickness": 0.25},
        material={"name": "Brick + EPS", "thermal_conductivity": 0.72},
    )
    assert len(pdf_bytes) > 0
    # Check PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF")


# ==============================================================================
# 12. 3D Shelter Generation & Gallery Tests
# ==============================================================================
def test_3d_shelter_generation():
    """Test that create_3d_shelter_figure returns a valid, complete Plotly 3D Figure."""
    import plotly.graph_objects as go
    from models.shelter_3d import create_3d_shelter_figure

    fig_gable = create_3d_shelter_figure(
        length=6.0,
        width=4.0,
        height=2.8,
        wall_thickness=0.25,
        window_area=3.5,
        door_area=1.8,
        roof_type="gable",
        material_name="Adobe/Earth",
        sun_azimuth=180.0,
        sun_elevation=35.0,
        show_sun_ray=True,
    )
    assert isinstance(fig_gable, go.Figure)
    assert len(fig_gable.data) >= 4  # Ground, Walls, Roof, Window, Door, Sun Ray, Cardinal markers

    fig_flat = create_3d_shelter_figure(roof_type="flat", material_name="Stone")
    assert isinstance(fig_flat, go.Figure)

    fig_shed = create_3d_shelter_figure(roof_type="shed", material_name="Timber")
    assert isinstance(fig_shed, go.Figure)


def test_shelter_gallery_schema():
    """Test that assets/shelter_gallery.json exists and contains at least 6 valid presets."""
    import json
    gallery_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "shelter_gallery.json",
    )
    assert os.path.exists(gallery_path), "shelter_gallery.json must exist"

    with open(gallery_path, "r", encoding="utf-8") as f:
        gallery_data = json.load(f)

    assert isinstance(gallery_data, list)
    assert len(gallery_data) >= 6, "Must contain at least 6 shelter designs"

    required_keys = ["id", "name", "type", "description", "geometry", "material", "thermal_params"]
    for item in gallery_data:
        for k in required_keys:
            assert k in item, f"Preset {item.get('name')} missing key {k}"
        # Validate geometry values are positive
        geom = item["geometry"]
        assert geom["length"] > 0
        assert geom["width"] > 0
        assert geom["height"] > 0
        assert geom["wall_thickness"] > 0


