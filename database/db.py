"""SQLite database storage module for ThermoShelter360.

Manages persistent storage for shelter simulation records, climate configurations,
geometry specifications, and thermal performance metrics using parameterized SQL.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "thermoshelter.db"
)


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection object with row_factory set to Row.
    """
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite schema for storing simulation runs.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    date TEXT NOT NULL,
                    climate_inputs TEXT NOT NULL,
                    geometry_inputs TEXT NOT NULL,
                    material TEXT NOT NULL,
                    duration REAL NOT NULL,
                    avg_temp REAL NOT NULL,
                    min_temp REAL NOT NULL,
                    max_temp REAL NOT NULL,
                    total_heat_loss REAL NOT NULL,
                    solar_gain REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        conn.close()


def save_simulation(
    project_name: str,
    location: str,
    date: str,
    climate_inputs: Union[Dict[str, Any], str],
    geometry_inputs: Union[Dict[str, Any], str],
    material: Union[Dict[str, Any], str],
    duration: float,
    avg_temp: float,
    min_temp: float,
    max_temp: float,
    total_heat_loss: float,
    solar_gain: float,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Save a completed thermal simulation record into SQLite.

    Args:
        project_name: Descriptive name for the shelter design project.
        location: Geographic region or site (e.g. "Leh, Ladakh").
        date: Simulation or project date string.
        climate_inputs: Dict or JSON string of climate parameters.
        geometry_inputs: Dict or JSON string of geometry parameters.
        material: Name or dictionary describing the wall material assembly.
        duration: Total simulation duration in hours.
        avg_temp: Average predicted indoor temperature (C).
        min_temp: Minimum predicted indoor temperature (C).
        max_temp: Maximum predicted indoor temperature (C).
        total_heat_loss: Total cumulative heat loss (kWh or MJ).
        solar_gain: Total cumulative solar heat gain (kWh or MJ).
        db_path: Path to SQLite database file.

    Returns:
        Inserted record ID (int).
    """
    initialize_database(db_path)

    climate_str = (
        json.dumps(climate_inputs)
        if isinstance(climate_inputs, (dict, list))
        else str(climate_inputs)
    )
    geom_str = (
        json.dumps(geometry_inputs)
        if isinstance(geometry_inputs, (dict, list))
        else str(geometry_inputs)
    )
    mat_str = (
        json.dumps(material)
        if isinstance(material, (dict, list))
        else str(material)
    )

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO simulations (
                    project_name, location, date, climate_inputs,
                    geometry_inputs, material, duration,
                    avg_temp, min_temp, max_temp, total_heat_loss, solar_gain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_name,
                    location,
                    date,
                    climate_str,
                    geom_str,
                    mat_str,
                    float(duration),
                    float(avg_temp),
                    float(min_temp),
                    float(max_temp),
                    float(total_heat_loss),
                    float(solar_gain),
                ),
            )
            sim_id = cursor.lastrowid
            return int(sim_id)
    finally:
        conn.close()


def get_saved_simulations(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all saved simulation summary records, ordered newest first.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        List of dictionary records.
    """
    initialize_database(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, project_name, location, date, material, duration,
                   avg_temp, min_temp, max_temp, total_heat_loss, solar_gain, created_at
            FROM simulations
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_simulation_by_id(
    sim_id: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific simulation record by its ID.

    Args:
        sim_id: Database record ID.
        db_path: Path to SQLite database file.

    Returns:
        Dictionary representation of the simulation, or None if not found.
    """
    initialize_database(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM simulations WHERE id = ?", (sim_id,))
        row = cursor.fetchone()
        if row is None:
            return None

        data = dict(row)
        # Parse JSON fields safely
        try:
            data["climate_inputs"] = json.loads(data["climate_inputs"])
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            data["geometry_inputs"] = json.loads(data["geometry_inputs"])
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            data["material"] = json.loads(data["material"])
        except (json.JSONDecodeError, TypeError):
            pass

        return data
    finally:
        conn.close()


def delete_simulation(sim_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a simulation record from SQLite by ID.

    Args:
        sim_id: Database record ID to delete.
        db_path: Path to SQLite database file.

    Returns:
        True if deleted, False if record did not exist.
    """
    initialize_database(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()
