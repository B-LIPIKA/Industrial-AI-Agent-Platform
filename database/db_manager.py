"""
Industrial Database Manager & Schema Repository
Provides SQLite database initialization, synthetic industrial equipment telemetry seeding,
and schema-aware SQL inspection tools.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any


DEFAULT_DB_PATH = "industrial_plant.db"


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes tables and populates baseline sensor data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS equipment (
        equipment_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        equipment_type TEXT NOT NULL,
        location_zone TEXT NOT NULL,
        status TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_telemetry (
        reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        metric_name TEXT NOT NULL,
        value REAL NOT NULL,
        unit TEXT NOT NULL,
        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL,
        logged_at DATETIME NOT NULL,
        severity TEXT NOT NULL,
        description TEXT NOT NULL,
        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
    );
    """)

    # Seed baseline records if empty
    cur.execute("SELECT COUNT(*) FROM equipment")
    if cur.fetchone()[0] == 0:
        equipment_records = [
            ("PUMP-101", "Main Inflow Slurry Pump", "Centrifugal Pump", "Zone-A", "OPERATIONAL"),
            ("PUMP-102", "Auxiliary Recirculation Pump", "Centrifugal Pump", "Zone-A", "WARNING"),
            ("TURBINE-01", "Primary Hydro Turbine", "Turbine", "Zone-B", "OPERATIONAL"),
            ("COMPRESSOR-04", "Air Supply Compressor", "Reciprocating Compressor", "Zone-C", "OPERATIONAL")
        ]
        cur.executemany("INSERT INTO equipment VALUES (?, ?, ?, ?, ?)", equipment_records)

        # Seed telemetry
        telemetry = [
            ("PUMP-101", "2026-10-01 08:00:00", "vibration_rms", 2.4, "mm/s"),
            ("PUMP-101", "2026-10-01 09:00:00", "vibration_rms", 2.6, "mm/s"),
            ("PUMP-101", "2026-10-01 10:00:00", "vibration_rms", 4.8, "mm/s"),  # anomaly
            ("PUMP-101", "2026-10-01 11:00:00", "vibration_rms", 2.7, "mm/s"),
            ("PUMP-101", "2026-10-01 12:00:00", "vibration_rms", 2.5, "mm/s"),
            ("PUMP-102", "2026-10-01 08:00:00", "temperature", 65.4, "deg_C"),
            ("PUMP-102", "2026-10-01 10:00:00", "temperature", 82.1, "deg_C"),  # elevated temp
            ("TURBINE-01", "2026-10-01 08:00:00", "power_kw", 1450.0, "kW"),
            ("TURBINE-01", "2026-10-01 12:00:00", "power_kw", 1485.0, "kW"),
        ]
        cur.executemany("INSERT INTO sensor_telemetry (equipment_id, timestamp, metric_name, value, unit) VALUES (?, ?, ?, ?, ?)", telemetry)

        # Seed maintenance log
        logs = [
            ("PUMP-101", "2026-09-25 14:00:00", "LOW", "Routine bearing lubrication completed."),
            ("PUMP-102", "2026-09-28 11:30:00", "HIGH", "Coolant line pressure drop noted; scheduled for valve inspection.")
        ]
        cur.executemany("INSERT INTO maintenance_logs (equipment_id, logged_at, severity, description) VALUES (?, ?, ?, ?)", logs)

    conn.commit()
    conn.close()


def get_schema_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, List[Dict[str, str]]]:
    """Inspects SQLite system catalog and returns schema metadata."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cur.fetchall()]

    schema = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({table});")
        columns = cur.fetchall()
        schema[table] = [{"column": col[1], "type": col[2]} for col in columns]

    conn.close()
    return schema


def execute_read_query(query: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Executes safe read-only SQL query and returns rows as dictionaries."""
    sanitized = query.strip()
    if not sanitized.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sanitized)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
