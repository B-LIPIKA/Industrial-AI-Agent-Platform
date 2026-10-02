"""
Schema-Aware Tools for Industrial Data Discovery & SQL Querying
"""

from typing import Dict, List, Any
from database.db_manager import get_schema_summary, execute_read_query, DEFAULT_DB_PATH


def inspect_database_schema(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Returns database schema layout with tables and column types."""
    return {
        "status": "success",
        "tables": get_schema_summary(db_path)
    }


def query_sensor_database(sql_query: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Executes a read-only SELECT query against the industrial plant database."""
    try:
        results = execute_read_query(sql_query, db_path)
        return {
            "status": "success",
            "row_count": len(results),
            "data": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def discover_available_time_ranges(equipment_id: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Discovers minimum and maximum timestamp bounds for an equipment asset."""
    query = f"""
    SELECT MIN(timestamp) as earliest_time, MAX(timestamp) as latest_time, COUNT(*) as reading_count
    FROM sensor_telemetry
    WHERE equipment_id = '{equipment_id}';
    """
    try:
        rows = execute_read_query(query, db_path)
        return {
            "status": "success",
            "equipment_id": equipment_id,
            "bounds": rows[0] if rows else {}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
