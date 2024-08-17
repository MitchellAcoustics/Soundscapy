import csv
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from loguru import logger


class ResultStorage(ABC):
    @abstractmethod
    def store(self, results: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def retrieve(self) -> List[Dict[str, Any]]:
        pass


class JSONResultStorage(ResultStorage):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def store(self, results: List[Dict[str, Any]]):
        try:
            with open(self.file_path, "w") as f:
                json.dump(results, f, indent=2)
        except IOError as e:
            logger.error(
                f"Failed to write JSON results to {self.file_path}. Error: {str(e)}"
            )

    def retrieve(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(
                f"Failed to read JSON results from {self.file_path}. Error: {str(e)}"
            )
            return []


class CSVResultStorage(ResultStorage):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def store(self, results: List[Dict[str, Any]]):
        if not results:
            logger.warning("No results to store in CSV")
            return

        try:
            with open(self.file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        except IOError as e:
            logger.error(
                f"Failed to write CSV results to {self.file_path}. Error: {str(e)}"
            )

    def retrieve(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except IOError as e:
            logger.error(
                f"Failed to read CSV results from {self.file_path}. Error: {str(e)}"
            )
            return []


class SQLiteResultStorage(ResultStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.table_name = "results"

    def store(self, results: List[Dict[str, Any]]):
        if not results:
            logger.warning("No results to store in SQLite database")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            columns = ", ".join([f"{key} TEXT" for key in results[0].keys()])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({columns})")

            # Insert results
            placeholders = ", ".join(["?" for _ in results[0]])
            cursor.executemany(
                f"INSERT INTO {self.table_name} VALUES ({placeholders})",
                [tuple(result.values()) for result in results],
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to store results in SQLite database. Error: {str(e)}")

    def retrieve(self) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {self.table_name}")
            rows = cursor.fetchall()

            conn.close()

            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(
                f"Failed to retrieve results from SQLite database. Error: {str(e)}"
            )
            return []


def create_result_storage(storage_type: str, file_path: str) -> ResultStorage:
    if storage_type == "json":
        return JSONResultStorage(file_path)
    elif storage_type == "csv":
        return CSVResultStorage(file_path)
    elif storage_type == "sqlite":
        return SQLiteResultStorage(file_path)
    else:
        logger.error(f"Unknown storage type: {storage_type}", "INVALID_STORAGE_TYPE")
        raise ValueError(f"Unknown storage type: {storage_type}")


# Example usage
# storage = create_result_storage('json', 'results.json')
# results = [{'metric1': 0.5, 'metric2': 0.7}, {'metric1': 0.6, 'metric2': 0.8}]
# storage.store(results)
# retrieved_results = storage.retrieve()
