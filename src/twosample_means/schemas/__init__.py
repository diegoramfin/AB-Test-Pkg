"""Machine-readable report schemas bundled with the package.

The experiment result schema is versioned independently from the Python
package. ``validate_experiment_json`` validates rendered JSON against the
bundled schema so future report changes cannot silently drift from the
declared contract.

The bundled schemas use only a small JSON Schema keyword subset (``type``,
``const``, ``enum``, ``required``, ``properties``, ``items``, ``pattern``,
``minimum``, ``maximum``). The validator here implements exactly that subset
so the package does not require a JSON Schema runtime dependency.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

EXPERIMENT_RESULT_SCHEMA_VERSION = "experiment-result-v1"
_SCHEMA_FILENAME = "experiment-result-v1.json"


class SchemaValidationError(ValueError):
    """Raised when a rendered report violates its declared JSON Schema."""


def load_experiment_schema() -> dict[str, Any]:
    """Load and return the bundled experiment result schema."""
    path = files(__package__).joinpath(_SCHEMA_FILENAME)
    parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], parsed)


def validate_experiment_json(document: str) -> dict[str, Any]:
    """Validate a rendered experiment report against the bundled schema.

    Parameters
    ----------
    document:
        JSON text produced by ``render_experiment_json`` or an equivalent
        report document for the same schema version.

    Returns
    -------
    dict[str, Any]
        The parsed document when validation succeeds.

    Raises
    ------
    SchemaValidationError
        If the document is not valid JSON or violates the schema.
    """
    try:
        data: Any = json.loads(document)
    except json.JSONDecodeError as error:
        raise SchemaValidationError(
            f"experiment report is not valid JSON: {error}"
        ) from error
    schema = load_experiment_schema()
    _validate(data, schema, "/")
    return cast(dict[str, Any], data)


def _validate(data: Any, schema: dict[str, Any], path: str) -> None:
    """Validate one value against a schema node using the bundled subset."""
    schema_type = schema.get("type")
    if schema_type is not None and not _matches_type(data, schema_type):
        raise SchemaValidationError(
            f"{path} expected {schema_type}, got {_type_name(data)}"
        )
    if isinstance(data, str):
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, data) is None:
            raise SchemaValidationError(
                f"{path} does not match pattern {pattern!r}"
            )
    if isinstance(data, int | float) and not isinstance(data, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            raise SchemaValidationError(
                f"{path} value {data} is below minimum {minimum}"
            )
        if maximum is not None and data > maximum:
            raise SchemaValidationError(
                f"{path} value {data} is above maximum {maximum}"
            )
    const = schema.get("const")
    if const is not None and data != const:
        raise SchemaValidationError(
            f"{path} must equal {const!r}, got {data!r}"
        )
    enum = schema.get("enum")
    if enum is not None and data not in enum:
        raise SchemaValidationError(
            f"{path} must be one of {enum}, got {data!r}"
        )
    if isinstance(data, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key in data:
                    _validate(data[key], property_schema, f"{path}{key}")
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in data:
                    raise SchemaValidationError(
                        f"{path} missing required property {key!r}"
                    )
    if isinstance(data, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(data):
                _validate(item, items, f"{path}{index}/")


def _matches_type(data: Any, schema_type: str | list[str]) -> bool:
    """Return whether a JSON value matches a JSON Schema ``type`` keyword."""
    if isinstance(schema_type, list):
        return any(_matches_type(data, item) for item in schema_type)
    if schema_type == "null":
        return data is None
    if schema_type == "string":
        return isinstance(data, str)
    if schema_type == "boolean":
        return isinstance(data, bool)
    if schema_type == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if schema_type == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if schema_type == "object":
        return isinstance(data, dict)
    if schema_type == "array":
        return isinstance(data, list)
    return False


def _type_name(data: Any) -> str:
    """Return the JSON type name of a value for error messages."""
    if data is None:
        return "null"
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, str):
        return "string"
    if isinstance(data, int):
        return "integer"
    if isinstance(data, float):
        return "number"
    if isinstance(data, list):
        return "array"
    if isinstance(data, dict):
        return "object"
    return type(data).__name__


def schema_path() -> Path:
    """Return the bundled schema file location."""
    return Path(str(files(__package__).joinpath(_SCHEMA_FILENAME)))
