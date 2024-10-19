# GiG

import tomllib
from typing import Any

import pytest
import yaml

from ds5612_pa1 import t2_tasks


# Example usage
t2_simple_dict = {
    "name": "John Doe",
    "age": 30,
    "hobbies": ["reading", "cycling", "coding"],
    "address": {"street": "123 Main St", "city": "Anytown", "zipcode": 12345},
}

t2_complex_dict = {
    "name": "Complex Example",
    "person": {
        "name": "John Doe",
        "age": 30,
        "hobbies": ["reading", "cycling"],
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zipcode": 12345,
        },
    },
}


def compare_dict_toml(python_dict: dict[str, Any], toml_string: str) -> bool:
    try:
        # Parse TOML string to a dictionary
        toml_dict = tomllib.loads(toml_string)
        print(toml_dict)

        # Perform deep comparison
        return deep_compare(python_dict, toml_dict)
    except tomllib.TOMLDecodeError:
        print("Error: Invalid TOML string")
        return False


def compare_dict_yaml(python_dict: dict[str, Any], yaml_string: str) -> bool:
    try:
        # Parse TOML string to a dictionary
        yaml_dict = yaml.safe_load(yaml_string)
        print(yaml_dict)

        # Perform deep comparison
        return deep_compare(python_dict, yaml_dict)
    except yaml.YAMLError:
        print("Error: Invalid YAML string")
        return False


# We are doing two noqa.
# Usually, this is frowned, but this is an exception, where this is fine.
def deep_compare(d1: Any, d2: Any) -> bool:  # noqa: ANN401
    if type(d1) != type(d2):  # pylint: disable=unidiomatic-typecheck  # noqa: E721
        return False

    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            return False
        return all(deep_compare(d1[key], d2[key]) for key in d1)

    if isinstance(d1, list):
        if len(d1) != len(d2):
            return False
        return all(deep_compare(v1, v2) for v1, v2 in zip(d1, d2, strict=False))

    return d1 == d2


@pytest.mark.t2c()
def test_t2c_dict_to_toml() -> None:
    t2c_output = t2_tasks.t2c_python_dict_to_toml_string(t2_simple_dict)
    assert compare_dict_toml(t2_simple_dict, t2c_output)

    t2c_output = t2_tasks.t2c_python_dict_to_toml_string(t2_complex_dict)
    assert compare_dict_toml(t2_complex_dict, t2c_output)


@pytest.mark.t2d()
def test_t2d_dict_to_yaml() -> None:
    t2c_output = t2_tasks.t2d_python_dict_to_yaml_string(t2_simple_dict)
    assert compare_dict_yaml(t2_simple_dict, t2c_output)

    t2c_output = t2_tasks.t2d_python_dict_to_yaml_string(t2_complex_dict)
    assert compare_dict_yaml(t2_complex_dict, t2c_output)
