# GiG
from typing import Any


# This function accepts python_dict where all keys are strings
# and the values are strings, integers and complex combination of lists and dictionaries
# It outputs a string that is a valid TOML string
# Note: the python dictionary can be nested (e.g. list of list, dict of dicts etc)
# So design this as a recursive function
# Hint: for nested dictionaries, it is simpler to output as fully named tables.
# For example, if there is a dict inner inside outer,
# then something like
# [outer]
# [outer.inner]
# is the way to go.
# This can be done by using the optional parent_key variable


def t2c_python_dict_to_toml_string(python_dict: dict[str, Any], parent_key: str = "") -> str:
    def serialize_value(value):
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, list):
            list_elements = [serialize_value(item) for item in value]
            return "[" + ", ".join(list_elements) + "]"
        elif isinstance(value, dict):
            return ""
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    toml_string = ""

    for key, value in python_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            toml_string += f"\n[{full_key}]\n"
            toml_string += t2c_python_dict_to_toml_string(value, full_key)
        else:
            toml_string += f"{key} = {serialize_value(value)}\n"

    return toml_string

def t2d_python_dict_to_yaml_string(python_dict: dict[str, Any], indent: int = 0) -> str:
    """Convert a Python dictionary to a YAML-formatted string."""
    temp = ""
    indent_space = "  " * indent  # Create indentation level

    for key, value in python_dict.items():
        if isinstance(value, dict):
            # Recursive case for nested dictionaries
            temp += f"{indent_space}{key}:\n"
            temp += t2d_python_dict_to_yaml_string(value, indent + 1)
        elif isinstance(value, list):
            # Handle list by iterating over its elements
            temp += f"{indent_space}{key}:\n"
            for item in value:
                if isinstance(item, dict):
                    temp += t2d_python_dict_to_yaml_string(item, indent + 1)
                else:
                    temp += f"{indent_space}- {item}\n"
        else:
            # Base case for strings, integers, and other simple values
            temp += f"{indent_space}{key}: {value}\n"

    return temp
