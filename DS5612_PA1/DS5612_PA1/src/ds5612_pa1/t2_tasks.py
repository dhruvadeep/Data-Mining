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
    """Ignore the below implementation."""
    # temp = ""
    # # print(python_dict)
    # for i in python_dict:
    #     if isinstance(python_dict[i], dict):
    #         temp += "\n"
    #         temp += f"[{i}]\n"
    #         for j in python_dict[i]:

    #             if isinstance(python_dict[i][j], dict):
    #                 temp += f"\n{t2c_python_dict_to_toml_string(python_dict[i][j])}"
    #             temp += j + "= " + f"{"" if isinstance(python_dict[i][j], int)
    #                       or isinstance(python_dict[i][j], list) else "'"}{
    #                           python_dict[i][j]}{"" if isinstance(python_dict[i][j], int)
    #                       or isinstance(python_dict[i][j], list) else "'"}\n"
    #     else:
    #         temp += i + "= " + f"{"" if isinstance(python_dict[i], int) or
    #                 isinstance(python_dict[i], list) else "'"}{
    #                       python_dict[i]}{"" if isinstance(python_dict[i], int)
    #                           or isinstance(python_dict[i], list) else "'"}\n"
    # # print(temp)
    # return temp
    temp = ""

    for key, value in python_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            temp += f"\n[{full_key}]\n"
            temp += t2c_python_dict_to_toml_string(value, full_key)
        elif isinstance(value, list or int):
            temp += f"{key} = {value}\n"
        else:
            temp += f"{key} = '{value}'\n"

    return temp


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
