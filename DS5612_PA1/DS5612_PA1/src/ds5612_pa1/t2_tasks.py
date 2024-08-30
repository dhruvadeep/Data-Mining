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
    return ""


def t2d_python_dict_to_yaml_string(python_dict: dict[str, Any], indent: int = 0) -> str:
    return ""
