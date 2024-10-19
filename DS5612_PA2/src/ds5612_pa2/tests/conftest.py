# GiG

import math
from typing import Any


MAX_ABS_TOLERANCE = 0.01


# This compares two dictionaries with same set of keys
def compare_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    for key in dict1:
        v1, v2 = dict1[key], dict2[key]
        if isinstance(v1, float) or isinstance(v2, float):
            v1, v2 = float(v1), float(v2)
            if not math.isclose(v1, v2, abs_tol=MAX_ABS_TOLERANCE):
                print(v1, v2)
                return False

        elif v1 != v2:
            return False

    return True


def compare_lists_of_dicts(list1: list[dict], list2: list[dict]) -> bool:
    if len(list1) != len(list2):
        return False

    for dict1, dict2 in zip(list1, list2, strict=False):  # noqa: SIM110
        if not compare_dicts(dict1, dict2):
            return False

    return True
