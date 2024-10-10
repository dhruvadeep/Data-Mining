# GiG

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, make_scorer


# Make changes only inside cost_function.
# Given the cost matrix, cost_00, cost_01, cost_10, cost_11,
# output the classifier cost score.
# You will return the closure of the scorer.
# Your code should not be more than 3 lines.
def create_cost_based_scorer(
    cost_00: float, cost_01: float, cost_10: float, cost_11: float
) -> Callable:
    def cost_function(y: NDArray, y_pred: NDArray, **kwargs) -> float:  # noqa: ANN003, ARG001
        cm = confusion_matrix(y, y_pred)
        cost = (
            cm[0, 0] * cost_00
            + cm[0, 1] * cost_01
            + cm[1, 0] * cost_10
            + cm[1, 1] * cost_11
        )
        return cost

    # Using meta programming, we have returned a function that is frozen
    # to use cost_00, cost_01, cost_10, and cost_11
    # when we call with different values for
    # cost_00, cost_01, cost_10, and cost_11, we will get a different function
    # This function can then be sent to make_scorer to get outputs
    return cost_function


# This function converts cost_function into a scorer function
# that can be used for training classifiers, hyper parameter tuning and error estimation.
# All you need to do is call the make_scorer and tell it that it is a loss function.
# So classifiers with higher values are worse than those with lower values.
# Your code should not be more than 1-2 lines.
def create_loss_function_scorer(
    cost_00: float, cost_01: float, cost_10: float, cost_11: float
) -> Callable:
    
    return make_scorer(
        create_cost_based_scorer(cost_00, cost_01, cost_10, cost_11), greater_is_better=False
    )

if __name__ == "__main__":
    pass
