"""
Utility functions for CNN mini assignment (NumPy-only).
"""

import pickle
import numpy as np
from typing import Tuple


def load_mnist(path: str = "data/mnist.pkl") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Light MNIST-like dataset loader.

    Returns:
        train_X: (N_train, 1, 28, 28)
        train_y: (N_train,)
        test_X:  (N_test, 1, 28, 28)
        test_y:  (N_test,)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["train_X"], data["train_y"], data["test_X"], data["test_y"]
