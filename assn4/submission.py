"""
CNN Mini Assignment - Student Submission (submission.py)

⚠️ Rules
- ONLY modify this file.
- You may ONLY use NumPy (no PyTorch, TensorFlow, etc.).
- Do NOT change function names, arguments, or return types.
- You MAY add helper functions if needed.
"""

import numpy as np
from typing import Dict


# ============================================================
# Problem 1: Basic CNN Building Blocks (NumPy)
# ============================================================

# ------------------------------------------------------------
# Problem 1a
# ------------------------------------------------------------
def problem1a(H: int, W: int, k_H: int, k_W: int, P: int, S: int) -> tuple:
    """
    Computes output spatial dimensions for a convolution layer.

    Args:
        H (int): Input height
        W (int): Input width
        k_H (int): Kernel height
        k_W (int): Kernel width
        P (int): Padding
        S (int): Stride

    Returns:
        (outH, outW): tuple of ints
    """
    raise NotImplementedError

# ------------------------------------------------------------
# Problem 1b
# ------------------------------------------------------------
def conv2d(x: np.ndarray, w: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """
    Problem 1b: Implement 2D convolution (no batch dimension).

    Args:
        x : np.ndarray
            Input feature map of shape (C_in, H, W)
        w : np.ndarray
            Convolution kernels of shape (C_out, C_in, kH, kW)
        padding : int
            Zero-padding (added to all sides of spatial dimensions)
        stride : int
            Stride for the convolution operation

    Returns:
        out : np.ndarray
            Output feature map of shape (C_out, outH, outW)

    Notes:
    - Bias is *NOT* added here. Bias will be added in the forward pass.
    - You must implement padding and stride manually.
    - Use for-loops (no broadcasting tricks).
    """
    raise NotImplementedError



# ------------------------------------------------------------
# Problem 2 — ReLU
# ------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    """
    Problem 2: Implement the ReLU activation function.

    Args:
        x : np.ndarray
            Any shape NumPy array

    Returns:
        out : np.ndarray
            Same shape as x, with negative values replaced by 0
    """
    raise NotImplementedError



# ------------------------------------------------------------
# Problem 3 — Max Pooling
# ------------------------------------------------------------
def maxpool2d(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """
    Problem 3: 2D max pooling (no batch dimension).

    Args:
        x : np.ndarray
            Input feature map of shape (C, H, W)
        pool_size : int
            Size of pooling window (pool_size x pool_size)
        stride : int
            How far the window moves each step

    Returns:
        out : np.ndarray
            Max-pooled feature map of shape (C, outH, outW)
    """
    raise NotImplementedError



# ------------------------------------------------------------
# Problem 4 — Flatten
# ------------------------------------------------------------
def flatten(x: np.ndarray) -> np.ndarray:
    """
    Problem 4: Flatten an input tensor.

    Args:
        x : np.ndarray
            Input array of any shape (typically (C, H, W))

    Returns:
        out : np.ndarray
            Flattened (1D) vector
    """
    raise NotImplementedError



# ------------------------------------------------------------
# Problem 5 — Fully Connected (FC) Layer
# ------------------------------------------------------------
def fc2d(f: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Problem 5: Fully Connected layer.

    Args:
        f : np.ndarray
            Flattened input vector of shape (N,)
        W : np.ndarray
            Weight matrix of shape (out_dim, N)
        b : np.ndarray
            Bias of shape (out_dim,)

    Returns:
        logits : np.ndarray
            Output scores of shape (out_dim,)
    """
    raise NotImplementedError



# ============================================================
# Problem 6: Full Tiny CNN Forward Pass
# ============================================================

def simple_cnn_forward(x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Problem 6: Forward pass of the Tiny CNN.

    Architecture:
        x : (1, 28, 28)
            -> conv2d (C_out = 4, kernel 3×3, padding=1, stride=1)
            -> + conv_b
            -> ReLU
            -> MaxPool2d (2×2, stride 2)   -> (4, 14, 14)
            -> Flatten                    -> (4*14*14,)
            -> FC layer (10 outputs)

    Args:
        x : np.ndarray
            Input image, shape (1, 28, 28)

        params : Dict[str, np.ndarray]
            Dictionary containing pretrained parameters:
                "conv_w": (4, 1, 3, 3)
                "conv_b": (4,)
                "fc_W":   (10, 4*14*14)
                "fc_b":   (10,)

    Returns:
        logits : np.ndarray
            Class scores of shape (10,)
    """
    raise NotImplementedError
