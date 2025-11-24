#!/usr/bin/env python3
"""
Autograder for CNN Mini Assignment (NumPy-only).
"""

import graderUtil
import numpy as np

grader = graderUtil.Grader()
submission = grader.load("submission")

try:
    import solution
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False


def np_allclose(a, b, tol=1e-4):
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=tol, rtol=0.0)


############################################################
# Problem 1a: Output size
############################################################

def student_test_problem1a():
    # Example #1
    H, W = submission.problem1a(28, 28, 3, 3, 1, 1)
    grader.requireIsTrue((H, W) == (28, 28))

    # Example #2
    H, W = submission.problem1a(10, 10, 5, 5, 0, 1)
    grader.requireIsTrue((H, W) == (6, 6))


def hidden_test_problem1a():
    np.random.seed(SEED)
    for _ in range(5):
        H = np.random.randint(5, 50)
        W = np.random.randint(5, 50)
        k = np.random.randint(1, 6)
        P = np.random.randint(0, 3)
        S = np.random.randint(1, 3)

        pred_H, pred_W = submission.problem1a(H, W, k, k, P, S)
        ans_H, ans_W = solution.problem1a(H, W, k, k, P, S)
        grader.requireIsTrue(pred_H == ans_H and pred_W == ans_W)


if solution_exist:
    grader.addBasicPart(
        "1a-output-size-hidden",
        hidden_test_problem1a,
        maxPoints=5,
        maxSeconds=1,
        description="Hidden test for output size."
    )
else:
    grader.addBasicPart(
        "1a-output-size-basic",
        student_test_problem1a,
        maxPoints=1,
        maxSeconds=1,
        description="Example tests for output size (0 points)."
    )


############################################################
# Problem 1b: Convolution
############################################################

def student_test_conv():
    # -------------------------------
    # Example 1
    # -------------------------------
    x = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    w = np.ones((1, 1, 3, 3), dtype=np.float32)

    out = submission.conv2d(x, w, padding=0, stride=1)

    expected1 = np.array(
        [[[ 54.,  63.,  72.],
          [ 99., 108., 117.],
          [144., 153., 162.]]],
        dtype=np.float32
    )
    grader.requireIsTrue(np_allclose(out, expected1))

    # -------------------------------
    # Example 2 (fixed)
    # -------------------------------
    x2 = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    w2 = np.ones((1, 1, 2, 2), dtype=np.float32)

    out2 = submission.conv2d(x2, w2, padding=0, stride=1)

    expected2 = np.array(
        [[[2., 1., 0.],
          [1., 2., 1.],
          [0., 1., 2.]]],
        dtype=np.float32
    )
    grader.requireIsTrue(np_allclose(out2, expected2))



def hidden_test_conv():
    np.random.seed(SEED)
    for _ in range(5):
        x = np.random.randn(3, 8, 8).astype(np.float32)
        w = np.random.randn(4, 3, 3, 3).astype(np.float32)
        pred = submission.conv2d(x, w, padding=1, stride=2)
        ans = solution.conv2d(x, w, padding=1, stride=2)
        grader.requireIsTrue(np_allclose(pred, ans))


if solution_exist:
    grader.addBasicPart(
        "1b-conv-hidden",
        hidden_test_conv,
        maxPoints=8,
        maxSeconds=5,
        description="Hidden convolution test."
    )
else:
    grader.addBasicPart(
        "1b-conv-basic",
        student_test_conv,
        maxPoints=1,
        maxSeconds=3,
        description="Example convolution tests (0 points)."
    )


############################################################
# Problem 2: ReLU
############################################################

def student_test_relu():
    out = submission.relu(np.array([-1, 0, 3]))
    grader.requireIsTrue(np.all(out == np.array([0, 0, 3])))

    out2 = submission.relu(np.array([[-5, 2], [3, -1]]))
    grader.requireIsTrue(np.all(out2 == np.array([[0, 2], [3, 0]])))


def hidden_test_relu():
    for _ in range(5):
        np.random.seed(SEED)
        x = np.random.randn(5, 5)
        grader.requireIsTrue(np_allclose(submission.relu(x), solution.relu(x)))


if solution_exist:
    grader.addBasicPart(
        "2-relu-hidden",
        hidden_test_relu,
        maxPoints=2,
        maxSeconds=3,
        description="Hidden ReLU test."
    )
else:
    grader.addBasicPart(
        "2-relu-basic",
        student_test_relu,
        maxPoints=1,
        maxSeconds=1,
        description="Example ReLU tests (0 points)."
    )


############################################################
# Problem 3: Max Pooling
############################################################

def student_test_pool():
    x = np.array(
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 9, 1, 1],
          [0, 0, 0, 0]]],
        dtype=np.float32,
    )
    out = submission.maxpool2d(x, 2, 2)
    expected = np.array([[[6, 8],
                      [9, 1]]], dtype=np.float32)
    grader.requireIsTrue(np_allclose(out, expected))



def hidden_test_pool():
    for _ in range(5):
        np.random.seed(SEED)
        x = np.random.randn(3, 6, 6)
        grader.requireIsTrue(
            np_allclose(submission.maxpool2d(x), solution.maxpool2d(x))
        )


if solution_exist:
    grader.addBasicPart(
        "3-pool-hidden",
        hidden_test_pool,
        maxPoints=5,
        maxSeconds=1,
        description="Hidden pooling test."
    )
else:
    grader.addBasicPart(
        "3-pool-basic",
        student_test_pool,
        maxPoints=1,
        maxSeconds=1,
        description="Example pooling tests (0 points)."
    )


############################################################
# Problem 4: Flatten
############################################################

def student_test_flatten():
    x = np.array([[1, 2], [3, 4]])
    out = submission.flatten(x)
    expected = np.array([1, 2, 3, 4])
    grader.requireIsTrue(np_allclose(out, expected))



def hidden_test_flatten():
    for _ in range(5):
        x = np.random.randn(3, 4, 5)
        grader.requireIsTrue(np_allclose(
            submission.flatten(x),
            solution.flatten(x)
        ))


if solution_exist:
    grader.addBasicPart(
        "4-flatten-hidden",
        hidden_test_flatten,
        maxPoints=2,
        maxSeconds=1,
        description="Hidden flatten test."
    )
else:
    grader.addBasicPart(
        "4-flatten-basic",
        student_test_flatten,
        maxPoints=1,
        maxSeconds=1,
        description="Example flatten tests (0 points)."
    )


############################################################
# Problem 5: FC Layer
############################################################

def student_test_fc():
    x = np.arange(8).astype(np.float32)
    W = np.ones((3, 8), dtype=np.float32)
    b = np.zeros(3)
    out = submission.fc2d(x, W, b)

    expected = np.array([
        np.sum(x),
        np.sum(x),
        np.sum(x)
    ])
    grader.requireIsTrue(np_allclose(out, expected))



def hidden_test_fc():
    for _ in range(5):
        x = np.random.randn(8)
        W = np.random.randn(3, 8)
        b = np.random.randn(3)
        grader.requireIsTrue(np_allclose(
            submission.fc2d(x, W, b),
            solution.fc2d(x, W, b)
        ))


if solution_exist:
    grader.addBasicPart(
        "5-fc-hidden",
        hidden_test_fc,
        maxPoints=2,
        maxSeconds=1,
        description="Hidden FC test."
    )
else:
    grader.addBasicPart(
        "5-fc-basic",
        student_test_fc,
        maxPoints=1,
        maxSeconds=1,
        description="Example FC tests (0 points)."
    )


############################################################
# Problem 6: Forward Pass
############################################################

def student_test_forward():
    # Very small case to avoid pooling issues
    x = np.array([[
        [1, 2],
        [3, 4]
    ]], dtype=np.float32)

    params = {
        "conv_w": np.array([[
            [[1, 0],
             [0, 1]]
        ]], dtype=np.float32),

        "conv_b": np.array([0.0], dtype=np.float32),

        "fc_W": np.array([[1]], dtype=np.float32),
        "fc_b": np.array([0.0], dtype=np.float32),
    }

    out = submission.simple_cnn_forward(x, params)
    expected = np.array([5.0], dtype=np.float32)  # 1*1 + 1*1 = 5
    grader.requireIsTrue(np_allclose(out, expected))




def hidden_test_forward():
    for _ in range(5):
        np.random.seed(SEED)
        x = np.random.randn(1, 28, 28)
        params = {
            "conv_w": np.random.randn(4, 1, 3, 3),
            "conv_b": np.random.randn(4),
            "fc_W": np.random.randn(10, 4 * 14 * 14),
            "fc_b": np.random.randn(10),
        }
        ans = solution.simple_cnn_forward(x, params)
        pred = submission.simple_cnn_forward(x, params)
        grader.requireIsTrue(np_allclose(ans, pred))


if solution_exist:
    grader.addBasicPart(
        "6-forward-hidden",
        hidden_test_forward,
        maxPoints=2,
        maxSeconds=10,
        description="Hidden forward pass test."
    )
else:
    grader.addBasicPart(
        "6-forward-basic",
        student_test_forward,
        maxPoints=1,
        maxSeconds=1,
        description="Example forward test (0 points)."
    )


############################################################
# Manual report (students only)
############################################################

# grader.addManualPart(
#     "7-report",
#     maxPoints=4,
#     description="Explain conv/pooling and describe your CNN."
# )

if __name__ == "__main__":
    grader.grade()
