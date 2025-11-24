import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import submission as S
import scipy.ndimage


def mnist_style_preprocess(path):
    """Convert hand-written digit to true MNIST-style 28×28."""

    # 1) Load grayscale
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)

    # 2) Invert if background is brighter
    if arr.mean() > 128:
        arr = 255 - arr

    # 3) Binarize for bounding box extraction
    arr_bin = (arr > 40).astype(np.uint8)

    # 4) Bounding box
    rows = np.where(arr_bin.max(axis=1) > 0)[0]
    cols = np.where(arr_bin.max(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No digit found!")

    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    digit = arr[rmin:rmax+1, cmin:cmax+1]

    # 5) Resize longest side to 20 px (MNIST rule)
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20.0 / h))
    else:
        new_w = 20
        new_h = int(h * (20.0 / w))

    digit_resized = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_resized)

    # 6) Center in 28×28
    canvas = np.zeros((28, 28), dtype=np.float32)
    r_start = (28 - new_h) // 2
    c_start = (28 - new_w) // 2
    canvas[r_start:r_start + new_h, c_start:c_start + new_w] = digit_resized

    # 7) Center-of-mass shift (MNIST official)
    cy, cx = scipy.ndimage.center_of_mass(canvas)
    shift_y = int(np.round(14 - cy))
    shift_x = int(np.round(14 - cx))

    # order=1 avoids excessive blur from default cubic interpolation
    canvas = scipy.ndimage.shift(canvas, shift=(shift_y, shift_x), order=1)

    # 8) Normalize to 0~1
    canvas = np.clip(canvas, 0, 255) / 255.0

    return canvas.reshape(1, 28, 28)


def predict_image(path, pretrained_folder="pretrained"):
    params = {
        "conv_w": np.load(f"{pretrained_folder}/conv_w.npy"),
        "conv_b": np.load(f"{pretrained_folder}/conv_b.npy"),
        "fc_W":   np.load(f"{pretrained_folder}/fc_W.npy"),
        "fc_b":   np.load(f"{pretrained_folder}/fc_b.npy"),
    }

    x = mnist_style_preprocess(path)
    logits = S.simple_cnn_forward(x, params)
    pred = int(np.argmax(logits))

    plt.imshow(x[0], cmap="gray")
    plt.title(f"PREDICTION: {pred}")
    plt.axis("off")
    plt.show()

    return pred


if __name__ == "__main__":
    img_path = input("Image path: ").strip()
    print("Prediction:", predict_image(img_path))
