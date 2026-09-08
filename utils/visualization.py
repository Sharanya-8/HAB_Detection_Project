# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_image(
    image,
    output_path,
    title="Image"
):
    """
    Save a single-channel image as a PNG.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def save_mask(
    mask,
    output_path,
    title="HAB Mask"
):
    """
    Save a HAB segmentation mask.

    0   = Non-HAB
    1   = HAB
    255 = Invalid / No-data
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    display_mask = np.ma.masked_where(
        mask == 255,
        mask
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(display_mask)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def save_prediction(
    prediction,
    output_path,
    title="HAB Prediction"
):
    """
    Save a model prediction mask.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(prediction)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def save_comparison(
    image,
    ground_truth,
    prediction,
    output_path
):
    """
    Save a comparison of input image,
    ground-truth HAB mask and model prediction.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Input")

    axes[1].imshow(
        np.ma.masked_where(
            ground_truth == 255,
            ground_truth
        )
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()