from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class HABDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_files = sorted(self.image_dir.glob("*.npy"))
        self.mask_files = sorted(self.mask_dir.glob("*.npy"))

        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self.image_files)} images, "
                f"{len(self.mask_files)} masks"
            )

        image_names = [f.name for f in self.image_files]
        mask_names = [f.name for f in self.mask_files]

        if image_names != mask_names:
            raise ValueError("Image and mask filenames do not match.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = np.load(self.image_files[index])
        mask = np.load(self.mask_files[index])

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        mask = mask.astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(mask)


def create_dataset_split(base_dir, split):
    """
    Create one dataset split.

    split must be:
        train
        val
        test
    """

    base_dir = Path(base_dir)
    split_dir = base_dir / split

    return HABDataset(
        split_dir / "images",
        split_dir / "masks"
    )


def create_datasets(base_dir):
    """
    Create train, validation and test datasets.
    """

    base_dir = Path(base_dir)

    train_dataset = create_dataset_split(base_dir, "train")
    val_dataset = create_dataset_split(base_dir, "val")
    test_dataset = create_dataset_split(base_dir, "test")

    return train_dataset, val_dataset, test_dataset