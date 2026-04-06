from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


@dataclass
class NumpyDataset:
    images: np.ndarray
    labels: np.ndarray


def _resize_and_normalize(images: np.ndarray, image_size: int) -> np.ndarray:
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError(f"Expected MNIST images with 3 or 4 dimensions, got shape {images.shape}")
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0
    if images.shape[1] != image_size or images.shape[2] != image_size:
        pad = (image_size - images.shape[1]) // 2
        images = np.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    return images


def load_mnist_numpy(image_size: int = 32) -> tuple[NumpyDataset, NumpyDataset]:
    train_ds, test_ds = tfds.as_numpy(
        tfds.load("mnist", split=["train", "test"], batch_size=-1, as_supervised=True)
    )
    train_images, train_labels = train_ds
    test_images, test_labels = test_ds

    train_images = _resize_and_normalize(train_images, image_size)
    test_images = _resize_and_normalize(test_images, image_size)

    return (
        NumpyDataset(train_images, train_labels.astype(np.int32)),
        NumpyDataset(test_images, test_labels.astype(np.int32)),
    )


def filter_by_class_ids(dataset: NumpyDataset, class_ids: tuple[int, ...]) -> NumpyDataset:
    mask = np.isin(dataset.labels, np.asarray(class_ids, dtype=np.int32))
    return NumpyDataset(dataset.images[mask], dataset.labels[mask])


def batch_iterator(
    dataset: NumpyDataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset.labels))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield {
            "images": jnp.asarray(dataset.images[batch_idx]),
            "labels": jnp.asarray(dataset.labels[batch_idx]),
        }
