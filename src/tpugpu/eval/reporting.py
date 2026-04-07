from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _denorm(images: np.ndarray) -> np.ndarray:
    images = np.clip((images + 1.0) * 0.5, 0.0, 1.0)
    return images


def save_image_grid(images: np.ndarray, labels: np.ndarray, path: str | Path, title: str) -> None:
    images = _denorm(images)
    n = min(len(images), 64)
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8), constrained_layout=True)
    for ax, img, label in zip(axes.flat, images[:n], labels[:n]):
        ax.imshow(img.squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(str(int(label)), fontsize=8, pad=2)
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(Path(path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def save_label_histogram(labels: np.ndarray, path: str | Path, title: str) -> None:
    counts = np.bincount(labels.astype(np.int32), minlength=10)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(np.arange(10), counts, color="#0f766e", width=0.75)
    ax.set_xticks(np.arange(10))
    ax.set_xlabel("Digit label")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.savefig(Path(path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    eigvals, eigvecs = np.linalg.eigh(sigma1 @ sigma2)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_pca_fid(real_images: np.ndarray, generated_images: np.ndarray, n_components: int = 32) -> float:
    real_flat = real_images.reshape(real_images.shape[0], -1)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1)
    pca = PCA(n_components=min(n_components, real_flat.shape[0], gen_flat.shape[0], real_flat.shape[1]))
    pca.fit(real_flat)
    real_features = pca.transform(real_flat)
    gen_features = pca.transform(gen_flat)
    mu_real = real_features.mean(axis=0)
    mu_gen = gen_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    return _frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def save_tsne_plot(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    path: str | Path,
    title: str,
    seed: int,
) -> None:
    real_flat = real_images.reshape(real_images.shape[0], -1)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1)
    features = np.concatenate([real_flat, gen_flat], axis=0)
    pca = PCA(n_components=min(32, features.shape[0], features.shape[1]))
    reduced = pca.fit_transform(features)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed, perplexity=30)
    emb = tsne.fit_transform(reduced)

    n_real = real_flat.shape[0]
    emb_real = emb[:n_real]
    emb_gen = emb[n_real:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    scatter = ax.scatter(emb_real[:, 0], emb_real[:, 1], c=real_labels, cmap="tab10", s=8, alpha=0.7)
    ax.set_title("Real samples")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    scatter = ax.scatter(emb_gen[:, 0], emb_gen[:, 1], c=generated_labels, cmap="tab10", s=8, alpha=0.7)
    ax.set_title("Generated samples")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(Path(path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry.get("train_loss", np.nan) for entry in history]
    pca_fids = [entry.get("pca_fid", np.nan) for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(epochs, train_losses, marker="o", color="#0f766e", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE noise prediction loss")

    axes[1].plot(epochs, pca_fids, marker="o", color="#b91c1c", linewidth=2)
    axes[1].set_title("PCA-FID proxy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fréchet distance")

    fig.savefig(Path(path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)
