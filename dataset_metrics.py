from __future__ import annotations

from pathlib import Path
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fashionmnist_classification_loaders(
    data_root: Path,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=str(data_root), train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


class FashionMNISTFeatureNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.embedding(x), inplace=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def _prepare_inputs(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.min().item() < -1e-4 or images.max().item() > 1.0 + 1e-4:
        images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = _prepare_inputs(images.to(device))
        labels = labels.to(device)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def load_or_train_feature_extractor(
    cache_path: Path,
    data_root: Path,
    device: torch.device,
    embedding_dim: int = 128,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 123,
) -> tuple[FashionMNISTFeatureNet, dict]:
    train_loader, test_loader = build_fashionmnist_classification_loaders(
        data_root=data_root,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    model = FashionMNISTFeatureNet(embedding_dim=embedding_dim).to(device)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location=device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model, payload.get("metadata", {})

    _set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = _prepare_inputs(images.to(device))
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.numel())
            running_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        test_acc = evaluate_classifier(model, test_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(
            f"feature extractor epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    metadata = {
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "history": history,
        "final_test_acc": history[-1]["test_acc"] if history else 0.0,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, cache_path)
    model.eval()
    return model, metadata


@torch.no_grad()
def extract_embeddings(
    model: FashionMNISTFeatureNet,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for images, _ in loader:
        images = _prepare_inputs(images.to(device))
        chunks.append(model.encode(images).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def extract_embeddings_from_tensor(
    model: FashionMNISTFeatureNet,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = _prepare_inputs(images[start : start + batch_size].to(device))
        outputs.append(model.encode(batch).cpu())
    return torch.cat(outputs, dim=0)


def compute_feature_stats(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    return mean, cov


def _sqrtm_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    matrix = ((matrix + matrix.T) * 0.5).double()
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.T


def dataset_fid_from_stats(
    real_mean: torch.Tensor,
    real_cov: torch.Tensor,
    gen_mean: torch.Tensor,
    gen_cov: torch.Tensor,
) -> float:
    diff = real_mean - gen_mean
    real_cov = ((real_cov + real_cov.T) * 0.5).double()
    gen_cov = ((gen_cov + gen_cov.T) * 0.5).double()
    real_cov_sqrt = _sqrtm_psd(real_cov)
    middle = real_cov_sqrt @ gen_cov @ real_cov_sqrt
    covmean = _sqrtm_psd(middle)
    fid = diff.dot(diff) + torch.trace(real_cov + gen_cov - 2.0 * covmean)
    return float(fid.clamp_min(0.0).item())


def dataset_fid(real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
    real_mean, real_cov = compute_feature_stats(real_features)
    gen_mean, gen_cov = compute_feature_stats(gen_features)
    return dataset_fid_from_stats(real_mean, real_cov, gen_mean, gen_cov)


def _polynomial_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> torch.Tensor:
    x = x.double()
    y = y.double()
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef0) ** degree


def _unbiased_polynomial_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> float:
    m = min(x.shape[0], y.shape[0])
    if m < 2:
        return 0.0
    k_xx = _polynomial_kernel(x, x, degree=degree, gamma=gamma, coef0=coef0)
    k_yy = _polynomial_kernel(y, y, degree=degree, gamma=gamma, coef0=coef0)
    k_xy = _polynomial_kernel(x, y, degree=degree, gamma=gamma, coef0=coef0)

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return float((sum_xx + sum_yy - 2.0 * sum_xy).item())


def dataset_kid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    subset_size: int = 1000,
    num_subsets: int = 50,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    seed: int = 123,
) -> dict:
    real_features = real_features.float().cpu()
    gen_features = gen_features.float().cpu()
    subset_size = min(subset_size, real_features.shape[0], gen_features.shape[0])
    if subset_size < 2:
        return {"mean": 0.0, "std": 0.0, "subset_size": subset_size, "num_subsets": 0}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    estimates = []
    for _ in range(num_subsets):
        real_idx = torch.randperm(real_features.shape[0], generator=generator)[:subset_size]
        gen_idx = torch.randperm(gen_features.shape[0], generator=generator)[:subset_size]
        estimates.append(
            _unbiased_polynomial_mmd2(
                real_features[real_idx],
                gen_features[gen_idx],
                degree=degree,
                gamma=gamma,
                coef0=coef0,
            )
        )

    values = torch.tensor(estimates, dtype=torch.float64)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "subset_size": subset_size,
        "num_subsets": int(num_subsets),
    }


def load_or_compute_reference_stats(
    cache_path: Path,
    model: FashionMNISTFeatureNet,
    data_root: Path,
    device: torch.device,
    eval_batch_size: int = 512,
    num_workers: int = 0,
) -> dict:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    _, test_loader = build_fashionmnist_classification_loaders(
        data_root=data_root,
        train_batch_size=eval_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    features = extract_embeddings(model, test_loader, device=device)
    mean, cov = compute_feature_stats(features)
    payload = {
        "features": features.cpu(),
        "mean": mean.cpu(),
        "cov": cov.cpu(),
        "num_samples": int(features.shape[0]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


@torch.no_grad()
def sample_in_batches(
    total_samples: int,
    batch_size: int,
    sampler,
    seed: int | None = None,
    show_progress: bool = True,
    progress_desc: str = "Sampling",
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    produced = 0
    batch_idx = 0
    total_batches = int(math.ceil(total_samples / batch_size))
    iterator = range(total_batches)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc=progress_desc, leave=False)

    for _ in iterator:
        current_batch = min(batch_size, total_samples - produced)
        batch_seed = None if seed is None else seed + batch_idx
        samples = sampler(current_batch, batch_seed).detach().cpu()
        batches.append(samples)
        produced += current_batch
        batch_idx += 1
    return torch.cat(batches, dim=0)
