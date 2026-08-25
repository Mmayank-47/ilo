"""
dataset.py — Dataset classes and data-loader factories for the Voice Emotion pipeline.

Design goals:
- RAVDESS is the primary loader; new datasets register via ``DatasetRegistry``.
- Speaker-leakage prevention: train/val/test splits are by actor ID, never by clip.
- 5-fold stratified cross-validation that respects speaker boundaries.
- Augmentation is applied only during training (never val/test).
- ``get_dataloaders()`` is the single entry-point for the training loop.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Type

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from data.features import FeatureConfig, extract_features
from data.augmentation import AudioAugmentor

import hashlib
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label / emotion utilities
# ---------------------------------------------------------------------------

# Default RAVDESS label map (emotion code → string name)
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Canonical class index order (alphabetical, stable)
CLASSES: List[str] = sorted(DEFAULT_LABEL_MAP.values())
CLASS_TO_IDX: Dict[str, int] = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}


def label_map_to_idx(label_map: Dict[str, str]) -> Dict[str, int]:
    """Convert an emotion-code→name map to an emotion-code→index map."""
    return {code: CLASS_TO_IDX[name] for code, name in label_map.items()}


# ---------------------------------------------------------------------------
# RAVDESS parser
# ---------------------------------------------------------------------------

_RAVDESS_PATTERN = re.compile(
    r"(?P<modality>\d{2})-"
    r"(?P<vocal_channel>\d{2})-"
    r"(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-"
    r"(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)


def parse_ravdess_filename(filepath: str) -> Optional[Dict[str, Any]]:
    """Parse a RAVDESS filename into its component metadata.

    Returns ``None`` if the filename does not match the RAVDESS pattern.

    Args:
        filepath: Full path or filename, e.g.
            ``03-01-06-01-02-01-12.wav``

    Returns:
        Dict with keys: modality, vocal_channel, emotion_code, intensity,
        statement, repetition, actor_id (all as ints, emotion also as str code).
    """
    name = Path(filepath).name
    m = _RAVDESS_PATTERN.match(name)
    if m is None:
        return None
    return {
        "modality": int(m.group("modality")),
        "vocal_channel": int(m.group("vocal_channel")),
        "emotion_code": m.group("emotion"),
        "intensity": int(m.group("intensity")),
        "statement": int(m.group("statement")),
        "repetition": int(m.group("repetition")),
        "actor_id": int(m.group("actor")),
        "filepath": filepath,
    }


def scan_ravdess(root_dir: str, label_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Recursively scan a RAVDESS directory and collect valid speech-audio samples.

    Only modality=03 (audio-only) files are included. Song files (modality=01/02)
    are excluded. Files whose emotion code is not in ``label_map`` are skipped.

    Args:
        root_dir: Path to ``Audio_Speech_Actors_01-24/`` or its parent.
        label_map: Emotion code → label name mapping from config.

    Returns:
        List of sample dicts, each with ``filepath``, ``actor_id``, ``emotion_code``,
        ``label`` (str), ``label_idx`` (int).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"RAVDESS root directory not found: '{root_dir}'. "
            "Download from https://zenodo.org/record/1188976 and unzip."
        )

    code_to_idx = label_map_to_idx(label_map)
    samples: List[Dict[str, Any]] = []

    for wav_path in root.rglob("*.wav"):
        meta = parse_ravdess_filename(str(wav_path))
        if meta is None:
            continue
        # Keep only audio-only speech files
        if meta["modality"] != 3:
            continue
        code = meta["emotion_code"]
        if code not in label_map:
            logger.debug("Skipping unknown emotion code %s in %s", code, wav_path.name)
            continue
        samples.append({
            "filepath": str(wav_path),
            "actor_id": meta["actor_id"],
            "emotion_code": code,
            "label": label_map[code],
            "label_idx": code_to_idx[code],
        })

    if not samples:
        raise ValueError(f"No RAVDESS samples found under '{root_dir}'.")
    logger.info("Scanned %d RAVDESS samples from %s", len(samples), root_dir)
    return samples


# ---------------------------------------------------------------------------
# Dataset registry — add new corpora here
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Registry mapping dataset names to scanner functions.

    A scanner function signature:
        ``(root_dir: str, label_map: Dict[str, str]) -> List[Dict[str, Any]]``

    New datasets register themselves with:
        ``DatasetRegistry.register("my_dataset", my_scanner_fn)``
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, scanner_fn: Callable) -> None:
        cls._registry[name] = scanner_fn
        logger.info("Registered dataset scanner: %s", name)

    @classmethod
    def scan(cls, name: str, root_dir: str, label_map: Dict[str, str]) -> List[Dict[str, Any]]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Registered datasets: {list(cls._registry.keys())}"
            )
        return cls._registry[name](root_dir, label_map)


# Register RAVDESS as the default dataset
DatasetRegistry.register("ravdess", scan_ravdess)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification from audio files.

    Applies feature extraction on-the-fly, with optional augmentation.

    Args:
        samples: List of sample dicts (filepath, label_idx, …).
        feature_cfg: Feature extraction configuration.
        augmentor: ``AudioAugmentor`` instance (pass ``None`` for val/test).
        cache_features: If ``True``, cache extracted features in memory after
            the first access (trades RAM for CPU during epoch).
    """

    def __init__(
        self,
            samples: List[Dict[str, Any]],
            feature_cfg: FeatureConfig,
            augmentor: Optional[AudioAugmentor] = None,
            cache_features: bool = False,
        ) -> None:
            self.samples = samples
            self.feature_cfg = feature_cfg
            self.augmentor = augmentor
            self.cache_features = cache_features
        
            # In-memory cache for the current run
            self._cache: Dict[int, np.ndarray] = {}
        
            # Persistent disk cache
            self.cache_dir = Path("data/cache")
            if self.cache_features:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label_idx = sample["label_idx"]
        
            # ----------------------------------------------------------
            # Load/extract the base features
            # ----------------------------------------------------------
        if self.cache_features:
                # Create a unique, stable cache filename from the audio filepath
                cache_key = hashlib.md5(
                    sample["filepath"].encode("utf-8")
                ).hexdigest()
        
                cache_path = self.cache_dir / f"{cache_key}.npy"
        
                # Load from disk if already cached
                if cache_path.exists():
                    features = np.load(cache_path)
        
                # Otherwise extract once and save for future use
                else:
                    features = self._extract(sample["filepath"])
                    np.save(cache_path, features)
        
        else:
                # Caching disabled → extract features normally
                features = self._extract(sample["filepath"])
        
            # ----------------------------------------------------------
            # Apply augmentation (training only)
            # IMPORTANT: augmentation happens AFTER loading the original
            # cached features, so every training access can still be random.
            # ----------------------------------------------------------
        if self.augmentor is not None:
                features = self.augmentor.augment_features(features)
        
        tensor = torch.from_numpy(features)   # (max_seq_len, feature_dim)
        return tensor, label_idx


    def _extract(self, filepath: str) -> np.ndarray:
        """Extract features with waveform-level augmentation if enabled."""
        import librosa as _librosa  # local import to keep top-level clean
        if self.augmentor is not None:
            # Load raw waveform, augment it, then extract features
            y, _ = _librosa.load(filepath, sr=self.feature_cfg.sample_rate, mono=True)
            y = self.augmentor.augment_waveform(y, self.feature_cfg.sample_rate)
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            sf.write(tmp_path, y, self.feature_cfg.sample_rate)
            try:
                features = extract_features(tmp_path, self.feature_cfg)
            finally:
                os.unlink(tmp_path)
        else:
            features = extract_features(filepath, self.feature_cfg)
        return features

    @property
    def labels(self) -> List[int]:
        """All label indices, in dataset order. Used for stratified sampling."""
        return [s["label_idx"] for s in self.samples]


# ---------------------------------------------------------------------------
# Splitting utilities (speaker-aware)
# ---------------------------------------------------------------------------

def split_by_actor(
    samples: List[Dict[str, Any]],
    test_actor_ids: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate samples into (trainval, test) by actor ID.

    Args:
        samples: All dataset samples.
        test_actor_ids: Actor IDs reserved for the held-out test set.

    Returns:
        (trainval_samples, test_samples)
    """
    test_ids = set(test_actor_ids)
    trainval = [s for s in samples if s["actor_id"] not in test_ids]
    test = [s for s in samples if s["actor_id"] in test_ids]
    logger.info(
        "Split: trainval=%d  test=%d  (test actors: %s)",
        len(trainval), len(test), sorted(test_actor_ids),
    )
    return trainval, test


def get_speaker_aware_kfolds(
    trainval_samples: List[Dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Stratified k-fold CV where entire actors are kept together.

    Groups samples by actor, then splits actors into folds.  Within each fold
    the actor-level split preserves the class distribution as well as possible
    (stratified at the actor group level).

    Args:
        trainval_samples: Samples that are eligible for training/validation.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        List of (train_indices, val_indices) tuples (indices into trainval_samples).
    """
    # Collect unique actors and their dominant label (for stratification proxy)
    actor_ids = sorted({s["actor_id"] for s in trainval_samples})
    # Build per-actor label distribution; use majority label as stratum proxy
    actor_to_majority_label: Dict[int, int] = {}
    for actor in actor_ids:
        actor_samples = [s for s in trainval_samples if s["actor_id"] == actor]
        labels = [s["label_idx"] for s in actor_samples]
        actor_to_majority_label[actor] = max(set(labels), key=labels.count)

    actor_array = np.array(actor_ids)
    actor_labels = np.array([actor_to_majority_label[a] for a in actor_ids])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Tuple[List[int], List[int]]] = []

    for train_actor_idx, val_actor_idx in skf.split(actor_array, actor_labels):
        train_actors = set(actor_array[train_actor_idx])
        val_actors = set(actor_array[val_actor_idx])
        train_indices = [
            i for i, s in enumerate(trainval_samples) if s["actor_id"] in train_actors
        ]
        val_indices = [
            i for i, s in enumerate(trainval_samples) if s["actor_id"] in val_actors
        ]
        folds.append((train_indices, val_indices))

    return folds


def compute_class_weights(samples: List[Dict[str, Any]], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for the loss function.

    Args:
        samples: Training samples (not the full dataset).
        num_classes: Total number of classes.

    Returns:
        Tensor of shape ``(num_classes,)`` with dtype float32.
    """
    counts = np.bincount([s["label_idx"] for s in samples], minlength=num_classes)
    counts = np.maximum(counts, 1)   # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalise so mean ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(samples: List[Dict[str, Any]], num_classes: int) -> WeightedRandomSampler:
    """Create a ``WeightedRandomSampler`` for balanced batch sampling."""
    class_weights = compute_class_weights(samples, num_classes)
    sample_weights = torch.tensor(
        [float(class_weights[s["label_idx"]]) for s in samples], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(samples), replacement=True
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def get_dataloaders(
    cfg: Dict[str, Any],
    train_indices: Optional[List[int]] = None,
    val_indices: Optional[List[int]] = None,
    trainval_samples: Optional[List[Dict[str, Any]]] = None,
    test_samples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from config.

    Call once per CV fold, passing the fold's train/val index lists.
    On the first call (fold=0), this function also scans and splits the dataset.

    Args:
        cfg: Full config dict (from ``load_config()``).
        train_indices: Sample indices for training (into trainval_samples).
        val_indices: Sample indices for validation.
        trainval_samples: Pre-scanned train+val samples (to avoid re-scanning).
        test_samples: Pre-scanned test samples.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    ds_cfg = cfg["dataset"]
    feat_cfg = FeatureConfig.from_dict(cfg["features"])
    train_cfg = cfg["training"]

    # --- Scan datasets (if not already done) ---
    if trainval_samples is None or test_samples is None:
        samples: List[Dict[str, Any]] = []
        # Primary dataset
        primary = DatasetRegistry.scan(
            ds_cfg.get("name", "ravdess"),
            ds_cfg["root_dir"],
            ds_cfg.get("label_map", DEFAULT_LABEL_MAP),
        )
        samples.extend(primary)
        # Extra datasets
        for extra in ds_cfg.get("extra_datasets", []):
            extra_samples = DatasetRegistry.scan(
                extra["name"],
                extra["root_dir"],
                ds_cfg.get("label_map", DEFAULT_LABEL_MAP),
            )
            samples.extend(extra_samples)

        trainval_samples, test_samples = split_by_actor(
            samples, ds_cfg.get("test_actor_ids", [23, 24])
        )

    # --- Resolve fold indices ---
    if train_indices is None:
        train_indices = list(range(len(trainval_samples)))
    if val_indices is None:
        # Simple 80/20 fallback (not speaker-stratified) if no fold given
        n = len(train_indices)
        split = int(0.8 * n)
        val_indices = train_indices[split:]
        train_indices = train_indices[:split]

    train_samples = [trainval_samples[i] for i in train_indices]
    val_samples = [trainval_samples[i] for i in val_indices]

    aug_cfg = cfg.get("augmentation", {})
    augmentor = AudioAugmentor(aug_cfg) if aug_cfg.get("enabled", True) else None

    # Training samples use random augmentation, so we don't cache the
# waveform-augmented features.
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = train_cfg.get("pin_memory", False)
    batch_size = train_cfg.get("batch_size", 64)
    cache_features = train_cfg.get("cache_features", False)

    train_ds = EmotionDataset(
    train_samples,
    feat_cfg,
    augmentor=augmentor,
    cache_features=False,
)

    # Validation and test data are deterministic, so caching saves a lot of time
# after the first epoch.
    val_ds = EmotionDataset(
    val_samples,
    feat_cfg,
    augmentor=None,
    cache_features=cache_features,
)

    test_ds = EmotionDataset(
    test_samples,
    feat_cfg,
    augmentor=None,
    cache_features=cache_features,
)



    train_sampler = make_weighted_sampler(train_samples, ds_cfg.get("num_classes", 8)
                                          if "num_classes" in ds_cfg else cfg["model"]["num_classes"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "DataLoaders ready — train: %d samples, val: %d samples, test: %d samples",
        len(train_samples), len(val_samples), len(test_samples),
    )
    return train_loader, val_loader, test_loader
