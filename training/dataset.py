import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional


class VideoSessionDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning FacialMentalHealthModel on student video session data.
    Loads sequences of aligned face frames alongside continuous severity labels [0.0, 1.0]
    for Depression, Anxiety, and Stress.
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        seq_length: int = 30,
        num_synthetic_samples: int = 50,
        transform=None
    ):
        self.seq_length = seq_length
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            # Generate synthetic samples for training/dry-run pipeline verification
            self.samples = self._generate_synthetic_samples(num_synthetic_samples, seq_length)

    def _generate_synthetic_samples(self, count: int, seq_len: int) -> List[Dict]:
        synthetic = []
        for i in range(count):
            # Generate continuous target severity labels
            dep = float(np.random.beta(2, 5))
            anx = float(np.random.beta(2, 5))
            stress = float(np.random.beta(2, 5))

            synthetic.append({
                "session_id": f"synthetic_session_{i:03d}",
                "frames": np.random.randint(0, 256, size=(seq_len, 224, 224, 3), dtype=np.uint8),
                "labels": {
                    "depression": dep,
                    "anxiety": anx,
                    "stress": stress
                }
            })
        return synthetic

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        frames_np = sample["frames"] # (T, H, W, C)

        # Normalize and transpose (T, H, W, C) -> (T, C, H, W)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

        norm_frames = (frames_np.astype(np.float32) / 255.0 - mean) / std
        chw_frames = np.transpose(norm_frames, (0, 3, 1, 2)) # (T, C, H, W)

        frames_tensor = torch.tensor(chw_frames, dtype=torch.float32)

        labels = {
            "depression": torch.tensor(sample["labels"]["depression"], dtype=torch.float32),
            "anxiety": torch.tensor(sample["labels"]["anxiety"], dtype=torch.float32),
            "stress": torch.tensor(sample["labels"]["stress"], dtype=torch.float32)
        }

        return frames_tensor, labels
