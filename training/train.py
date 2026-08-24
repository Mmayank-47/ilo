import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from app.models.facial_model import FacialMentalHealthModel
from training.dataset import VideoSessionDataset
from training.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Dataset Preparation
    dataset = VideoSessionDataset(
        num_synthetic_samples=100 if args.dry_run else 300,
        seq_length=args.seq_length
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model Initialization (Transfer learning from VGGFace2/AffectNet backbone)
    model = FacialMentalHealthModel(
        backbone_type=args.backbone,
        embed_dim=512,
        hidden_dim=256,
        aggregator_type="tcn"
    ).to(device)

    # 3. Optimised Loss + Scheduler
    #    GaussianNLLLoss pairs with Softplus std from new TaskHead
    #    → drives model to output well-calibrated uncertainty alongside severity scores
    epochs = 1 if args.dry_run else args.epochs
    criterion = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    logger.info(f"Starting training for {epochs} epoch(s)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)
            optimizer.zero_grad()

            outputs = model(frames)
            loss = 0.0
            for task in ["depression", "anxiety", "stress"]:
                task_labels = labels[task].to(device).unsqueeze(1)
                pred_mean, pred_std = outputs[task]
                # GaussianNLLLoss(mean, target, var=std^2)
                task_loss = criterion(pred_mean, task_labels, pred_std ** 2)
                loss += task_loss

            loss.backward()
            # Gradient clipping prevents exploding gradients in temporal attention layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # 4. Validation & Evaluation Metrics
    logger.info("Executing model validation and metrics calculation...")
    model.eval()
    evaluator = ModelEvaluator()

    val_targets = {"depression": [], "anxiety": [], "stress": []}
    val_preds = {"depression": [], "anxiety": [], "stress": []}

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            outputs = model(frames)
            for task in ["depression", "anxiety", "stress"]:
                pred_val, _ = outputs[task]
                val_preds[task].extend(pred_val.squeeze().cpu().numpy().tolist())
                val_targets[task].extend(labels[task].numpy().tolist())

    logger.info("=== Validation Metrics Report ===")
    logger.info("Note: Single-modality (face-only) confidence scored lower than future fused model.")
    
    for task in ["depression", "anxiety", "stress"]:
        y_t = np.array(val_targets[task])
        y_p = np.array(val_preds[task])
        
        metrics = evaluator.compute_task_metrics(y_t, y_p)
        logger.info(
            f"Task: {task.capitalize():<10} | MAE: {metrics['mae']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc_roc']:.4f} | ECE: {metrics['ece']:.4f}"
        )

        if not args.dry_run:
            plot_path = f"calibration_{task}.png"
            evaluator.generate_calibration_curve_plot((y_t >= 0.5).astype(int), y_p, task, plot_path)
            logger.info(f"Saved calibration plot to {plot_path}")

    # 5. Checkpoint Saving
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    logger.info(f"Model checkpoint saved successfully at '{args.save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial Mental Health Risk Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=15, help="Frame sequence length")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone architecture")
    parser.add_argument("--save-path", type=str, default="checkpoints/facial_model.pt", help="Path to save checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run for verification")

    cli_args = parser.parse_args()
    train_model(cli_args)
