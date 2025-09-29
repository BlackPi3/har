"""Lightning module for unified HAR training.

Combines three model components:
- pose2imu (regressor): pose -> simulated accelerometer
- fe (feature extractor): accelerometer sequence -> feature embedding
- ac (activity classifier): embedding -> activity logits

Loss Components (combined):
  total = mse(sim_acc, real_acc) + alpha * (CE(real_logits, y) + CE(sim_logits, y)) + beta * feature_similarity_loss
Where feature_similarity_loss = mean(1 - cosine_sim(fe(sim_acc), fe(real_acc))) when beta > 0 else 0.

Logged Metrics (per epoch):
- train_loss, train_mse, train_act, (train_sim if beta>0)
- val_loss, val_f1 (macro)

Scheduler: ReduceLROnPlateau on val_loss (factor=0.1) with patience from namespace.
"""
from __future__ import annotations
from typing import Dict, Any
import torch
import pytorch_lightning as pl


class HARLightningModule(pl.LightningModule):
    def __init__(self, cfg: Any, ns: Any, models: Dict[str, torch.nn.Module]):  # ns is a SimpleNamespace; cfg is DictConfig
        super().__init__()
        # Persist only scalar hyperparameters; ignore model module weights to avoid bloating checkpoints
        self.save_hyperparameters({
            'alpha': float(getattr(cfg.experiment, 'alpha', 1.0)),
            'beta': float(getattr(cfg.experiment, 'beta', 0.0)),
            'lr': float(getattr(ns, 'lr', 1e-3)),
            'weight_decay': float(getattr(ns, 'weight_decay', 0.0)),
            'patience': int(getattr(ns, 'patience', 10)),
        })
        self.pose2imu = models['pose2imu']
        self.fe = models['fe']
        self.ac = models['ac']
        self.alpha = self.hparams['alpha']
        self.beta = self.hparams['beta']
        self.lr = self.hparams['lr']
        self.weight_decay = self.hparams['weight_decay']
        self.patience = self.hparams['patience']
        self._val_preds = []  # list of tensors
        self._val_targets = []  # list of tensors
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pose: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.pose2imu(pose)

    def _shared_step(self, batch):
        pose, acc, labels = batch
        pose, acc, labels = pose.to(self.device), acc.to(self.device), labels.to(self.device)
        sim_acc = self.pose2imu(pose)
        mse_loss = torch.nn.functional.mse_loss(sim_acc, acc)
        real_feat = self.fe(acc)
        sim_feat = self.fe(sim_acc)
        if self.beta > 0:
            sim_loss = (1 - torch.nn.functional.cosine_similarity(sim_feat, real_feat, dim=1)).mean()
        else:
            sim_loss = torch.tensor(0.0, device=self.device)
        logits_real = self.ac(real_feat)
        logits_sim = self.ac(sim_feat)
        act_loss = self.ce(logits_real, labels) + self.ce(logits_sim, labels)
        total = mse_loss + self.alpha * act_loss + self.beta * sim_loss
        return total, mse_loss.detach(), sim_loss.detach(), act_loss.detach(), logits_real, labels

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        total, mse_l, sim_l, act_l, logits, labels = self._shared_step(batch)
        self.log("train_loss", total, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_mse", mse_l, prog_bar=False, on_epoch=True)
        self.log("train_act", act_l, prog_bar=False, on_epoch=True)
        if self.beta > 0:
            self.log("train_sim", sim_l, prog_bar=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        total, mse_l, sim_l, act_l, logits, labels = self._shared_step(batch)
        preds = torch.argmax(logits, dim=1)
        self._val_preds.append(preds.cpu())
        self._val_targets.append(labels.cpu())
        self.log("val_loss", total, prog_bar=True, on_epoch=True, on_step=False)
        return total

    def on_validation_epoch_end(self):  # type: ignore[override]
        if self._val_preds:
            preds = torch.cat(self._val_preds)
            targets = torch.cat(self._val_targets)
            num_classes = int(targets.max().item() + 1)
            conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
            for p, t in zip(preds, targets):
                conf[t, p] += 1
            tp = conf.diag()
            fp = conf.sum(0) - tp
            fn = conf.sum(1) - tp
            precision = tp.float() / (tp + fp + 1e-8)
            recall = tp.float() / (tp + fn + 1e-8)
            f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
            macro_f1 = f1_per_class.mean().item()
            self.log("val_f1", macro_f1, prog_bar=True, on_epoch=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):  # type: ignore[override]
        params = list(self.pose2imu.parameters()) + list(self.fe.parameters()) + list(self.ac.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=self.patience
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
