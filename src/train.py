import copy
import numpy as np
import torch
import torch.nn.functional as F

try:  # Keep optional import; fallback if sklearn not installed
    from sklearn.metrics import f1_score  # type: ignore
except Exception:  # pragma: no cover - fallback path
    def f1_score(trues, preds, average="macro"):
        import numpy as _np
        trues = _np.asarray(trues)
        preds = _np.asarray(preds)
        classes = _np.unique(_np.concatenate([trues, preds]))
        f1s = []
        for c in classes:
            tp = ((preds == c) & (trues == c)).sum()
            fp = ((preds == c) & (trues != c)).sum()
            fn = ((preds != c) & (trues == c)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1s.append(2 * precision * recall / (precision + recall + 1e-8))
        return float(_np.mean(f1s))

# ...existing code...
class Trainer:
    def __init__(self, models: dict, dataloaders: dict, optimizer, scheduler, cfg, device):
        self.models = models
        self.dl = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        # Use scenario-specific naming; fall back to generic if present
        self.alpha = getattr(cfg, "scenario2_alpha", getattr(cfg, "alpha", 1.0))
        self.beta = getattr(cfg, "scenario2_beta", getattr(cfg, "beta", 0.0))
        self.patience = getattr(cfg, "patience", 10)  # fallback

    def _cosine(self, a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).mean()

    def _forward_losses(self, batch):
        pose, acc, labels = batch
        pose, acc, labels = pose.to(self.device), acc.to(self.device), labels.to(self.device)
        sim_acc = self.models["pose2imu"](pose)
        mse = torch.nn.functional.mse_loss(sim_acc, acc)
        real_feat = self.models["fe"](acc)
        sim_feat = self.models["fe"](sim_acc)
        sim_loss = self._cosine(sim_feat, real_feat) if self.beta > 0 else 0.0
        logits_real = self.models["ac"](real_feat)
        logits_sim = self.models["ac"](sim_feat)
        ce = torch.nn.CrossEntropyLoss()
        act_loss = ce(logits_real, labels) + ce(logits_sim, labels)
        total = mse + self.alpha * act_loss + self.beta * sim_loss
        return total, mse.item(), sim_loss if isinstance(sim_loss, float) else sim_loss.item(), act_loss.item(), logits_real, labels

    def _run_epoch(self, split="train"):
        is_train = split == "train"
        for m in self.models.values():
            m.train() if is_train else m.eval()

        total, mse_acc, sim_acc, act_acc = 0.0, 0.0, 0.0, 0.0
        preds, trues = [], []
        with torch.set_grad_enabled(is_train):
            for batch in self.dl[split]:
                total_loss, mse_l, sim_l, act_l, logits, labels = self._forward_losses(batch)
                if is_train:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                total += total_loss.item()
                mse_acc += mse_l
                sim_acc += sim_l
                act_acc += act_l
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(labels.cpu().numpy())

        denom = max(len(self.dl[split]), 1)
        avg_loss = total / denom
        mse_avg = mse_acc / denom
        sim_avg = sim_acc / denom
        act_avg = act_acc / denom

        if trues:
            trues_arr = np.asarray(trues)
            preds_arr = np.asarray(preds)
            acc = float((preds_arr == trues_arr).mean())
            f1 = float(f1_score(trues_arr, preds_arr, average="macro"))
        else:
            acc = 0.0
            f1 = 0.0

        return avg_loss, f1, mse_avg, sim_avg, act_avg, acc

    def fit(self, epochs):
        best = {"f1": -np.inf, "state": None, "epoch": None}
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_acc": [],
            "val_acc": [],
            "train_mse": [],
            "val_mse": [],
            "train_sim_loss": [],
            "val_sim_loss": [],
            "train_act_loss": [],
            "val_act_loss": [],
        }
        
        print(f"Starting training for {epochs} epochs...")
        print("-" * 70)
        
        for epoch in range(epochs):
            tr_loss, tr_f1, tr_mse, tr_sim, tr_act, tr_acc = self._run_epoch("train")
            val_loss, val_f1, val_mse, val_sim, val_act, val_acc = self._run_epoch("val")
            
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(tr_f1)
            history["val_f1"].append(val_f1)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)
            history["train_mse"].append(tr_mse)
            history["val_mse"].append(val_mse)
            history["train_sim_loss"].append(tr_sim)
            history["val_sim_loss"].append(val_sim)
            history["train_act_loss"].append(tr_act)
            history["val_act_loss"].append(val_act)
            
            # Print epoch progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f}")
            
            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()
            
            if val_f1 > best["f1"]:
                best["f1"] = val_f1
                best["state"] = {k: copy.deepcopy(m.state_dict()) for k, m in self.models.items()}
                best["epoch"] = epoch
                print(f"    → New best Val F1: {val_f1:.4f}")
            
            if (epoch - np.argmax(history["val_f1"])) >= self.cfg.patience:
                print(f"Early stopping triggered after {epoch+1} epochs (patience: {self.cfg.patience})")
                break
        
        print("-" * 70)
        print(f"Training completed. Best Val F1: {best['f1']:.4f}")
        
        # restore best
        if best["state"]:
            for k, m in self.models.items():
                m.load_state_dict(best["state"][k])
        self.best_state = best["state"]
        self.best_epoch = best["epoch"]
        self.best_score = best["f1"]
        return history
# ...existing code...
