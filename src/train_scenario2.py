import copy
import math
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
        self.gamma = getattr(cfg, "scenario2_gamma", getattr(cfg, "gamma", 1.0))
        self.patience = getattr(cfg, "patience", 10)  # fallback
        optim_cfg = getattr(cfg, "optim", None)
        self.lr_warmup_epochs = int(getattr(optim_cfg, "warmup_epochs", 0) or 0) if optim_cfg else 0
        start_factor = float(getattr(optim_cfg, "warmup_start_factor", 0.1) or 0.0) if optim_cfg else 0.0
        self.lr_warmup_start_factor = float(min(max(start_factor, 0.0), 1.0))
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self._lr_warmup_finished = self.lr_warmup_epochs <= 0
        self.ce = torch.nn.CrossEntropyLoss()

        # Trainer-specific controls (objective + loss toggles)
        self.trainer_cfg = getattr(cfg, "trainer", None)
        objective_cfg = getattr(self.trainer_cfg, "objective", None) if self.trainer_cfg else None
        metric_name = getattr(objective_cfg, "metric", None) if objective_cfg else None
        self.objective_metric = metric_name or "val_f1"
        mode = getattr(objective_cfg, "mode", "max") if objective_cfg else "max"
        if isinstance(mode, str):
            mode = mode.lower()
        self.objective_mode = "min" if mode == "min" else "max"

        losses_cfg = getattr(self.trainer_cfg, "losses", None) if self.trainer_cfg else None

        def _loss_flag(name: str, default: bool) -> bool:
            if not losses_cfg:
                return default
            value = getattr(losses_cfg, name, None)
            if value is None:
                return default
            return bool(value)

        activity_global = getattr(losses_cfg, "activity", None) if losses_cfg else None
        self.use_mse_loss = _loss_flag("mse", True)
        # Support either feature_similarity or shorthand sim
        feature_flag = getattr(losses_cfg, "feature_similarity", None) if losses_cfg else None
        if feature_flag is None and losses_cfg:
            feature_flag = getattr(losses_cfg, "sim", None)
        if feature_flag is None:
            self.use_feature_similarity_loss = _loss_flag("feature_similarity", True)
        else:
            self.use_feature_similarity_loss = bool(feature_flag)
        self.use_activity_loss_real = _loss_flag("activity_real", True)
        self.use_activity_loss_sim = _loss_flag("activity_sim", True)
        if activity_global is not None:
            flag = bool(activity_global)
            self.use_activity_loss_real = self.use_activity_loss_real and flag
            self.use_activity_loss_sim = self.use_activity_loss_sim and flag
        self.use_activity_loss = self.use_activity_loss_real or self.use_activity_loss_sim

        self.dual_classifiers = bool(getattr(self.trainer_cfg, "separate_classifiers", False))
        self.dual_feature_extractors = bool(getattr(self.trainer_cfg, "dual_feature_extractors", False))
        clip_val = getattr(self.trainer_cfg, "gradient_clip", None)
        try:
            clip_val = float(clip_val) if clip_val is not None else None
        except Exception:
            clip_val = None
        self.gradient_clip = clip_val if clip_val and clip_val > 0 else None
        self.real_classifier_key = "ac"
        self.sim_classifier_key = "ac_sim" if self.dual_classifiers else "ac"
        self.sim_fe_key = "fe_sim" if self.dual_feature_extractors else "fe"
        if self.dual_classifiers and self.sim_classifier_key not in self.models:
            raise ValueError(
                "dual_classifiers enabled but 'ac_sim' model is missing. Ensure experiments.run_trial builds it."
            )
        if self.dual_feature_extractors and self.sim_fe_key not in self.models:
            raise ValueError(
                "dual_feature_extractors enabled but 'fe_sim' model is missing. Ensure experiments.run_trial builds it."
            )

    def _cosine(self, a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).mean()

    def _forward_losses(self, batch):
        pose, acc, labels = batch
        pose, acc, labels = pose.to(self.device), acc.to(self.device), labels.to(self.device)
        sim_acc = self.models["pose2imu"](pose)
        mse = torch.nn.functional.mse_loss(sim_acc, acc)
        real_feat = self.models["fe"](acc)
        sim_feat = self.models[self.sim_fe_key](sim_acc)
        logits_real = self.models[self.real_classifier_key](real_feat)
        logits_sim = self.models[self.sim_classifier_key](sim_feat)
        zero = torch.zeros((), device=self.device, dtype=pose.dtype)

        if self.beta > 0 and self.use_feature_similarity_loss:
            sim_loss = self._cosine(sim_feat, real_feat)
        else:
            sim_loss = zero

        act_loss = zero
        if self.use_activity_loss and self.alpha > 0:
            if self.use_activity_loss_real:
                act_loss = act_loss + self.ce(logits_real, labels)
            if self.use_activity_loss_sim:
                act_loss = act_loss + self.ce(logits_sim, labels)

        total = torch.zeros((), device=self.device, dtype=pose.dtype)
        if self.use_mse_loss:
            total = total + self.gamma * mse
        if self.beta > 0 and self.use_feature_similarity_loss:
            total = total + self.beta * sim_loss
        if self.alpha > 0 and self.use_activity_loss:
            total = total + self.alpha * act_loss

        return (
            total,
            float(mse.detach().cpu()),
            float(sim_loss.detach().cpu()),
            float(act_loss.detach().cpu()),
            logits_real,
            labels,
        )

    def _apply_lr_warmup(self, epoch: int) -> None:
        if self._lr_warmup_finished:
            return
        if epoch >= self.lr_warmup_epochs:
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr
            self._lr_warmup_finished = True
            return
        progress = epoch / max(self.lr_warmup_epochs, 1)
        factor = self.lr_warmup_start_factor + (1.0 - self.lr_warmup_start_factor) * progress
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def _is_improvement(self, current: float, best_value: float) -> bool:
        if current is None:
            return False
        current_val = float(current)
        if not math.isfinite(current_val):
            return False
        if not math.isfinite(best_value):
            return True
        if self.objective_mode == "min":
            return current_val < best_value
        return current_val > best_value

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
                    if self.gradient_clip:
                        clip_params = []
                        for group in self.optimizer.param_groups:
                            clip_params.extend(group.get("params", []))
                        if clip_params:
                            torch.nn.utils.clip_grad_norm_(clip_params, self.gradient_clip)
                    self.optimizer.step()
                total += total_loss.detach().item()
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
        if self.objective_metric not in (
            "train_loss",
            "val_loss",
            "train_f1",
            "val_f1",
            "train_acc",
            "val_acc",
            "train_mse",
            "val_mse",
            "train_sim_loss",
            "val_sim_loss",
            "train_act_loss",
            "val_act_loss",
        ):
            raise KeyError(
                f"Objective metric '{self.objective_metric}' is not tracked by the Trainer. "
                "Choose one of: train_loss, val_loss, train_f1, val_f1, train_acc, val_acc, "
                "train_mse, val_mse, train_sim_loss, val_sim_loss, train_act_loss, val_act_loss."
            )

        best_default = np.inf if self.objective_mode == "min" else -np.inf
        best = {"value": best_default, "state": None, "epoch": None}
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
            self._apply_lr_warmup(epoch)
            tr_loss, tr_f1, tr_mse, tr_sim, tr_act, tr_acc = self._run_epoch("train")
            val_loss, val_f1, val_mse, val_sim, val_act, val_acc = self._run_epoch("val")
            current_lr = self.optimizer.param_groups[0]["lr"]
            
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
                  f"Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.3e}")

            if self.scheduler is not None and self._lr_warmup_finished:
                sched_cfg = getattr(self.cfg, "optim", None)
                sched_cfg = getattr(sched_cfg, "scheduler", None) if sched_cfg else None
                metric_name = getattr(sched_cfg, "metric", None) if sched_cfg else None
                if not metric_name:
                    raise ValueError("Scheduler metric is not set; provide optim.scheduler.metric in the config.")
                metric_history = history.get(metric_name)
                if not (isinstance(metric_history, list) and metric_history):
                    raise KeyError(f"Scheduler metric '{metric_name}' missing from history.")
                target_metric = metric_history[-1]
                self.scheduler.step(target_metric)

            metric_history = history.get(self.objective_metric)
            if metric_history is None or not metric_history:
                raise KeyError(
                    f"Objective metric '{self.objective_metric}' not populated in history; available keys: {list(history.keys())}"
                )
            current_metric = metric_history[-1]
            if self._is_improvement(current_metric, best["value"]):
                best["value"] = current_metric
                best["state"] = {k: copy.deepcopy(m.state_dict()) for k, m in self.models.items()}
                best["epoch"] = epoch
                print(f"    → New best {self.objective_metric}: {current_metric:.4f}")

            if best["epoch"] is not None and (epoch - best["epoch"]) >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs (patience: {self.patience})")
                break

        print("-" * 70)
        if math.isfinite(best["value"]):
            print(f"Training completed. Best {self.objective_metric}: {best['value']:.4f}")
        else:
            print(f"Training completed. Best {self.objective_metric}: N/A")

        # restore best
        if best["state"]:
            for k, m in self.models.items():
                m.load_state_dict(best["state"][k])
        self.best_state = best["state"]
        self.best_epoch = best["epoch"]
        self.best_score = best["value"]
        return history
# ...existing code...
