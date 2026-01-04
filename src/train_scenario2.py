import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

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

        # Trainer-specific controls (objective + loss toggles)
        self.trainer_cfg = getattr(cfg, "trainer", None)

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
        smoothing = 0.0
        try:
            smoothing = float(getattr(self.trainer_cfg, "label_smoothing", 0.0))
        except Exception:
            smoothing = 0.0
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=smoothing if smoothing > 0 else 0.0)

        # Secondary pose dataset support (optional, pose-only)
        sec_cfg = getattr(self.trainer_cfg, "secondary", None) if self.trainer_cfg else None
        self.use_secondary_pose = bool(getattr(sec_cfg, "enabled", False))
        self.secondary_loss_weight = float(getattr(sec_cfg, "loss_weight", 1.0)) if sec_cfg else 1.0
        self.secondary_classifier_key = getattr(sec_cfg, "classifier_key", "ac_secondary") if sec_cfg else None
        self.secondary_loader = self.dl.get("secondary_train") if self.use_secondary_pose else None
        if self.use_secondary_pose:
            if not self.secondary_classifier_key or self.secondary_classifier_key not in self.models:
                raise ValueError(
                    f"secondary classifier '{self.secondary_classifier_key}' missing in models while secondary is enabled."
                )
            if self.secondary_loader is None:
                raise ValueError("secondary is enabled but no secondary_train loader was provided.")

        objective_cfg = getattr(self.trainer_cfg, "objective", None) if self.trainer_cfg else None
        metric_name = getattr(objective_cfg, "metric", None) if objective_cfg else None
        self.objective_metric = metric_name or "val_f1"
        mode = getattr(objective_cfg, "mode", "max") if objective_cfg else "max"
        if isinstance(mode, str):
            mode = mode.lower()
        self.objective_mode = "min" if mode == "min" else "max"
        self.skip_val = (
            not self.dl.get("val")
            or (self.dl.get("val") is None)
            or bool(getattr(self.trainer_cfg, "disable_val", False))
        )
        if self.skip_val and isinstance(self.objective_metric, str) and self.objective_metric.startswith("val_"):
            self.objective_metric = f"train_{self.objective_metric[4:]}"

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
        self.dual_feature_extractors = bool(getattr(self.trainer_cfg, "separate_feature_extractors", False))
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
        sched_cfg = getattr(self.trainer_cfg, "objective", None)
        optim_cfg = getattr(cfg, "optim", None)
        sched_cfg = getattr(optim_cfg, "scheduler", None) if optim_cfg else None
        self.scheduler_metric = getattr(sched_cfg, "metric", None) if sched_cfg else None
        if self.skip_val and isinstance(self.scheduler_metric, str) and self.scheduler_metric.startswith("val_"):
            self.scheduler_metric = f"train_{self.scheduler_metric[4:]}"

        # Adversarial training support (Scenario 4)
        adv_cfg = getattr(self.trainer_cfg, "adversarial", None) if self.trainer_cfg else None
        self.use_adversarial = bool(getattr(adv_cfg, "enabled", False)) if adv_cfg else False
        self.adv_weight = float(getattr(adv_cfg, "weight", 0.1)) if adv_cfg else 0.1
        self.adv_use_grl = bool(getattr(adv_cfg, "use_grl", True)) if adv_cfg else True
        self.adv_grl_lambda = float(getattr(adv_cfg, "grl_lambda", 1.0)) if adv_cfg else 1.0
        self.adv_schedule_lambda = bool(getattr(adv_cfg, "schedule_lambda", False)) if adv_cfg else False
        self.adv_schedule_gamma = float(getattr(adv_cfg, "schedule_gamma", 10.0)) if adv_cfg else 10.0
        # For alternating optimization (future extension)
        self.adv_alternating = bool(getattr(adv_cfg, "alternating", False)) if adv_cfg else False
        self.adv_n_critic = int(getattr(adv_cfg, "n_critic", 1)) if adv_cfg else 1
        
        if self.use_adversarial:
            if "discriminator" not in self.models:
                raise ValueError(
                    "adversarial.enabled=true but 'discriminator' model is missing. "
                    "Ensure experiments.run_trial builds it."
                )
            # BCE with logits for discriminator (numerically stable)
            self.bce = torch.nn.BCEWithLogitsLoss()
            # Track total epochs for lambda scheduling (set in fit())
            self._total_epochs = None

    def _cosine(self, a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).mean()

    def _forward_losses(self, batch, secondary_batch=None, eval_only=False):
        """
        Compute forward pass and losses.
        
        Args:
            batch: (pose, acc, labels) tuple from primary dataset
            secondary_batch: optional (pose, labels) from secondary pose-only dataset
            eval_only: if True, only compute real-branch predictions for val/test
                       (skip simulated branch, MSE, similarity losses)
        """
        pose, acc, labels = batch
        pose, acc, labels = pose.to(self.device), acc.to(self.device), labels.to(self.device)
        zero = torch.zeros((), device=self.device, dtype=pose.dtype)
        
        # For validation/test: only use real accelerometer through real branch
        if eval_only:
            real_feat = self.models["fe"](acc)
            logits_real = self.models[self.real_classifier_key](real_feat)
            act_loss_real = self.ce(logits_real, labels)
            # Return activity loss as total; zeros for training-only losses (MSE, sim, secondary, adv)
            return (
                act_loss_real,  # total loss = activity loss on real branch
                0.0,            # mse (not computed in eval)
                0.0,            # sim_loss (not computed in eval)
                float(act_loss_real.detach().cpu()),  # act_loss (real only)
                0.0,            # sec_act_loss (not computed in eval)
                0.0,            # adv_loss (not computed in eval)
                logits_real,
                labels,
            )
        
        # Training path: compute all branches
        sim_acc = self.models["pose2imu"](pose)
        mse = torch.nn.functional.mse_loss(sim_acc, acc)
        real_feat = self.models["fe"](acc)
        sim_feat = self.models[self.sim_fe_key](sim_acc)
        logits_real = self.models[self.real_classifier_key](real_feat)
        logits_sim = self.models[self.sim_classifier_key](sim_feat)

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

        # Secondary pose-only batch (e.g., NTU) - computed separately
        sec_act_loss = zero
        if secondary_batch is not None:
            sec_pose, sec_labels = secondary_batch
            sec_pose = sec_pose.to(self.device)
            sec_labels = sec_labels.to(self.device)
            sec_sim_acc = self.models["pose2imu"](sec_pose)
            sec_sim_feat = self.models[self.sim_fe_key](sec_sim_acc)
            sec_logits = self.models[self.secondary_classifier_key](sec_sim_feat)
            sec_act_loss = self.ce(sec_logits, sec_labels)

        # Adversarial loss (Scenario 4): discriminator classifies real vs simulated features
        adv_loss = zero
        if self.use_adversarial:
            D = self.models["discriminator"]
            batch_size = real_feat.size(0)
            real_labels = torch.ones(batch_size, 1, device=self.device)
            sim_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # Get current GRL lambda (may be scheduled)
            current_lambda = self._current_grl_lambda if hasattr(self, '_current_grl_lambda') else self.adv_grl_lambda
            
            # With GRL: single forward pass, gradients reversed for G/FE
            # D receives features through GRL → D loss backprops normally to D,
            # but gradients to FE/G are reversed (adversarial signal)
            d_real_logits = D(real_feat, apply_grl=True, grl_lambda=current_lambda)
            d_sim_logits = D(sim_feat, apply_grl=True, grl_lambda=current_lambda)
            
            # Discriminator loss: correctly classify real vs sim
            adv_loss = self.bce(d_real_logits, real_labels) + self.bce(d_sim_logits, sim_labels)
            
            # TODO: For secondary dataset, add sim features from secondary to adversarial
            # This would enforce NTU sim_acc to also look "real" by UTD standards
            # if secondary_batch is not None and self.use_adversarial:
            #     d_sec_sim_logits = D(sec_sim_feat, apply_grl=True, grl_lambda=current_lambda)
            #     adv_loss = adv_loss + self.bce(d_sec_sim_logits, sim_labels)

        total = torch.zeros((), device=self.device, dtype=pose.dtype)
        if self.use_mse_loss:
            total = total + self.gamma * mse
        if self.beta > 0 and self.use_feature_similarity_loss:
            total = total + self.beta * sim_loss
        if self.alpha > 0 and self.use_activity_loss:
            total = total + self.alpha * act_loss
        # Secondary loss added directly with its own weight (not nested under alpha)
        if secondary_batch is not None:
            total = total + self.secondary_loss_weight * sec_act_loss
        # Adversarial loss with its own weight
        if self.use_adversarial:
            total = total + self.adv_weight * adv_loss

        return (
            total,
            float(mse.detach().cpu()),
            float(sim_loss.detach().cpu()),
            float(act_loss.detach().cpu()),
            float(sec_act_loss.detach().cpu()) if secondary_batch is not None else 0.0,
            float(adv_loss.detach().cpu()) if self.use_adversarial else 0.0,
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
        eval_only = not is_train  # val/test: only use real accelerometer branch
        
        for m in self.models.values():
            m.train() if is_train else m.eval()

        secondary_iter = None
        if self.use_secondary_pose and split == "train":
            secondary_iter = iter(self.secondary_loader) if self.secondary_loader is not None else None

        total, mse_acc, sim_acc, act_acc, sec_acc, adv_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        preds, trues = [], []
        with torch.set_grad_enabled(is_train):
            for batch in self.dl[split]:
                sec_batch = None
                if secondary_iter is not None:
                    try:
                        sec_batch = next(secondary_iter)
                    except StopIteration:
                        secondary_iter = None
                total_loss, mse_l, sim_l, act_l, sec_l, adv_l, logits, labels = self._forward_losses(
                    batch, sec_batch, eval_only=eval_only
                )
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
                sec_acc += sec_l
                adv_acc += adv_l
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(labels.cpu().numpy())

        denom = max(len(self.dl[split]), 1)
        avg_loss = total / denom
        mse_avg = mse_acc / denom
        sim_avg = sim_acc / denom
        act_avg = act_acc / denom
        sec_avg = sec_acc / denom
        adv_avg = adv_acc / denom

        if trues:
            trues_arr = np.asarray(trues)
            preds_arr = np.asarray(preds)
            acc = float((preds_arr == trues_arr).mean())
            f1 = float(f1_score(trues_arr, preds_arr, average="macro"))
        else:
            acc = 0.0
            f1 = 0.0

        return avg_loss, f1, mse_avg, sim_avg, act_avg, sec_avg, adv_avg, acc

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
            "train_sec_loss",
            "train_adv_loss",
        ):
            raise KeyError(
                f"Objective metric '{self.objective_metric}' is not tracked by the Trainer. "
                "Choose one of: train_loss, val_loss, train_f1, val_f1, train_acc, val_acc, "
                "train_mse, val_mse, train_sim_loss, val_sim_loss, train_act_loss, val_act_loss, "
                "train_sec_loss, train_adv_loss."
            )

        # Store total epochs for GRL lambda scheduling
        if self.use_adversarial:
            self._total_epochs = epochs

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
            "train_sec_loss": [],
            "train_adv_loss": [],
        }
        
        print(f"Starting training for {epochs} epochs...")
        print("-" * 70)

        for epoch in range(epochs):
            self._apply_lr_warmup(epoch)
            
            # Update GRL lambda if scheduled (Scenario 4)
            if self.use_adversarial and self.adv_schedule_lambda:
                from src.models.discriminator import compute_grl_lambda_schedule
                self._current_grl_lambda = compute_grl_lambda_schedule(
                    epoch, epochs, gamma=self.adv_schedule_gamma
                )
            elif self.use_adversarial:
                self._current_grl_lambda = self.adv_grl_lambda
            
            tr_loss, tr_f1, tr_mse, tr_sim, tr_act, tr_sec, tr_adv, tr_acc = self._run_epoch("train")
            if self.skip_val:
                val_loss, val_f1, val_mse, val_sim, val_act, _, _, val_acc = tr_loss, tr_f1, tr_mse, tr_sim, tr_act, 0.0, 0.0, tr_acc
            else:
                val_loss, val_f1, val_mse, val_sim, val_act, _, _, val_acc = self._run_epoch("val")
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
            history["train_sec_loss"].append(tr_sec)
            history["train_adv_loss"].append(tr_adv)
            
            # Print epoch progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.3e}")

            if self.scheduler is not None and self._lr_warmup_finished:
                metric_name = self.scheduler_metric
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
