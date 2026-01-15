import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

try:  # Keep optional import; fallback if sklearn not installed
    from sklearn.metrics import f1_score, silhouette_score  # type: ignore
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

    def silhouette_score(features, labels):
        return 0.0  # Fallback if sklearn not available

# MMD + Contrastive losses for Scenario 5
from src.models.losses import MMDLoss, ContrastiveLoss

# WGAN-GP for Scenario 42, ACGAN support
from src.models.discriminator import (
    compute_gradient_penalty,
    compute_gradient_penalty_acgan,
    WGANLoss,
    ACSignalDiscriminator,
)

def _to_bool(value):
    """
    Convert a value to boolean, handling string representations.

    Hydra may pass boolean config values as strings (e.g., "False", "true").
    Python's bool("False") returns True (non-empty string), which is wrong.
    This helper correctly parses string booleans.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


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

        # Adversarial training support (Scenario 4/42)
        adv_cfg = getattr(self.trainer_cfg, "adversarial", None) if self.trainer_cfg else None
        self.use_adversarial = _to_bool(getattr(adv_cfg, "enabled", False)) if adv_cfg else False
        self.adv_weight = float(getattr(adv_cfg, "weight", 0.1)) if adv_cfg else 0.1
        self.adv_use_grl = _to_bool(getattr(adv_cfg, "use_grl", True)) if adv_cfg else True
        self.adv_grl_lambda = float(getattr(adv_cfg, "grl_lambda", 1.0)) if adv_cfg else 1.0
        self.adv_schedule_lambda = _to_bool(getattr(adv_cfg, "schedule_lambda", False)) if adv_cfg else False
        self.adv_schedule_gamma = float(getattr(adv_cfg, "schedule_gamma", 10.0)) if adv_cfg else 10.0
        # For alternating optimization (future extension)
        self.adv_alternating = _to_bool(getattr(adv_cfg, "alternating", False)) if adv_cfg else False
        self.adv_n_critic = int(getattr(adv_cfg, "n_critic", 1)) if adv_cfg else 1
        # Scenario 42: signal-level discrimination (D on raw acc instead of features)
        disc_cfg = getattr(adv_cfg, "discriminator", None) if adv_cfg else None
        disc_input_type = str(getattr(disc_cfg, "input_type", "features")).lower() if disc_cfg else "features"
        self.adv_signal_input = (disc_input_type == "signal")

        # WGAN vs BCE loss type
        self.adv_loss_type = str(getattr(disc_cfg, "loss_type", "bce")).lower() if disc_cfg else "bce"
        self.adv_lambda_gp = float(getattr(disc_cfg, "lambda_gp", 10.0)) if disc_cfg else 10.0

        # Staged training: pretrain with MSE before adversarial
        self.adv_pretrain_epochs = int(getattr(adv_cfg, "pretrain_epochs", 0)) if adv_cfg else 0

        # ACGAN: class-conditional discriminator with auxiliary classifier
        self.use_acgan = _to_bool(getattr(disc_cfg, "use_acgan", False)) if disc_cfg else False
        self.acgan_aux_weight = float(getattr(disc_cfg, "aux_weight", 1.0)) if disc_cfg else 1.0

        if self.use_adversarial:
            if "discriminator" not in self.models:
                raise ValueError(
                    "adversarial.enabled=true but 'discriminator' model is missing. "
                    "Ensure experiments.run_trial builds it."
                )
            # Validate ACGAN discriminator type
            if self.use_acgan and not isinstance(self.models["discriminator"], ACSignalDiscriminator):
                raise ValueError(
                    "use_acgan=true but discriminator is not ACSignalDiscriminator. "
                    "Ensure experiments.run_trial builds ACSignalDiscriminator when use_acgan is enabled."
                )
            # Loss function based on config
            if self.adv_loss_type == "wgan":
                self.wgan_loss = WGANLoss(lambda_gp=self.adv_lambda_gp, use_gp=True)
                self.bce = None  # Not used for WGAN
            else:
                # BCE with logits for discriminator (numerically stable)
                self.bce = torch.nn.BCEWithLogitsLoss()
                self.wgan_loss = None
            # CE loss for ACGAN auxiliary classifier (no label smoothing for aux)
            if self.use_acgan:
                self.aux_ce = torch.nn.CrossEntropyLoss()
            # Track total epochs for lambda scheduling (set in fit())
            self._total_epochs = None
            # Track current epoch for pretrain logic
            self._current_epoch = 0

            # Determine if we should use alternating optimization
            # WGAN should use alternating (not GRL) unless explicitly using GRL
            # For signal-level (scenario42): WGAN + alternating is recommended
            # For feature-level (scenario4): BCE + GRL is recommended
            self.use_alternating = (
                self.adv_alternating  # explicit config
                or (self.adv_loss_type == "wgan" and not self.adv_use_grl)  # WGAN without GRL
            )

            # Collect discriminator parameters for selective optimization
            if self.use_alternating:
                self.d_params = set(self.models["discriminator"].parameters())
                # Collect all non-D params (generator = pose2imu, fe, ac, etc.)
                self.g_params = set()
                for name, model in self.models.items():
                    if name != "discriminator":
                        self.g_params.update(model.parameters())

        # MMD + Contrastive training support (Scenario 5)
        mmd_cfg = getattr(self.trainer_cfg, "mmd", None) if self.trainer_cfg else None
        self.use_mmd = _to_bool(getattr(mmd_cfg, "enabled", False)) if mmd_cfg else False
        self.mmd_weight = float(getattr(mmd_cfg, "weight", 0.5)) if mmd_cfg else 0.5
        mmd_kernel_mul = float(getattr(mmd_cfg, "kernel_mul", 2.0)) if mmd_cfg else 2.0
        mmd_kernel_num = int(getattr(mmd_cfg, "kernel_num", 5)) if mmd_cfg else 5

        con_cfg = getattr(self.trainer_cfg, "contrastive", None) if self.trainer_cfg else None
        self.use_contrastive = _to_bool(getattr(con_cfg, "enabled", False)) if con_cfg else False
        self.contrastive_weight = float(getattr(con_cfg, "weight", 0.3)) if con_cfg else 0.3
        contrastive_temp = float(getattr(con_cfg, "temperature", 0.5)) if con_cfg else 0.5

        # Initialize loss functions if enabled
        if self.use_mmd:
            self.mmd_loss_fn = MMDLoss(kernel_mul=mmd_kernel_mul, kernel_num=mmd_kernel_num)
        if self.use_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temp)

    def _cosine(self, a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).mean()

    def _compute_silhouette(self, real_feat, sim_feat, real_labels, sim_labels):
        """Compute silhouette score for feature quality monitoring (Scenario 5)."""
        try:
            all_feat = torch.cat([real_feat, sim_feat], dim=0).detach().cpu().numpy()
            all_labels = torch.cat([real_labels, sim_labels], dim=0).detach().cpu().numpy()
            if len(np.unique(all_labels)) > 1 and len(all_feat) > len(np.unique(all_labels)):
                return float(silhouette_score(all_feat, all_labels))
        except Exception:
            pass
        return 0.0

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
            d_acc = 0.0
            if self.use_adversarial:
                sim_acc = self.models["pose2imu"](pose)
                D = self.models["discriminator"]
                if self.adv_signal_input:
                    d_input_real = acc
                    d_input_sim = sim_acc
                else:
                    sim_feat = self.models[self.sim_fe_key](sim_acc)
                    d_input_real = real_feat
                    d_input_sim = sim_feat
                if self.use_acgan:
                    d_real_out = D(d_input_real, labels=labels, apply_grl=False)
                    d_sim_out = D(d_input_sim, labels=labels, apply_grl=False)
                    d_real_logits, _ = d_real_out
                    d_sim_logits, _ = d_sim_out
                else:
                    d_real_logits = D(d_input_real, apply_grl=False)
                    d_sim_logits = D(d_input_sim, apply_grl=False)
                d_pred_real = (torch.sigmoid(d_real_logits) > 0.5).float()
                d_pred_sim = (torch.sigmoid(d_sim_logits) < 0.5).float()
                d_acc = float((d_pred_real.mean() + d_pred_sim.mean()) / 2)
            # Return activity loss as total; zeros for training-only losses
            return (
                act_loss_real,  # total loss = activity loss on real branch
                0.0,            # mse (not computed in eval)
                0.0,            # sim_loss (not computed in eval)
                float(act_loss_real.detach().cpu()),  # act_loss (real only)
                0.0,            # sec_act_loss (not computed in eval)
                0.0,            # adv_loss (not computed in eval)
                d_acc,          # d_acc (computed for eval diagnostics)
                0.0,            # feat_dist (not computed in eval)
                0.0,            # mmd_loss (not computed in eval)
                0.0,            # contrastive_loss (not computed in eval)
                0.0,            # silhouette (not computed in eval)
                0.0,            # aux_loss (not computed in eval)
                0.0,            # aux_acc (not computed in eval)
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

        # Adversarial loss (Scenario 4/42): discriminator classifies real vs simulated
        adv_loss = zero
        aux_loss = zero  # ACGAN auxiliary classifier loss
        d_acc = 0.0
        feat_dist = 0.0
        aux_acc = 0.0  # ACGAN auxiliary classifier accuracy

        # Check if we're in pretrain phase (skip adversarial)
        in_pretrain = (
            self.use_adversarial
            and self.adv_pretrain_epochs > 0
            and hasattr(self, '_current_epoch')
            and self._current_epoch < self.adv_pretrain_epochs
        )

        if self.use_adversarial and not in_pretrain:
            D = self.models["discriminator"]
            batch_size = real_feat.size(0)

            # Choose discriminator input based on mode
            if self.adv_signal_input:
                # Scenario 42: D operates on raw accelerometer signals
                d_input_real = acc      # real accelerometer
                d_input_sim = sim_acc   # simulated accelerometer from regressor
            else:
                # Scenario 4: D operates on encoder features
                d_input_real = real_feat
                d_input_sim = sim_feat

            # Check if using alternating optimization (for WGAN)
            use_alternating = getattr(self, 'use_alternating', False)

            # Helper to unpack discriminator output (handles both regular and ACGAN)
            def _unpack_d_output(d_out):
                """Unpack discriminator output: (adv_logits,) or (adv_logits, aux_logits)."""
                if isinstance(d_out, tuple):
                    return d_out[0], d_out[1]  # ACGAN: (adv, aux)
                return d_out, None  # Regular: just adv logits

            if use_alternating and self.adv_loss_type == "wgan":
                # WGAN with alternating D/G updates (proper approach for signal-level)
                # This is called from _run_epoch which handles the alternating logic
                # Here we just compute the combined loss for logging purposes

                # Forward pass without GRL
                if self.use_acgan:
                    d_real_out = D(d_input_real, labels=labels, apply_grl=False)
                    d_sim_out = D(d_input_sim, labels=labels, apply_grl=False)
                else:
                    d_real_out = D(d_input_real, apply_grl=False)
                    d_sim_out = D(d_input_sim, apply_grl=False)

                d_real_logits, aux_real_logits = _unpack_d_output(d_real_out)
                d_sim_logits, aux_sim_logits = _unpack_d_output(d_sim_out)

                # Combined for logging: D(real) - D(fake) (Wasserstein distance estimate)
                adv_loss = d_real_logits.mean() - d_sim_logits.mean()

                # ACGAN auxiliary loss
                if self.use_acgan and aux_real_logits is not None:
                    aux_loss = self.aux_ce(aux_real_logits, labels) + self.aux_ce(aux_sim_logits, labels)

            elif self.adv_use_grl:
                # GRL mode (for BCE + feature-level, scenario 4)
                current_lambda = self._current_grl_lambda if hasattr(self, '_current_grl_lambda') else self.adv_grl_lambda

                # Forward pass through discriminator with GRL
                if self.use_acgan:
                    d_real_out = D(d_input_real, labels=labels, apply_grl=True, grl_lambda=current_lambda)
                    d_sim_out = D(d_input_sim, labels=labels, apply_grl=True, grl_lambda=current_lambda)
                else:
                    d_real_out = D(d_input_real, apply_grl=True, grl_lambda=current_lambda)
                    d_sim_out = D(d_input_sim, apply_grl=True, grl_lambda=current_lambda)

                d_real_logits, aux_real_logits = _unpack_d_output(d_real_out)
                d_sim_logits, aux_sim_logits = _unpack_d_output(d_sim_out)

                if self.adv_loss_type == "wgan":
                    # WGAN with GRL (unconventional but supported)
                    if self.use_acgan:
                        gp = compute_gradient_penalty_acgan(
                            D, d_input_real.detach(), d_input_sim.detach(),
                            labels, self.device, lambda_gp=self.adv_lambda_gp
                        )
                    else:
                        gp = compute_gradient_penalty(
                            D, d_input_real.detach(), d_input_sim.detach(),
                            self.device, lambda_gp=self.adv_lambda_gp
                        )
                    adv_loss = self.wgan_loss.combined_loss_with_grl(d_real_logits, d_sim_logits, gp)
                else:
                    # BCE loss (vanilla GAN) - standard for scenario 4
                    if hasattr(D, 'get_smooth_labels'):
                        real_labels_d = D.get_smooth_labels(batch_size, real=True, device=self.device)
                        sim_labels_d = D.get_smooth_labels(batch_size, real=False, device=self.device)
                    else:
                        real_labels_d = torch.ones(batch_size, 1, device=self.device)
                        sim_labels_d = torch.zeros(batch_size, 1, device=self.device)
                    adv_loss = self.bce(d_real_logits, real_labels_d) + self.bce(d_sim_logits, sim_labels_d)

                # ACGAN auxiliary loss
                if self.use_acgan and aux_real_logits is not None:
                    aux_loss = self.aux_ce(aux_real_logits, labels) + self.aux_ce(aux_sim_logits, labels)

            else:
                # No GRL, no alternating - just forward pass
                if self.use_acgan:
                    d_real_out = D(d_input_real, labels=labels, apply_grl=False)
                    d_sim_out = D(d_input_sim, labels=labels, apply_grl=False)
                else:
                    d_real_out = D(d_input_real, apply_grl=False)
                    d_sim_out = D(d_input_sim, apply_grl=False)

                d_real_logits, aux_real_logits = _unpack_d_output(d_real_out)
                d_sim_logits, aux_sim_logits = _unpack_d_output(d_sim_out)

                if hasattr(D, 'get_smooth_labels'):
                    real_labels_d = D.get_smooth_labels(batch_size, real=True, device=self.device)
                    sim_labels_d = D.get_smooth_labels(batch_size, real=False, device=self.device)
                else:
                    real_labels_d = torch.ones(batch_size, 1, device=self.device)
                    sim_labels_d = torch.zeros(batch_size, 1, device=self.device)
                adv_loss = self.bce(d_real_logits, real_labels_d) + self.bce(d_sim_logits, sim_labels_d)

                # ACGAN auxiliary loss
                if self.use_acgan and aux_real_logits is not None:
                    aux_loss = self.aux_ce(aux_real_logits, labels) + self.aux_ce(aux_sim_logits, labels)

            # --- Adversarial diagnostics ---
            with torch.no_grad():
                # D accuracy: how well can D distinguish real from sim?
                if self.adv_loss_type == "wgan":
                    d_pred_real = (d_real_logits > 0).float()
                    d_pred_sim = (d_sim_logits < 0).float()
                else:
                    d_pred_real = (torch.sigmoid(d_real_logits) > 0.5).float()
                    d_pred_sim = (torch.sigmoid(d_sim_logits) < 0.5).float()
                d_acc = float((d_pred_real.mean() + d_pred_sim.mean()) / 2)

                # Distance metric
                if self.adv_signal_input:
                    real_flat = acc.view(batch_size, -1)
                    sim_flat = sim_acc.view(batch_size, -1)
                    feat_dist = float(torch.norm(real_flat.mean(0) - sim_flat.mean(0), p=2).cpu())
                else:
                    feat_dist = float(torch.norm(real_feat.mean(0) - sim_feat.mean(0), p=2).cpu())

                # ACGAN auxiliary classifier accuracy
                if self.use_acgan and aux_real_logits is not None:
                    aux_pred_real = aux_real_logits.argmax(dim=1)
                    aux_pred_sim = aux_sim_logits.argmax(dim=1)
                    aux_acc = float(((aux_pred_real == labels).float().mean() +
                                    (aux_pred_sim == labels).float().mean()) / 2)

        # MMD loss (Scenario 5): domain alignment without adversarial
        mmd_loss = zero
        if self.use_mmd:
            mmd_loss = self.mmd_loss_fn(real_feat, sim_feat)

        # Contrastive loss (Scenario 5): class structure preservation
        contrastive_loss = zero
        if self.use_contrastive:
            all_features = torch.cat([real_feat, sim_feat], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)  # same activity labels
            contrastive_loss = self.contrastive_loss_fn(all_features, all_labels)

        # Silhouette score (Scenario 5 diagnostic)
        sil_score = 0.0
        if self.use_mmd or self.use_contrastive:
            with torch.no_grad():
                sil_score = self._compute_silhouette(real_feat, sim_feat, labels, labels)
                # Also compute feat_dist if not already computed by adversarial
                if not self.use_adversarial:
                    feat_dist = float(torch.norm(real_feat.mean(0) - sim_feat.mean(0), p=2).cpu())

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
        # ACGAN auxiliary classifier loss
        if self.use_acgan:
            total = total + self.acgan_aux_weight * aux_loss
        # MMD + Contrastive losses (Scenario 5)
        if self.use_mmd:
            total = total + self.mmd_weight * mmd_loss
        if self.use_contrastive:
            total = total + self.contrastive_weight * contrastive_loss

        return (
            total,
            float(mse.detach().cpu()),
            float(sim_loss.detach().cpu()),
            float(act_loss.detach().cpu()),
            float(sec_act_loss.detach().cpu()) if secondary_batch is not None else 0.0,
            float(adv_loss.detach().cpu()) if self.use_adversarial else 0.0,
            d_acc,
            feat_dist,
            float(mmd_loss.detach().cpu()) if isinstance(mmd_loss, torch.Tensor) else 0.0,
            float(contrastive_loss.detach().cpu()) if isinstance(contrastive_loss, torch.Tensor) else 0.0,
            sil_score,
            float(aux_loss.detach().cpu()) if isinstance(aux_loss, torch.Tensor) else 0.0,
            aux_acc,
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

    def _alternating_step(self, batch, secondary_batch=None):
        """
        Perform alternating D/G updates for WGAN training.

        This implements the proper WGAN-GP training procedure:
        1. Update D for n_critic steps (D learns to distinguish real from fake)
        2. Update G for 1 step (G learns to fool D)

        Returns the same tuple as _forward_losses for consistency.
        """
        pose, acc, labels = batch
        pose, acc, labels = pose.to(self.device), acc.to(self.device), labels.to(self.device)

        D = self.models["discriminator"]
        zero = torch.zeros((), device=self.device, dtype=pose.dtype)

        # Helper to unpack discriminator output (handles both regular and ACGAN)
        def _unpack_d_output(d_out):
            if isinstance(d_out, tuple):
                return d_out[0], d_out[1]  # ACGAN: (adv, aux)
            return d_out, None  # Regular: just adv logits

        # ============ D UPDATE (n_critic times) ============
        for _ in range(self.adv_n_critic):
            # Forward pass through generator (no grad needed for D update)
            with torch.no_grad():
                sim_acc_d = self.models["pose2imu"](pose)

            # Choose D input based on mode
            if self.adv_signal_input:
                d_input_real = acc
                d_input_sim = sim_acc_d
            else:
                real_feat_d = self.models["fe"](acc)
                sim_feat_d = self.models[self.sim_fe_key](sim_acc_d)
                d_input_real = real_feat_d
                d_input_sim = sim_feat_d

            # D forward (no GRL - we're doing alternating)
            if self.use_acgan:
                d_real_out = D(d_input_real, labels=labels, apply_grl=False)
                d_fake_out = D(d_input_sim.detach(), labels=labels, apply_grl=False)
                d_real, aux_real = _unpack_d_output(d_real_out)
                d_fake, aux_fake = _unpack_d_output(d_fake_out)
            else:
                d_real = D(d_input_real, apply_grl=False)
                d_fake = D(d_input_sim.detach(), apply_grl=False)
                aux_real, aux_fake = None, None

            # WGAN-GP: D loss = D(fake) - D(real) + GP
            if self.use_acgan:
                gp = compute_gradient_penalty_acgan(
                    D, d_input_real, d_input_sim.detach(),
                    labels, self.device, lambda_gp=self.adv_lambda_gp
                )
            else:
                gp = compute_gradient_penalty(
                    D, d_input_real, d_input_sim.detach(),
                    self.device, lambda_gp=self.adv_lambda_gp
                )
            d_loss = self.wgan_loss.discriminator_loss(d_real, d_fake, gp)

            # ACGAN: add auxiliary classification loss to D
            if self.use_acgan and aux_real is not None:
                d_aux_loss = self.aux_ce(aux_real, labels) + self.aux_ce(aux_fake, labels)
                d_loss = d_loss + self.acgan_aux_weight * d_aux_loss

            # Update only D params
            self.optimizer.zero_grad()
            d_loss.backward()

            # Zero out non-D gradients before step
            for param in self.g_params:
                if param.grad is not None:
                    param.grad.zero_()

            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(D.parameters(), self.gradient_clip)
            self.optimizer.step()

        # ============ G UPDATE (1 time) ============
        # Full forward pass for G and other losses
        sim_acc_g = self.models["pose2imu"](pose)
        mse = torch.nn.functional.mse_loss(sim_acc_g, acc)

        real_feat = self.models["fe"](acc)
        sim_feat = self.models[self.sim_fe_key](sim_acc_g)
        logits_real = self.models[self.real_classifier_key](real_feat)
        logits_sim = self.models[self.sim_classifier_key](sim_feat)

        # Feature similarity loss
        if self.beta > 0 and self.use_feature_similarity_loss:
            sim_loss = self._cosine(sim_feat, real_feat)
        else:
            sim_loss = zero

        # Activity loss
        act_loss = zero
        if self.use_activity_loss and self.alpha > 0:
            if self.use_activity_loss_real:
                act_loss = act_loss + self.ce(logits_real, labels)
            if self.use_activity_loss_sim:
                act_loss = act_loss + self.ce(logits_sim, labels)

        # Secondary loss
        sec_act_loss = zero
        if secondary_batch is not None:
            sec_pose, sec_labels = secondary_batch
            sec_pose = sec_pose.to(self.device)
            sec_labels = sec_labels.to(self.device)
            sec_sim_acc = self.models["pose2imu"](sec_pose)
            sec_sim_feat = self.models[self.sim_fe_key](sec_sim_acc)
            sec_logits = self.models[self.secondary_classifier_key](sec_sim_feat)
            sec_act_loss = self.ce(sec_logits, sec_labels)

        # G adversarial loss: G wants to maximize D(fake), i.e., minimize -D(fake)
        if self.adv_signal_input:
            d_input_sim_g = sim_acc_g
        else:
            d_input_sim_g = sim_feat

        if self.use_acgan:
            d_fake_out_g = D(d_input_sim_g, labels=labels, apply_grl=False)
            d_fake_for_g, aux_fake_g = _unpack_d_output(d_fake_out_g)
        else:
            d_fake_for_g = D(d_input_sim_g, apply_grl=False)
            aux_fake_g = None

        g_adv_loss = self.wgan_loss.generator_loss(d_fake_for_g)

        # ACGAN auxiliary loss for G (G wants sim samples to be correctly classified)
        aux_loss = zero
        if self.use_acgan and aux_fake_g is not None:
            aux_loss = self.aux_ce(aux_fake_g, labels)

        # Total G loss
        g_total = zero
        if self.use_mse_loss:
            g_total = g_total + self.gamma * mse
        if self.beta > 0 and self.use_feature_similarity_loss:
            g_total = g_total + self.beta * sim_loss
        if self.alpha > 0 and self.use_activity_loss:
            g_total = g_total + self.alpha * act_loss
        if secondary_batch is not None:
            g_total = g_total + self.secondary_loss_weight * sec_act_loss
        # Add G adversarial loss
        g_total = g_total + self.adv_weight * g_adv_loss
        # Add ACGAN auxiliary loss for G
        if self.use_acgan:
            g_total = g_total + self.acgan_aux_weight * aux_loss

        # Update only G params (non-D)
        self.optimizer.zero_grad()
        g_total.backward()

        # Zero out D gradients before step
        for param in self.d_params:
            if param.grad is not None:
                param.grad.zero_()

        if self.gradient_clip:
            clip_params = list(self.g_params)
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, self.gradient_clip)
        self.optimizer.step()

        # ============ DIAGNOSTICS ============
        aux_acc = 0.0
        with torch.no_grad():
            # D accuracy
            if self.use_acgan:
                d_real_diag_out = D(d_input_real if not self.adv_signal_input else acc, labels=labels, apply_grl=False)
                d_fake_diag_out = D(d_input_sim_g.detach(), labels=labels, apply_grl=False)
                d_real_diag, aux_real_diag = _unpack_d_output(d_real_diag_out)
                d_fake_diag, aux_fake_diag = _unpack_d_output(d_fake_diag_out)
            else:
                d_real_diag = D(d_input_real if not self.adv_signal_input else acc, apply_grl=False)
                d_fake_diag = D(d_input_sim_g.detach(), apply_grl=False)
                aux_real_diag, aux_fake_diag = None, None

            d_pred_real = (d_real_diag > 0).float()
            d_pred_sim = (d_fake_diag < 0).float()
            d_acc = float((d_pred_real.mean() + d_pred_sim.mean()) / 2)

            # Distance metric
            if self.adv_signal_input:
                batch_size = acc.size(0)
                real_flat = acc.view(batch_size, -1)
                sim_flat = sim_acc_g.view(batch_size, -1)
                feat_dist = float(torch.norm(real_flat.mean(0) - sim_flat.mean(0), p=2).cpu())
            else:
                feat_dist = float(torch.norm(real_feat.mean(0) - sim_feat.mean(0), p=2).cpu())

            # Wasserstein distance estimate for logging
            adv_loss_val = float((d_real_diag.mean() - d_fake_diag.mean()).cpu())

            # ACGAN auxiliary classifier accuracy
            if self.use_acgan and aux_real_diag is not None:
                aux_pred_real = aux_real_diag.argmax(dim=1)
                aux_pred_sim = aux_fake_diag.argmax(dim=1)
                aux_acc = float(((aux_pred_real == labels).float().mean() +
                                (aux_pred_sim == labels).float().mean()) / 2)

        return (
            g_total,
            float(mse.detach().cpu()),
            float(sim_loss.detach().cpu()) if isinstance(sim_loss, torch.Tensor) else 0.0,
            float(act_loss.detach().cpu()) if isinstance(act_loss, torch.Tensor) else 0.0,
            float(sec_act_loss.detach().cpu()) if secondary_batch is not None else 0.0,
            adv_loss_val,
            d_acc,
            feat_dist,
            0.0,  # mmd_loss
            0.0,  # contrastive_loss
            0.0,  # sil_score
            float(aux_loss.detach().cpu()) if isinstance(aux_loss, torch.Tensor) else 0.0,
            aux_acc,
            logits_real,
            labels,
        )

    def _run_epoch(self, split="train"):
        is_train = split == "train"
        eval_only = not is_train  # val/test: only use real accelerometer branch

        # Check if we should use alternating training
        use_alternating = (
            is_train
            and self.use_adversarial
            and getattr(self, 'use_alternating', False)
            and not (
                self.adv_pretrain_epochs > 0
                and hasattr(self, '_current_epoch')
                and self._current_epoch < self.adv_pretrain_epochs
            )
        )

        for m in self.models.values():
            m.train() if is_train else m.eval()

        secondary_iter = None
        if self.use_secondary_pose and split == "train":
            secondary_iter = iter(self.secondary_loader) if self.secondary_loader is not None else None

        total, mse_acc, sim_acc, act_acc, sec_acc, adv_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        d_acc_acc, feat_dist_acc = 0.0, 0.0  # Adversarial diagnostics
        mmd_acc, con_acc, sil_acc = 0.0, 0.0, 0.0  # MMD + Contrastive diagnostics
        aux_loss_acc, aux_acc_acc = 0.0, 0.0  # ACGAN diagnostics
        preds, trues = [], []
        with torch.set_grad_enabled(is_train):
            for batch in self.dl[split]:
                sec_batch = None
                if secondary_iter is not None:
                    try:
                        sec_batch = next(secondary_iter)
                    except StopIteration:
                        secondary_iter = None

                if use_alternating:
                    # Use alternating D/G updates for WGAN
                    (total_loss, mse_l, sim_l, act_l, sec_l, adv_l, d_acc_l, feat_dist_l,
                     mmd_l, con_l, sil_l, aux_l, aux_acc_l, logits, labels) = self._alternating_step(
                        batch, sec_batch
                    )
                    # No separate backward/step needed - done in _alternating_step
                else:
                    # Standard training (GRL or non-adversarial)
                    (total_loss, mse_l, sim_l, act_l, sec_l, adv_l, d_acc_l, feat_dist_l,
                     mmd_l, con_l, sil_l, aux_l, aux_acc_l, logits, labels) = self._forward_losses(
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
                d_acc_acc += d_acc_l
                feat_dist_acc += feat_dist_l
                mmd_acc += mmd_l
                con_acc += con_l
                sil_acc += sil_l
                aux_loss_acc += aux_l
                aux_acc_acc += aux_acc_l
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
        d_acc_avg = d_acc_acc / denom
        feat_dist_avg = feat_dist_acc / denom
        mmd_avg = mmd_acc / denom
        con_avg = con_acc / denom
        sil_avg = sil_acc / denom
        aux_loss_avg = aux_loss_acc / denom
        aux_acc_avg = aux_acc_acc / denom

        if trues:
            trues_arr = np.asarray(trues)
            preds_arr = np.asarray(preds)
            acc = float((preds_arr == trues_arr).mean())
            f1 = float(f1_score(trues_arr, preds_arr, average="macro"))
        else:
            acc = 0.0
            f1 = 0.0

        return avg_loss, f1, mse_avg, sim_avg, act_avg, sec_avg, adv_avg, d_acc_avg, feat_dist_avg, mmd_avg, con_avg, sil_avg, aux_loss_avg, aux_acc_avg, acc

    def fit(self, epochs):
        valid_metrics = (
            "train_loss", "val_loss", "train_f1", "val_f1", "train_acc", "val_acc",
            "train_mse", "val_mse", "train_sim_loss", "val_sim_loss",
            "train_act_loss", "val_act_loss", "train_sec_loss",
            "train_adv_loss", "train_d_acc", "val_d_acc", "train_feat_dist", "train_signal_dist",
            "train_mmd_loss", "train_contrastive_loss", "train_silhouette",
            "train_aux_loss", "train_aux_acc",  # ACGAN metrics
        )
        if self.objective_metric not in valid_metrics:
            raise KeyError(
                f"Objective metric '{self.objective_metric}' is not tracked by the Trainer. "
                f"Choose one of: {', '.join(valid_metrics)}"
            )

        # Store total epochs for GRL lambda scheduling
        if self.use_adversarial:
            self._total_epochs = epochs

        best_default = np.inf if self.objective_mode == "min" else -np.inf
        best = {"value": best_default, "state": None, "epoch": None}
        history = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_acc": [], "val_acc": [],
            "train_mse": [], "val_mse": [],
            "train_sim_loss": [], "val_sim_loss": [],
            "train_act_loss": [], "val_act_loss": [],
            "train_sec_loss": [],
            "train_adv_loss": [], "train_d_acc": [], "val_d_acc": [],
            "train_feat_dist": [], "train_signal_dist": [],
            "train_grl_lambda": [],
            "train_mmd_loss": [], "train_contrastive_loss": [], "train_silhouette": [],
            "train_aux_loss": [], "train_aux_acc": [],  # ACGAN metrics
            "val_aux_loss": [], "val_aux_acc": [],  # ACGAN validation metrics
        }

        # Print training mode
        if self.use_mmd or self.use_contrastive:
            print(f"Starting Scenario 5 training (MMD + Contrastive) for {epochs} epochs...")
            print(f"Loss weights: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ(mmd)={self.mmd_weight}, ε(con)={self.contrastive_weight}")
        elif self.use_adversarial:
            loss_type_str = "WGAN-GP" if self.adv_loss_type == "wgan" else "BCE"
            input_type_str = "signal" if self.adv_signal_input else "features"
            use_alternating = getattr(self, 'use_alternating', False)
            opt_mode = "alternating D/G" if use_alternating else "GRL"
            print(f"Starting adversarial training for {epochs} epochs...")
            print(f"  Discriminator: {input_type_str}-level, loss={loss_type_str}, optimization={opt_mode}")
            print(f"  Adversarial weight: {self.adv_weight}" + (f", n_critic={self.adv_n_critic}" if use_alternating else ""))
            if self.adv_pretrain_epochs > 0:
                print(f"  Staged training: {self.adv_pretrain_epochs} pretrain epochs (no adversarial), then adversarial")
            if self.adv_loss_type == "wgan":
                print(f"  WGAN-GP: λ_gp={self.adv_lambda_gp}")
        else:
            print(f"Starting training for {epochs} epochs...")
        print("-" * 80)

        for epoch in range(epochs):
            # Track current epoch for pretrain logic
            if self.use_adversarial:
                self._current_epoch = epoch

            self._apply_lr_warmup(epoch)

            # Check if we're in pretrain phase
            in_pretrain = (
                self.use_adversarial
                and self.adv_pretrain_epochs > 0
                and epoch < self.adv_pretrain_epochs
            )

            # Update GRL lambda if scheduled (Scenario 4)
            # During pretrain, lambda is 0 (no adversarial gradient)
            if self.use_adversarial:
                if in_pretrain:
                    self._current_grl_lambda = 0.0
                elif self.adv_schedule_lambda:
                    from src.models.discriminator import compute_grl_lambda_schedule
                    # Adjust epoch for scheduling: start from 0 after pretrain
                    adjusted_epoch = epoch - self.adv_pretrain_epochs
                    adjusted_total = epochs - self.adv_pretrain_epochs
                    self._current_grl_lambda = compute_grl_lambda_schedule(
                        adjusted_epoch, adjusted_total, gamma=self.adv_schedule_gamma
                    )
                else:
                    self._current_grl_lambda = self.adv_grl_lambda

            (tr_loss, tr_f1, tr_mse, tr_sim, tr_act, tr_sec, tr_adv, tr_d_acc,
             tr_feat_dist, tr_mmd, tr_con, tr_sil, tr_aux_loss, tr_aux_acc, tr_acc) = self._run_epoch("train")
            if self.skip_val:
                (val_loss, val_f1, val_mse, val_sim, val_act, val_sec, val_adv, val_d_acc,
                 val_feat_dist, val_mmd, val_con, val_sil, val_aux_loss, val_aux_acc, val_acc) = (
                    tr_loss, tr_f1, tr_mse, tr_sim, tr_act, tr_sec, tr_adv, tr_d_acc,
                    tr_feat_dist, tr_mmd, tr_con, tr_sil, tr_aux_loss, tr_aux_acc, tr_acc
                )
            else:
                (val_loss, val_f1, val_mse, val_sim, val_act, val_sec, val_adv, val_d_acc,
                 val_feat_dist, val_mmd, val_con, val_sil, val_aux_loss, val_aux_acc, val_acc) = self._run_epoch("val")
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
            history["train_d_acc"].append(tr_d_acc)
            history["val_d_acc"].append(val_d_acc)
            history["train_feat_dist"].append(tr_feat_dist)
            history["train_signal_dist"].append(tr_feat_dist if self.adv_signal_input else 0.0)
            history["train_mmd_loss"].append(tr_mmd)
            history["train_contrastive_loss"].append(tr_con)
            history["train_silhouette"].append(tr_sil)
            # Track GRL lambda for adversarial training
            grl_lam = getattr(self, '_current_grl_lambda', self.adv_grl_lambda) if self.use_adversarial else 0.0
            history["train_grl_lambda"].append(grl_lam)
            # ACGAN metrics
            history["train_aux_loss"].append(tr_aux_loss)
            history["train_aux_acc"].append(tr_aux_acc)
            history["val_aux_loss"].append(val_aux_loss)
            history["val_aux_acc"].append(val_aux_acc)

            # Print epoch progress (with scenario-specific diagnostics)
            if self.use_mmd or self.use_contrastive:
                # Scenario 5: MMD + Contrastive diagnostics
                sil_status = "✓" if tr_sil > 0.3 else "⚠️" if tr_sil > 0.1 else "❌"
                base_msg = (f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {tr_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"MMD: {tr_mmd:.4f} | Con: {tr_con:.4f} | "
                      f"Feat_dist: {tr_feat_dist:.2f} | Sil: {tr_sil:.3f} {sil_status} | "
                      f"LR: {current_lr:.3e}")
            else:
                base_msg = (f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | "
                      f"LR: {current_lr:.3e}")
                if self.use_adversarial:
                    if in_pretrain:
                        base_msg += " | [PRETRAIN - no adversarial]"
                    else:
                        dist_label = "Signal_dist" if self.adv_signal_input else "Feat_dist"
                        base_msg += f" | λ: {grl_lam:.2f} | D_acc: {tr_d_acc:.3f} | {dist_label}: {tr_feat_dist:.2f}"
            print(base_msg)

            # Print transition message when exiting pretrain
            if (self.use_adversarial and self.adv_pretrain_epochs > 0
                    and epoch == self.adv_pretrain_epochs - 1):
                loss_type_str = "WGAN-GP" if self.adv_loss_type == "wgan" else "BCE"
                opt_mode = "alternating" if getattr(self, 'use_alternating', False) else "GRL"
                print(f"    → Pretrain complete. Starting {loss_type_str} adversarial training ({opt_mode})...")

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

            # Collapse warning for Scenario 5
            if (self.use_mmd or self.use_contrastive) and tr_sil < 0.1 and epoch > 10:
                print(f"    ⚠️ WARNING: Low silhouette score ({tr_sil:.3f}) - possible feature collapse!")

        print("-" * 80)
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
