import torch
from utils import config, utils
from src.data import get_dataloaders
from src.train import Trainer
from utils.modules import Regressor, FeatureExtractor, ActivityClassifier

def main(cfg):
    dls = get_dataloaders(cfg.dataset_name, cfg)  # returns {'train','val','test'}
    models = {
        "pose2imu": Regressor(in_ch=cfg.in_ch, num_joints=cfg.num_joints, window_length=cfg.sensor_window_length).to(cfg.device),
        "fe": FeatureExtractor().to(cfg.device),
        "ac": ActivityClassifier(f_in=cfg.ac_fin, n_classes=cfg.ac_num_classes).to(cfg.device),
    }
    params = sum([list(m.parameters()) for m in models.values()], [])
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=cfg.patience)

    trainer = Trainer(models=models, dataloaders=dls, optimizer=optimizer, scheduler=scheduler, cfg=cfg, device=cfg.device)
    history = trainer.fit(cfg.epochs)

    # optional: save models / plots via utils
    utils.save_plot(epochs=len(history["train_loss"]), train_metric_history=history["train_loss"], val_metric_history=history["val_loss"], metric="Total Loss", file_name="scenario2_loss")
    return history

if __name__ == "__main__":
    main(config)