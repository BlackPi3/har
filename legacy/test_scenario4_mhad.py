# region Imports
import copy
import torch.nn as nn
import torch
from utils import config
from utils import utils
from datasets.mhad_dataloader import mhad_dataloader
from legacy.utils.modules import RegressorNew, Regressor
import sys

sys.path.append("..")
# endregion

# region MHAD dataset
mhad_train_dataloader, mhad_val_dataloader, mhad_test_dataloader = mhad_dataloader()
# endregion

# region Models
window = int(config.mhad_window_sec * config.mhad_accel_sampling_rate)
# pose2imu_model = RegressorNew(
#     in_ch=config.in_ch, num_joints=config.num_joints, window_length=window
# ).to(config.device, config.dtype)
pose2imu_model = Regressor(
    in_ch=config.in_ch, num_joints=config.num_joints, window_length=window
).to(config.device, config.dtype)

# >>> Loss + Optimization <<< #
MSELoss = nn.MSELoss()

def smoothness_loss(pred):
    diff = torch.diff(pred, dim=2)  # Compute differences along the temporal axis
    loss = torch.mean(diff ** 2)  # Penalize large differences (encourages smooth transitions)
    return loss

params = (
    list(pose2imu_model.parameters())
)
optimizer = torch.optim.Adam(params, lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=15
)
# endregion

# region Training + Validation
(
    train_loss_history,
    train_mse_loss_history,
) = [], [], 
(
    val_loss_history,
    val_mse_loss_history,
) = [], [], 

best_pose2imu_model_state = None
best_val_mse_loss = float("inf")
best_epoch = float("inf")

epochs_no_improve = 0
log = ""

for epoch in range(config.epochs):
    # Initialize tracking variables
    (
        total_train_loss,
        total_train_mse_loss,
    ) = 0, 0

    # Set models to training mode
    pose2imu_model.train()

    # region Training
    for accel, skel, _ in mhad_train_dataloader:
        # move to GPU
        accel = accel.to(config.device)
        skel = skel.to(config.device)

        # -- Forward pass
        # Conv3d expect input of shape (N, C, D, H, W)
        # so we reshape the skel to (N, 3, frames, joints, 1)
        sim_accel = pose2imu_model(skel)
        mse_loss = MSELoss(sim_accel, accel)
        total_train_mse_loss += mse_loss.item()

        # Total loss
        total_loss = (
            mse_loss + smoothness_loss(sim_accel)
        )
        total_train_loss += total_loss.item()

        # -- Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Average losses
    average_train_loss = total_train_loss / len(mhad_train_dataloader)
    train_loss_history.append(average_train_loss)
    # --  MSE Loss
    average_train_mse_loss = total_train_mse_loss / len(mhad_train_dataloader)
    train_mse_loss_history.append(average_train_mse_loss)
    # endregion

    # region Validation
    (
        total_val_loss,
        total_val_mse_loss,
    ) = 0, 0
    total_predictions, correct_predictions = 0, 0

    pose2imu_model.eval()
    with torch.no_grad():
        for accel, skel, _ in mhad_val_dataloader:
            # -- Move to GPU
            skel = skel.to(config.device)
            accel = accel.to(config.device)

            # -- Forward pass
            # --- Regressor
            sim_accel = pose2imu_model(skel)
            mse_loss = MSELoss(sim_accel, accel)
            total_val_mse_loss += mse_loss.item()

            # -- Total Loss
            total_loss = (
                mse_loss + smoothness_loss(sim_accel)
            )
            total_val_loss += total_loss.item()

    # Average losses
    average_val_loss = total_val_loss / len(mhad_val_dataloader)
    val_loss_history.append(average_val_loss)
    # -- MSE Loss
    average_val_mse_loss = total_val_mse_loss / len(mhad_val_dataloader)
    val_mse_loss_history.append(average_val_mse_loss)

    # endregion

    out = (
        f"Epoch {epoch+1}/{config.epochs}, alpha: {config.scenario4_alpha}, beta: {config.scenario4_beta}"
        + f"\nTRAIN Total Loss: {average_train_loss:.4f}, MSE Loss: {average_train_mse_loss:.4f}"
        + f"\nVAL Total Loss: {average_val_loss:.4f}, MSE Loss: {average_val_mse_loss:.4f}"
        + f"\n----------------------------------------------------\n"
    )

    print(out)

    if average_val_loss < best_val_mse_loss:
        epochs_no_improve = 0

        best_epoch = epoch

        best_pose2imu_model_state = copy.deepcopy(pose2imu_model.state_dict())

        log = out

    else:
        epochs_no_improve += 1

    if epochs_no_improve == config.patience:
        pose2imu_model.load_state_dict(best_pose2imu_model_state)
        break

    scheduler.step(total_val_loss)
# endregion

# region Save best models
# >>> Save models and metrics <<< #
prefix = (
    config.scenario4_name
    + "[s="
    + str(utils.args.seed)
    + "]"
    + "[a="
    + str(config.scenario4_alpha)
    + "]"
    + "[b="
    + str(config.scenario4_beta)
    + "]"
)

# SAVE models to file
file_name = "0_" + prefix + "(regressor)"
utils.save_model(best_pose2imu_model_state, file_name)  # saving the best model

# name = 'allacc2activity-fc(model)'
# save_model(classifier, name)

# Total Loss Plot
metric = "Total Loss"
file_name = "1_" + prefix + "(" + metric + ")"
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_loss_history,
    val_metric_history=val_loss_history,
    metric=metric,
    file_name=file_name,
)
# Save MSE Loss plot
metric = "MSE Loss"
file_name = "2_" + prefix + "(" + metric + ")"
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_mse_loss_history,
    val_metric_history=val_mse_loss_history,
    metric=metric,
    file_name=file_name,
)