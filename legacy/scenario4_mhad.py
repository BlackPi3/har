# region Imports
import copy
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import config
from utils import utils
from datasets.mhad_dataloader import mhad_dataloader
from legacy.utils.modules import Regressor, FeatureExtractor, ActivityClassifier
import sys

sys.path.append("..")
# endregion

# region MHAD dataset
mhad_train_dataloader, mhad_val_dataloader, mhad_test_dataloader = mhad_dataloader()
# endregion

# region Models
window = int(config.mhad_window_sec * config.mhad_accel_sampling_rate)
pose2imu_model = Regressor(
    in_ch=config.in_ch, num_joints=config.num_joints, window_length=window
).to(config.device, config.dtype)

fe_model = FeatureExtractor(window_size=window).to(config.device, config.dtype)
ac_model = ActivityClassifier(
    f_in=config.ac_fin, n_classes=len(config.mhad_actions_subset)
).to(config.device, config.dtype)

MSELoss = nn.MSELoss()


def smoothness_loss(pred):
    diff = torch.diff(pred, dim=2)  # Compute differences along the temporal axis
    loss = torch.mean(
        diff**2
    )  # Penalize large differences (encourages smooth transitions)
    return loss


def cosine_similarity_loss(output, target):
    cosine_loss = 1 - F.cosine_similarity(output, target, dim=1)
    return cosine_loss.mean()


CrossEntropyLoss = nn.CrossEntropyLoss()

params = (
    list(pose2imu_model.parameters())
    + list(fe_model.parameters())
    + list(ac_model.parameters())
)
optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=1e-03)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.1, patience=10
# )
# endregion

# region Training + Validation
# region init
(
    train_mse_loss_history,
    train_similarity_loss_history,
    train_activity_loss_history,
    train_f1_history,
    train_accuracy_history,
    train_aggregate_loss_history,
) = [], [], [], [], [], []
(
    val_mse_loss_history,
    val_similarity_loss_history,
    val_activity_loss_history,
    val_f1_history,
    val_accuracy_history,
    val_aggregate_loss_history,
) = [], [], [], [], [], []

(
    best_pose2imu_model_state,
    best_fe_model_state,
    best_ac_model_state,
) = None, None, None
best_val_f1 = 0
best_epoch = float("inf")

epochs_no_improve = 0
log = ""
# endregion

for epoch in range(config.epochs):
    # region Training
    # Initialize tracking variables
    (
        total_mse_loss,
        total_similarity_loss,
        total_activity_loss,
        aggregate_loss,
    ) = 0, 0, 0, 0
    all_pred_labels, all_true_labels = [], []
    total_predictions, correct_predictions = 0, 0

    # Set models to training mode
    pose2imu_model.train()  # TEST with not updating this module.
    fe_model.train()
    ac_model.train()

    for accel, skel, label in mhad_train_dataloader:
        # move to GPU
        accel = accel.to(config.device)
        skel = skel.to(config.device)
        label = label.to(config.device)

        # Regressor
        sim_accel = pose2imu_model(skel)
        mse_loss = MSELoss(sim_accel, accel)
        total_mse_loss += mse_loss.item()

        # Feature Extractor
        accel_f = fe_model(accel)
        sim_accel_f = fe_model(sim_accel)
        similarity_loss = cosine_similarity_loss(sim_accel_f, accel_f)
        total_similarity_loss += similarity_loss.item()

        # Activity Classifier
        logits = ac_model(accel_f)
        sim_logits = ac_model(sim_accel_f)
        activity_loss = CrossEntropyLoss(logits, label) + CrossEntropyLoss(
            sim_logits, label
        )
        total_activity_loss += activity_loss.item()

        # Aggregate loss
        loss = (
            mse_loss
            + smoothness_loss(sim_accel)
            + config.scenario4_beta * similarity_loss
            + config.scenario4_alpha * activity_loss
        )
        aggregate_loss += loss.item()

        # -- Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute F1 and Accuracy on the batch
        # Only consider real logits for metrics
        pred_labels = torch.argmax(logits, dim=1)
        all_pred_labels.extend(pred_labels.cpu().numpy())
        all_true_labels.extend(label.cpu().numpy())

        correct_predictions += (pred_labels == label).sum().item()
        total_predictions += label.size(0)

    # Average losses
    # MSE
    avg = total_mse_loss / len(mhad_train_dataloader)
    train_mse_loss_history.append(avg)
    # Similarity
    avg = total_similarity_loss / len(mhad_train_dataloader)
    train_similarity_loss_history.append(avg)
    # Activity
    avg = total_activity_loss / len(mhad_train_dataloader)
    train_activity_loss_history.append(avg)
    # Aggregate
    avg = aggregate_loss / len(mhad_train_dataloader)
    train_aggregate_loss_history.append(avg)

    # F1 Score
    f1 = f1_score(all_true_labels, all_pred_labels, average="macro")
    train_f1_history.append(f1)
    # Accuracy
    accuracy = correct_predictions / total_predictions
    train_accuracy_history.append(accuracy)

    # endregion

    # region Validation
    (
        total_mse_loss,
        total_similarity_loss,
        total_activity_loss,
        aggregate_loss,
    ) = 0, 0, 0, 0
    all_pred_labels, all_true_labels = [], []
    total_predictions, correct_predictions = 0, 0

    pose2imu_model.eval()
    fe_model.eval()
    ac_model.eval()
    with torch.no_grad():
        for accel, skel, label in mhad_val_dataloader:
            # move to GPU
            accel = accel.to(config.device)
            skel = skel.to(config.device)
            label = label.to(config.device)

            # Regressor
            sim_accel = pose2imu_model(skel)
            mse_loss = MSELoss(sim_accel, accel)
            total_mse_loss += mse_loss.item()

            # Feature Extractor
            accel_f = fe_model(accel)
            sim_accel_f = fe_model(sim_accel)
            similarity_loss = cosine_similarity_loss(sim_accel_f, accel_f)
            total_similarity_loss += similarity_loss.item()

            # Activity Classifier
            logits = ac_model(accel_f)
            sim_logits = ac_model(sim_accel_f)
            activity_loss = CrossEntropyLoss(logits, label) + CrossEntropyLoss(
                sim_logits, label
            )
            total_activity_loss += activity_loss.item()

            # Aggregate loss
            loss = (
                mse_loss
                + smoothness_loss(sim_accel)
                + config.scenario4_beta * similarity_loss
                + config.scenario4_alpha * activity_loss
            )
            aggregate_loss += loss.item()

            # Compute F1 and Accuracy on the batch
            # Only consider real logits for metrics
            pred_labels = torch.argmax(logits, dim=1)
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_true_labels.extend(label.cpu().numpy())

            correct_predictions += (pred_labels == label).sum().item()
            total_predictions += label.size(0)

    # Average losses
    # -- MSE Loss
    avg = total_mse_loss / len(mhad_val_dataloader)
    val_mse_loss_history.append(avg)
    # -- Similarity Loss
    avg = total_similarity_loss / len(mhad_val_dataloader)
    val_similarity_loss_history.append(avg)
    # -- Activity Loss
    avg = total_activity_loss / len(mhad_val_dataloader)
    val_activity_loss_history.append(avg)
    # -- Aggregate Loss
    avg = aggregate_loss / len(mhad_val_dataloader)
    val_aggregate_loss_history.append(avg)
    # -- F1
    f1 = f1_score(all_true_labels, all_pred_labels, average="macro")
    val_f1_history.append(f1)
    # -- Accuracy
    accuracy = correct_predictions / total_predictions
    val_accuracy_history.append(accuracy)

    # endregion

    # region post training
    out = (
        f"Epoch {epoch+1}/{config.epochs}, alpha: {config.scenario4_alpha}, beta: {config.scenario4_beta}"
        + "\nTRAIN stats:"
        + f"\nAggregate Loss: {train_aggregate_loss_history[-1]:.4f}, MSE Loss: {train_mse_loss_history[-1]:.4f}"
        + f"\nbeta*Similarity Loss: {train_similarity_loss_history[-1] * config.scenario4_beta:.4f}, alpha*Activity Loss: {train_activity_loss_history[-1] * config.scenario4_alpha:.4f}"
        + f"\nF1: {train_f1_history[-1]:.4f}, Accuracy: {train_accuracy_history[-1]:.4f}"
        + "\nVAL stats:"
        + f"\nAggregate Loss: {val_aggregate_loss_history[-1]:.4f}, MSE Loss: {val_mse_loss_history[-1]:.4f}"
        + f"\nbeta*Similarity Loss: {val_similarity_loss_history[-1] * config.scenario4_beta:.4f}, alpha*Activity Loss: {val_activity_loss_history[-1] * config.scenario4_alpha:.4f}"
        + f"\nF1: {val_f1_history[-1]:.4f}, Accuracy: {val_accuracy_history[-1]:.4f}"
        + f"\n----------------------------------------------------\n"
    )

    print(out)

    if best_val_f1 < val_f1_history[-1]:
        epochs_no_improve = 0

        best_val_f1 = val_f1_history[-1]
        best_val_acc = val_accuracy_history[-1]
        best_epoch = epoch

        best_pose2imu_model_state = copy.deepcopy(pose2imu_model.state_dict())
        best_fe_model_state = copy.deepcopy(fe_model.state_dict())
        best_ac_model_state = copy.deepcopy(ac_model.state_dict())

        log = out

    else:
        epochs_no_improve += 1

    if epochs_no_improve == config.patience:
        pose2imu_model.load_state_dict(best_pose2imu_model_state)
        fe_model.load_state_dict(best_fe_model_state)
        ac_model.load_state_dict(best_ac_model_state)

        break
    # endregion

# endregion

# region Save best models

prefix = (
    'mhad'
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

# region models
file_name = prefix + "_model_" + "reg"
utils.save_model(best_pose2imu_model_state, file_name)  # saving the best model

file_name = prefix + "_model_" + "fe"
utils.save_model(best_fe_model_state, file_name)  # saving the best model

file_name = prefix + "_model_" + "ac"
utils.save_model(best_ac_model_state, file_name)  # saving the best model
# endregion

# region plots

# Total Loss Plot
# metric = "Total Loss"
# file_name = prefix + "_plot_" + metric
# utils.save_plot(
#     epochs=epoch,
#     best_epoch=best_epoch,
#     train_metric_history=train_aggregate_loss_history,
#     val_metric_history=val_aggregate_loss_history,
#     metric=metric,
#     file_name=file_name,
# )
# Save MSE Loss plot
# metric = "MSE Loss"
# file_name = prefix + "_plot_" + metric
# utils.save_plot(
#     epochs=epoch,
#     best_epoch=best_epoch,
#     train_metric_history=train_mse_loss_history,
#     val_metric_history=val_mse_loss_history,
#     metric=metric,
#     file_name=file_name,
# )
# Save Similarity Loss plot
metric = "Similarity Loss"
file_name = prefix + "_plot_" + metric
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_similarity_loss_history,
    val_metric_history=val_similarity_loss_history,
    metric=metric,
    file_name=file_name,
)
# Save Activity Loss plot
# metric = "Activity Loss"
# file_name = prefix + "_plot_" + metric
# utils.save_plot(
#     epochs=epoch,
#     best_epoch=best_epoch,
#     train_metric_history=train_activity_loss_history,
#     val_metric_history=val_activity_loss_history,
#     metric=metric,
#     file_name=file_name,
# )

# Save F1 score plot
metric = "F1"
file_name = prefix + "_plot_" + metric
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_f1_history,
    val_metric_history=val_f1_history,
    metric=metric,
    file_name=file_name,
)

# Save Accuracy plot
# metric = "Accuracy"
# file_name = prefix + "_plot_" + metric
# utils.save_plot(
#     epochs=epoch,
#     best_epoch=best_epoch,
#     train_metric_history=train_accuracy_history,
#     val_metric_history=val_accuracy_history,
#     metric=metric,
#     file_name=file_name,
# )
# endregion

# endregion