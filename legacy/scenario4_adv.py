# region Imports
import copy
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import config
from utils import utils
from datasets.adv_dataloader import adv_dataloader
from legacy.utils.modules import Regressor, FeatureExtractor, ActivityClassifier, Discriminator
import sys
import os
import torch.autograd as autograd

sys.path.append("..")
# endregion

# region MHAD dataset
adv_train_dataloader, adv_val_dataloader, adv_test_dataloader = adv_dataloader()
# endregion

# region Models
window = int(config.mhad_window_sec * config.mhad_accel_sampling_rate)

pose2imu_model = Regressor(
    in_ch=config.in_ch, num_joints=config.num_joints, window_length=window
).to(config.device, config.dtype)
model = "11.09/02.23_mhad[s=0][a=0.1][b=10]_model_reg.pth"
model_path = os.path.join(config.train_out_dir, model)
pose2imu_model.load_state_dict(torch.load(model_path, map_location=config.device))

fe_model = FeatureExtractor(window_size=window).to(config.device, config.dtype)
model = "11.09/02.23_mhad[s=0][a=0.1][b=10]_model_fe.pth"
model_path = os.path.join(config.train_out_dir, model)
fe_model.load_state_dict(torch.load(model_path, map_location=config.device))

ac_model = ActivityClassifier(
    f_in=config.ac_fin, n_classes=len(config.mhad_actions_subset)
).to(config.device, config.dtype)
# model = "11.09/02.23_mhad[s=0][a=0.1][b=10]_model_reg.pth"
# model_path = os.path.join(config.train_out_dir, model)
# ac_model.load_state_dict(torch.load(model_path, map_location=config.device))

adv_model = Discriminator(3 * window).to(config.device, config.dtype)
# adv_model = Discriminator(config.ac_fin).to(config.device, config.dtype)


def compute_gradient_penalty(critic, real_data, fake_data):
    """Computes the gradient penalty for WGAN-GP on 3D data (e.g., time-series or sequences)."""
    batch_size = real_data.size(0)  # Get the batch size
    device = real_data.device  # Get the device

    # Interpolate between real and fake data
    epsilon = torch.rand(batch_size, *([1] * (real_data.dim() - 1)), device=device)
    epsilon = epsilon.expand_as(real_data)
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates.requires_grad_(True)

    # Pass interpolates through the critic
    critic_interpolates = critic(interpolates)

    # Compute gradients of critic scores with respect to the interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Reshape gradients to [batch_size, -1] and compute the L2 norm
    gradients = gradients.view(batch_size, -1)  # Flatten gradients to [batch_size, num_elements]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

CrossEntropyLoss = nn.CrossEntropyLoss()

BCELoss = nn.BCELoss()

params = (
    list(pose2imu_model.parameters())
    + list(fe_model.parameters())
    + list(ac_model.parameters())
    # + list(adv_model.parameters())
)
optimizer = torch.optim.Adam(params, lr=1e-4)
adv_optimizer = torch.optim.Adam(adv_model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.2, patience=10
# )
# endregion

# region Training + Validation
# region init
(
    train_activity_loss_history,
    train_adv_gen_loss_history,
    train_adv_crit_loss_history,
    train_f1_history,
    train_accuracy_history,
    train_aggregate_loss_history,
) = [], [], [], [], [], []
(
    val_activity_loss_history,
    val_adv_gen_loss_history,
    val_adv_crit_loss_history,
    val_f1_history,
    val_accuracy_history,
    val_aggregate_loss_history,
) = [], [], [], [], [], []

(
    best_pose2imu_model_state,
    best_fe_model_state,
    best_ac_model_state,
    best_adv_model_state,
) = None, None, None, None
best_val_f1 = 0
best_epoch = float("inf")

epochs_no_improve = 0
log = ""
# endregion

for epoch in range(config.epochs):
    # region Training
    # Initialize tracking variables
    (
        total_activity_loss,
        total_adv_gen_loss,
        total_adv_crit_loss,
        aggregate_loss,
    ) = 0, 0, 0, 0#, 0, 0
    all_pred_labels, all_true_labels = [], []
    total_predictions, correct_predictions = 0, 0

    # Set models to training mode
    pose2imu_model.train()
    # pose2imu_model.eval()
    fe_model.train
    ac_model.train()
    adv_model.train()

    n_crit_update = 2
    for accel, skel, label in adv_train_dataloader:
        # move to GPU
        accel = accel.to(config.device)
        skel = skel.to(config.device)
        label = label.to(config.device)

        # with torch.no_grad():
        # Regressor
        sim_accel = pose2imu_model(skel)

        # Feature Extractor
        accel_f = fe_model(accel)
        sim_accel_f = fe_model(sim_accel)

        # Adversarial Critic
        for _ in range(n_crit_update):
            real_pred = adv_model(accel)
            fake_pred = adv_model(sim_accel.detach())
            real_pred_mean = real_pred.mean(dim=1)
            fake_pred_mean = fake_pred.mean(dim=1)
            critic_loss = fake_pred_mean.mean() - real_pred_mean.mean() # # D(fake) - D(real)
            gradient_penalty = compute_gradient_penalty(adv_model, accel, sim_accel.detach())
            critic_total_loss = critic_loss + config.scenario4_grad_pen * gradient_penalty
            total_adv_crit_loss += critic_total_loss


            adv_optimizer.zero_grad()
            critic_total_loss.backward()  # Backprop for the critic
            adv_optimizer.step()
        
        # Adversarial Gen
        fake_pred_gen = adv_model(sim_accel)
        adv_loss_gen = -fake_pred_gen.mean()
        total_adv_gen_loss += adv_loss_gen.item()

        # Activity Classifier
        logits = ac_model(accel_f)
        sim_logits = ac_model(sim_accel_f)
        activity_loss = CrossEntropyLoss(logits, label) + CrossEntropyLoss(
            sim_logits, label
        )
        total_activity_loss += activity_loss.item()

        # Aggregate loss
        loss = (
            + config.scenario4_alpha * activity_loss
            + config.scenario4_lambda * adv_loss_gen
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
    avg = total_activity_loss / len(adv_train_dataloader)
    train_activity_loss_history.append(avg)
    # Adv Gen
    avg = total_adv_gen_loss / len(adv_train_dataloader)
    train_adv_gen_loss_history.append(avg)
    # Adv Crit
    avg = total_adv_crit_loss / len(adv_train_dataloader)
    train_adv_crit_loss_history.append(avg)
    # aggregate
    avg = aggregate_loss / len(adv_train_dataloader)
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
        total_adv_gen_loss,
        aggregate_loss,
    ) = 0, 0, 0, 0, 0
    all_pred_labels, all_true_labels = [], []
    total_predictions, correct_predictions = 0, 0

    pose2imu_model.eval()
    fe_model.eval()
    ac_model.eval()
    adv_model.eval()
    with torch.no_grad():
        for accel, skel, label in adv_val_dataloader:
            # move to GPU
            accel = accel.to(config.device)
            skel = skel.to(config.device)
            label = label.to(config.device)

            # Regressor
            sim_accel = pose2imu_model(skel)

            # Feature Extractor
            accel_f = fe_model(accel)
            sim_accel_f = fe_model(sim_accel)

            # Adversarial Gen
            fake_pred_gen = adv_model(sim_accel)
            adv_loss_gen = -fake_pred_gen.mean()
            total_adv_gen_loss += adv_loss_gen.item()

            # Activity Classifier
            logits = ac_model(accel_f)
            sim_logits = ac_model(sim_accel_f)
            activity_loss = CrossEntropyLoss(logits, label) + CrossEntropyLoss(
                sim_logits, label
            )
            total_activity_loss += activity_loss.item()

            # Aggregate loss
            loss = (
                + config.scenario4_alpha * activity_loss
                + config.scenario4_lambda * adv_loss_gen
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
    # -- Activity Loss
    avg = total_activity_loss / len(adv_val_dataloader)
    val_activity_loss_history.append(avg)
    # -- Adv Loss
    avg = total_adv_gen_loss / len(adv_val_dataloader)
    val_adv_gen_loss_history.append(avg)
    # -- Aggregate Loss
    avg = aggregate_loss / len(adv_val_dataloader)
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
        f"Epoch {epoch+1}/{config.epochs}, alpha: {config.scenario4_alpha}, lambda: {config.scenario4_lambda}"
        + "\nTRAIN stats:"
        + f"\nAggregate Loss: {train_aggregate_loss_history[-1]:.4f}"
        + f"\nAdv Gen Loss: {train_adv_gen_loss_history[-1]:.4f}, Adv Crit Loss: {train_adv_crit_loss_history[-1]:.4f}"
        + f"\nalpha*Activity Loss: {train_activity_loss_history[-1] * config.scenario4_alpha:.4f}"
        + f"\nF1: {train_f1_history[-1]:.4f}, Accuracy: {train_accuracy_history[-1]:.4f}"
        + "\nVAL stats:"
        + f"\nAggregate Loss: {val_aggregate_loss_history[-1]:.4f}"
         + f"\nAdv Gen Loss: {val_adv_gen_loss_history[-1]:.4f}"#, Adv Crit Loss: {val_adv_crit_loss_history[-1]:.4f}"
        + f"\nActivity Loss: {val_activity_loss_history[-1] * config.scenario4_alpha:.4f}"
        + f"\nF1: {val_f1_history[-1]:.4f}, Accuracy: {val_accuracy_history[-1]:.4f}"
        + f"\n----------------------------------------------------\n"
    )

    print(out)

    # if best_val_f1 < val_f1_history[-1]:
    #     epochs_no_improve = 0

    #     best_val_f1 = val_f1_history[-1]
    #     best_val_acc = val_accuracy_history[-1]
    #     best_epoch = epoch

    #     best_pose2imu_model_state = copy.deepcopy(pose2imu_model.state_dict())
    #     best_fe_model_state = copy.deepcopy(fe_model.state_dict())
    #     best_ac_model_state = copy.deepcopy(ac_model.state_dict())

    #     log = out

    # else:
    #     epochs_no_improve += 1

    # if epochs_no_improve == config.patience:
    #     pose2imu_model.load_state_dict(best_pose2imu_model_state)
    #     fe_model.load_state_dict(best_fe_model_state)
    #     ac_model.load_state_dict(best_ac_model_state)

    #     break

    # scheduler.step(val_f1_history[-1])
    # endregion

# endregion

# region Save best models
prefix = (
    'adv'
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

# # SAVE models to file
# # file_name = prefix + '_model_' + "reg"
# # utils.save_model(best_pose2imu_model_state, file_name)  # saving the best model

# # file_name = prefix + '_model_' + "fe"
# # utils.save_model(best_fe_model_state, file_name)  # saving the best model

# # file_name = prefix + '_model_' + "ac"
# # utils.save_model(best_ac_model_state, file_name)  # saving the best model

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
# # Save MSE Loss plot
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
# # Save Similarity Loss plot
# metric = "Similarity Loss"
# file_name = prefix + "_plot_" + metric
# utils.save_plot(
#     epochs=epoch,
#     best_epoch=best_epoch,
#     train_metric_history=train_similarity_loss_history,
#     val_metric_history=val_similarity_loss_history,
#     metric=metric,
#     file_name=file_name,
# )
# Save Activity Loss plot
metric = "Activity Loss"
file_name = prefix + "_plot_" + metric
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_activity_loss_history,
    val_metric_history=val_activity_loss_history,
    metric=metric,
    file_name=file_name,
)
# Save adv loss plot
metric = "Adv Loss"
file_name = prefix + "_plot_" + metric
utils.save_plot(
    epochs=epoch,
    best_epoch=best_epoch,
    train_metric_history=train_adv_gen_loss_history,
    val_metric_history=val_adv_gen_loss_history,
    metric=metric,
    file_name=file_name,
)

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

# # Save Accuracy plot
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