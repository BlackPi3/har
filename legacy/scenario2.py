import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.metrics import f1_score

import sys
sys.path.append('..')

from utils import config
from utils import utils

from legacy.utils.modules import Regressor, FeatureExtractor, ActivityClassifier

if config.dataset == config.Dataset.MMFIT:
    from datasets import mmfit_data as data
    window_length = config.sensor_window_length
elif config.dataset == config.Dataset.MHAD:
    pass
    # import datasets.mhad_dataset as data
    # window_length = int(config.mhad_window_length * config.mhad_sampling_rate)

# ------------------------------------------------------------------------------------------------------- #


# >>> Models <<< #
pose2imu_model = Regressor(
    in_ch=config.in_ch,
    num_joints=config.num_joints,
    window_length=window_length,
).to(config.device)

fe_model = FeatureExtractor(window_size=window_length).to(config.device, non_blocking=True)
ac_model = ActivityClassifier(f_in=config.ac_fin, n_classes=config.mhad_num_classes).to(
    config.device, non_blocking=True)

# >>> Loss + Optimization <<< #
MSELoss = nn.MSELoss()


def cosine_similarity_loss(output, target):
    cosine_loss = 1 - F.cosine_similarity(output, target, dim=1)
    return cosine_loss.mean()


CrossEntropyLoss = nn.CrossEntropyLoss()

params = (
    list(pose2imu_model.parameters())
    + list(fe_model.parameters())
    + list(ac_model.parameters())
)
optimizer = torch.optim.Adam(params, lr=1e-3)#config.lr)  # , weight_decay=1e-03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=50#10
)

# ------------------------------------------------------------------------------------------------------- #


def main():
    # >>> traing model <<< #
    train_loss_history, train_mse_loss_history, train_similarity_loss_history, train_activity_loss_history, train_f1_history, train_accuracy_history = [], [], [], [], [], []
    val_loss_history, val_mse_loss_history, val_similarity_loss_history, val_activity_loss_history, val_f1_history, val_accuracy_history = [], [], [], [], [], []

    best_pose2imu_model_state, best_fe_model_state, best_ac_model_state = None, None, None
    best_epoch = float('inf')
    best_val_f1 = 0
    best_val_acc = 0

    epochs = 100#config.epochs
    patience = 50#config.patience
    epochs_no_improve = 0

    log = ''

    for epoch in range(epochs):

        # - TRAIN
        total_train_loss, total_train_mse_loss, total_train_similarity_loss, total_train_activity_loss = 0, 0, 0, 0
        all_pred_labels, all_true_labels = [], []  # F1
        total_predictions, correct_predictions = 0, 0  # Accuracy

        pose2imu_model.train()
        fe_model.train()
        ac_model.train()
        for pose, acc, labels in data.train_loader:
            """
            pose: (batch_size, 3, num_joints, sensor_window_length)
            acc: (batch_size, 3, sensor_window_length)
            """
            # -- Move to GPU
            pose = pose.to(config.device, non_blocking=True)
            acc = acc.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            # -- Forward pass
            # --- Regressor
            sim_acc = pose2imu_model(pose)
            mse_loss = MSELoss(sim_acc, acc)  # LOSS
            total_train_mse_loss += mse_loss.item()
            # --- Feature Extractor
            real_acc_features = fe_model(acc)
            sim_acc_features = fe_model(sim_acc)
            similarity_loss = cosine_similarity_loss(
                sim_acc_features, real_acc_features)
            total_train_similarity_loss += similarity_loss.item()
            # --- Activity Classifier
            label_logits = ac_model(real_acc_features)
            sim_label_logits = ac_model(sim_acc_features)
            activity_loss = CrossEntropyLoss(
                label_logits, labels) + CrossEntropyLoss(sim_label_logits, labels)  # LOSS CE
            total_train_activity_loss += activity_loss.item()

            # --  Total Loss
            total_loss = mse_loss + config.scenario2_alpha * \
                activity_loss + config.scenario2_beta * similarity_loss
            total_train_loss += total_loss.item()

            # --  F1
            pred_labels = torch.argmax(label_logits, dim=1)
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

            # -- Accuracy
            total_predictions += labels.size(0)
            correct_predictions += (pred_labels == labels).sum().item()

            # -- Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # --  Total Loss
        average_train_loss = total_train_loss / len(data.train_loader)
        train_loss_history.append(average_train_loss)
        # --  MSE Loss
        average_train_mse_loss = total_train_mse_loss / len(data.train_loader)
        train_mse_loss_history.append(average_train_mse_loss)
        # --  Similarity Loss
        average_train_similarity_loss = total_train_similarity_loss / \
            len(data.train_loader)
        train_similarity_loss_history.append(average_train_similarity_loss)
        # --  Activity Loss
        average_train_activity_loss = total_train_activity_loss / \
            len(data.train_loader)
        train_activity_loss_history.append(average_train_activity_loss)

        # --  F1
        train_f1 = f1_score(all_true_labels, all_pred_labels, average="macro")
        train_f1_history.append(train_f1)

        # -- Accuracy
        train_accuracy = correct_predictions / total_predictions
        train_accuracy_history.append(train_accuracy)

        # ------------------------------------------------------------------------------------------------------- #

        # - VAL
        total_val_loss, total_val_mse_loss, total_val_similarity_loss, total_val_activity_loss = 0, 0, 0, 0
        all_pred_labels, all_true_labels = [], []
        total_predictions, correct_predictions = 0, 0

        pose2imu_model.eval()
        fe_model.eval()
        ac_model.eval()
        with torch.no_grad():
            for pose, acc, labels in data.val_loader:
                # -- Move to GPU
                pose = pose.to(config.device, non_blocking=True)
                acc = acc.to(config.device, non_blocking=True)
                labels = labels.to(config.device, non_blocking=True)

                # -- Forward pass
                # --- Regressor
                sim_acc = pose2imu_model(pose)
                mse_loss = MSELoss(sim_acc, acc)
                total_val_mse_loss += mse_loss.item()
                # --- Feature Extractor
                real_acc_features = fe_model(acc)
                sim_acc_features = fe_model(sim_acc)
                similarity_loss = cosine_similarity_loss(
                    sim_acc_features, real_acc_features)
                total_val_similarity_loss += similarity_loss.item()
                # --- Activity Classifier
                label_logits = ac_model(real_acc_features)
                sim_label_logits = ac_model(sim_acc_features)
                activity_loss = CrossEntropyLoss(
                    label_logits, labels) + CrossEntropyLoss(sim_label_logits, labels)
                total_val_activity_loss += activity_loss.item()
                # -- Total Loss
                total_loss = mse_loss + config.scenario2_alpha * \
                    activity_loss + config.scenario2_beta * similarity_loss
                total_val_loss += total_loss.item()

                # -- F1
                pred_labels = torch.argmax(label_logits, dim=1)
                all_pred_labels.extend(pred_labels.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

                # -- Accuracy
                total_predictions += labels.size(0)
                correct_predictions += (pred_labels == labels).sum().item()

        # -- Total Loss
        average_val_loss = total_val_loss / len(data.val_loader)
        val_loss_history.append(average_val_loss)
        # -- MSE Loss
        average_val_mse_loss = total_val_mse_loss / len(data.val_loader)
        val_mse_loss_history.append(average_val_mse_loss)
        # -- Similarity Loss
        average_val_similarity_loss = total_val_similarity_loss / \
            len(data.val_loader)
        val_similarity_loss_history.append(average_val_similarity_loss)
        # -- Activity Loss
        average_val_activity_loss = total_val_activity_loss / \
            len(data.val_loader)
        val_activity_loss_history.append(average_val_activity_loss)

        # -- F1
        val_f1 = f1_score(all_true_labels, all_pred_labels, average="macro")
        val_f1_history.append(val_f1)

        # -- Accuracy
        val_accuracy = correct_predictions / total_predictions
        val_accuracy_history.append(val_accuracy)

        # ------------------------------------------------------------------------------------------------------- #

        out = (f"Epoch {epoch+1}/{epochs}, alpha: {config.scenario2_alpha}, beta: {config.scenario2_beta}" +
               f"\nTRAIN Total Loss: {average_train_loss:.4f}, MSE Loss: {average_train_mse_loss:.4f}, Activity Loss * alpha: {average_train_activity_loss * config.scenario2_alpha:.4f}, Similarity Loss * beta: {average_train_similarity_loss * config.scenario2_beta:.4f}" +
               f'\nTRAIN F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}' +
               f'\nVAL Total Loss: {average_val_loss:.4f}, MSE Loss: {average_val_mse_loss:.4f}, Activity Loss * alpha: {average_val_activity_loss * config.scenario2_alpha:.4f}, Similarity Loss * beta: {average_val_similarity_loss * config.scenario2_beta:.4f}' +
               f'\nVAL F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}' +
               f'\n----------------------------------------------------\n')

        print(out)

        if best_val_f1 < val_f1:
            epochs_no_improve = 0

            best_val_f1 = val_f1
            best_val_acc = val_accuracy
            best_epoch = epoch

            best_pose2imu_model_state = copy.deepcopy(
                pose2imu_model.state_dict())
            best_fe_model_state = copy.deepcopy(fe_model.state_dict())
            best_ac_model_state = copy.deepcopy(ac_model.state_dict())

            log = out

        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            pose2imu_model.load_state_dict(best_pose2imu_model_state)
            fe_model.load_state_dict(best_fe_model_state)
            ac_model.load_state_dict(best_ac_model_state)

            break

        scheduler.step(average_val_loss)

    # ------------------------------------------------------------------------------------------------------- #
    # >>> Test <<< #
    # - Test
    total_loss = 0
    all_pred_labels, all_true_labels = [], []
    total_predictions, correct_predictions = 0, 0

    pose2imu_model.eval()
    fe_model.eval()
    ac_model.eval()
    with torch.no_grad():
        for pose, acc, labels in data.test_loader:
            # -- Move to GPU
            acc = acc.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            # -- Forward pass
            # --- Feature Extractor
            real_acc_features = fe_model(acc)
            # --- Activity Classifier
            label_logits = ac_model(real_acc_features)

            # -- F1
            pred_labels = torch.argmax(label_logits, dim=1)
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

            # -- Accuracy
            total_predictions += labels.size(0)
            correct_predictions += (pred_labels == labels).sum().item()

        # -- F1
        f1 = f1_score(all_true_labels, all_pred_labels, average="macro")

        # -- Accuracy
        accuracy = correct_predictions / total_predictions

        log += f"Test F1: {f1:.4f}, Test Accuracy: {accuracy:.4f}" + \
            f'\n----------------------------------------------------\n'
        
        print(log)

    # ------------------------------------------------------------------------------------------------------- #

    # >>> Save models and metrics <<< #
    prefix = config.scenario2_name + "[s=" + str(utils.args.seed) + "]" + '[a=' + str(
        config.scenario2_alpha) + ']' + '[b=' + str(config.scenario2_beta) + ']'
    metric_suffix = '[MSE+Similarity+Activity]' if config.scenario2_beta != 0 else '[MSE+Activity]'

    # SAVE models to file
    file_name = '0_' + prefix + "(regression)"
    utils.save_model(best_pose2imu_model_state,
                     file_name)  # saving the best model

    # name = 'allacc2activity-fc(model)'
    # save_model(classifier, name)

    # Total Loss Plot
    metric = 'Total Loss' + metric_suffix
    file_name = '1_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_loss_history,
        val_metric_history=val_loss_history,
        metric=metric,
        file_name=file_name,
    )
    # Save MSE Loss plot
    metric = 'MSE Loss'
    file_name = '2_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_mse_loss_history,
        val_metric_history=val_mse_loss_history,
        metric=metric,
        file_name=file_name,
    )
    # Save Similarity Loss plot
    metric = 'Similarity Loss'
    file_name = '3_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_similarity_loss_history,
        val_metric_history=val_similarity_loss_history,
        metric=metric,
        file_name=file_name,
    )
    # Save Activity Loss plot
    metric = 'Activity Loss'
    file_name = '4_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_activity_loss_history,
        val_metric_history=val_activity_loss_history,
        metric=metric,
        file_name=file_name,
    )

    # Save F1 score plot
    metric = 'F1' + metric_suffix
    file_name = '5_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_f1_history,
        val_metric_history=val_f1_history,
        metric=metric,
        file_name=file_name,
    )

    # Save Accuracy
    metric = 'Accuracy' + metric_suffix
    file_name = '6_' + prefix + '(' + metric + ')'
    utils.save_plot(
        epochs=epoch,
        best_epoch=best_epoch,
        train_metric_history=train_accuracy_history,
        val_metric_history=val_accuracy_history,
        metric=metric,
        file_name=file_name,
    )

    # Save log
    log += prefix + '\n' + metric_suffix
    file_name = '7_' + prefix + '(Log)'
    utils.save_log(log=log, file_name=file_name)


if __name__ == '__main__':
    main()