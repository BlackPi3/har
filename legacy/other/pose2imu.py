import torch
import torch.nn as nn
import copy

from utils import config
from utils import utils
from utils.models import Regressor

if config.dataset == config.Dataset.MMFIT:
    from utils import mmfit_data as data
    window_length = config.sensor_window_length
elif config.dataset == config.Dataset.MHAD:
    import utils.mhad_data as data
    window_length = int(config.mhad_window_length * config.mhad_sampling_rate)

# >>> pose2imu models <<< #
# >>> CNN <<< #
pose2imu_model = Regressor(
    in_ch=config.in_ch,
    num_joints=config.num_joints,
    window_length=window_length,
).to(config.device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(pose2imu_model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10
)

    
def main():
    # >>> Training <<< #
    epochs = config.epochs
    train_loss_history, val_loss_history = [], []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = float('inf')
    patience = config.patience
    epochs_no_improve = 0
    log = ''

    for epoch in range(epochs):
        # - Train
        total_train_loss = 0

        pose2imu_model.train()
        for pose, acc, label in data.train_loader:
            """
            pose: (batch_size, ch, num_joints, sensor_window_length)
            acc: (batch_size, ch, sensor_window_length)
            label: (batch_size)
            """
            # -- Move to GPU
            pose = pose.to(config.device, non_blocking=True)
            acc = acc.to(config.device, non_blocking=True)
            labels = label.to(config.device, non_blocking=True)

            # -- Forward pass
            pred_acc = pose2imu_model(pose) # (batch_size, 3, sensor_window_length)

            # -- Calculate loss
            loss = criterion(pred_acc, acc)
            total_train_loss += loss.item()

            # -- Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / len(data.train_loader)
        train_loss_history.append(average_train_loss)

        # - Validation
        total_val_loss = 0

        pose2imu_model.eval()
        with torch.no_grad():
            for pose, acc, label in data.val_loader:
                # -- Move to GPU
                pose = pose.to(config.device, non_blocking=True)
                acc = acc.to(config.device, non_blocking=True)
                labels = label.to(config.device, non_blocking=True)
                # forward pass
                pred_acc = pose2imu_model(pose)

                # calculate loss
                loss = criterion(pred_acc, acc)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(data.val_loader)
        val_loss_history.append(average_val_loss)

        out = f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}" + \
                f'\n----------------------------------------------------\n'

        print(out)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_state = copy.deepcopy(pose2imu_model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0

            log = f"Seed [{utils.args.seed}]" \
                f"\nVal Best Loss: {best_val_loss:.4f}" + \
                f'\n-----------------------------------------------------------\n'
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            break

        scheduler.step(average_val_loss)

if __name__ == '__main__':
    main()
        