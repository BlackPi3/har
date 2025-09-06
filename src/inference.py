"""
Inference utilities for model evaluation and prediction.
"""
import torch
from torch.utils.data import DataLoader
from datasets.common import SequentialStridedSampler


def unfold_predictions(model, dataset, pred_acc, batch_size=128, window_stride=20, 
                      device='cpu', sensor_window_length=300):
    """
    Apply a trained model to a dataset and unfold predictions back to original sequence length.
    
    This function handles overlapping windows by using a sequential strided sampler and
    carefully managing which parts of the prediction tensor get written to avoid conflicts.
    
    Args:
        model: Trained PyTorch model for inference
        dataset: Dataset to run inference on
        pred_acc: Pre-allocated tensor to store predictions (shape: [channels, sequence_length])
        batch_size: Batch size for inference
        window_stride: Stride between consecutive windows  
        device: Device to run inference on ('cpu', 'cuda', etc.)
        sensor_window_length: Length of each window in samples
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialStridedSampler(dataset, window_stride),
    )

    # Track which positions have been written to avoid overwriting
    written = torch.zeros(pred_acc.shape[1], dtype=torch.bool).to(device=device)

    global_start = 0  # Track global position across batches
    model.eval()
    
    with torch.no_grad():
        for pose, _, _ in dataloader:
            pose = pose.to(device, non_blocking=True)
            batch_pred_acc = model(pose)  # (batch_size, channels, window_length)

            for j in range(batch_pred_acc.shape[0]):  # Process each sample in batch
                start = global_start + j * window_stride
                end = start + sensor_window_length
                
                # Only write to positions that haven't been written yet
                mask = written[start:end] == 0
                pred_acc[:, start:end][:, mask] = batch_pred_acc[j, :, mask]
                written[start:end][mask] = True

            # Update global position for next batch
            global_start += batch_pred_acc.shape[0] * window_stride
