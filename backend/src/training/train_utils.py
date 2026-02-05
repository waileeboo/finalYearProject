import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from typing import Optional



def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = 0.5,
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)

        if use_amp:
            # scale the loss, to make it bigger call backward()
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                # unscale the gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # update model weight but skip if gradients are inf/nan (maybe case if the scale loss to too large causing overflow)   
            scaler.step(optimizer)
            # learn from this iteration and adjust teh scale for the next time 
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate model (validation or test).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # calculat the loss without tracking gradients 
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        # calculate the total loss for the dataset depending on batch size
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


