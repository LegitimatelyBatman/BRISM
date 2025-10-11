"""
Training functions for BRISM model with alternating batch training.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
from .model import BRISM
from .loss import BRISMLoss


def train_brism(
    model: BRISM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BRISMLoss,
    num_epochs: int,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 100,
    callback: Optional[Callable[[int, int, Dict], None]] = None
) -> Dict[str, list]:
    """
    Train BRISM model with alternating batches.
    
    Training alternates between:
    - Forward path: symptoms -> ICD
    - Reverse path: ICD -> symptoms
    - Forward cycle: symptoms -> ICD -> symptoms
    - Reverse cycle: ICD -> symptoms -> ICD
    
    Args:
        model: BRISM model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
        val_loader: Optional validation data loader
        scheduler: Optional learning rate scheduler
        log_interval: Steps between logging
        callback: Optional callback function(epoch, step, metrics)
        
    Returns:
        Dictionary of training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_metrics': []
    }
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_codes'].to(device)
            
            optimizer.zero_grad()
            
            # Alternate training direction every batch
            # Cycle: forward, reverse, forward_cycle, reverse_cycle
            direction = batch_idx % 4
            
            if direction == 0:
                # Forward path: symptoms -> ICD
                output = model.forward_path(symptoms)
                loss, loss_dict = loss_fn.forward_loss(output, symptoms, icd_codes)
                
            elif direction == 1:
                # Reverse path: ICD -> symptoms
                output = model.reverse_path(icd_codes, symptoms)
                loss, loss_dict = loss_fn.reverse_loss(output, icd_codes, symptoms)
                
            elif direction == 2:
                # Forward cycle: symptoms -> ICD -> symptoms
                output = model.cycle_forward(symptoms, symptoms)
                loss, loss_dict = loss_fn.cycle_forward_loss(output, symptoms, icd_codes)
                
            else:  # direction == 3
                # Reverse cycle: ICD -> symptoms -> ICD
                output = model.cycle_reverse(icd_codes, symptoms)
                loss, loss_dict = loss_fn.cycle_reverse_loss(output, icd_codes, symptoms)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            for key, val in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(val)
            
            # Logging
            if global_step % log_interval == 0:
                avg_loss = sum(epoch_losses[-log_interval:]) / min(log_interval, len(epoch_losses))
                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {avg_loss:.4f}")
                
                if callback:
                    callback(epoch, global_step, {'loss': avg_loss, **loss_dict})
            
            global_step += 1
        
        # Average epoch metrics
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history['train_loss'].append(avg_epoch_loss)
        
        epoch_avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        history['epoch_metrics'].append(epoch_avg_metrics)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        for key, val in epoch_avg_metrics.items():
            print(f"  {key}: {val:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate_brism(model, val_loader, loss_fn, device)
            history['val_loss'].append(val_loss)
            print(f"  Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        print()
    
    return history


def evaluate_brism(
    model: BRISM,
    data_loader: DataLoader,
    loss_fn: BRISMLoss,
    device: torch.device
) -> float:
    """
    Evaluate BRISM model on validation/test set.
    
    Args:
        model: BRISM model
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            symptoms = batch['symptoms'].to(device)
            icd_codes = batch['icd_codes'].to(device)
            
            # Evaluate all directions
            # Forward
            output = model.forward_path(symptoms)
            loss1, _ = loss_fn.forward_loss(output, symptoms, icd_codes)
            
            # Reverse
            output = model.reverse_path(icd_codes, symptoms)
            loss2, _ = loss_fn.reverse_loss(output, icd_codes, symptoms)
            
            # Average
            loss = (loss1 + loss2) / 2
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_epoch_simple(
    model: BRISM,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BRISMLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Simple single epoch training function (for testing/examples).
    
    Args:
        model: BRISM model
        data_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device
        epoch: Current epoch number
        
    Returns:
        Dictionary of average metrics
    """
    model.train()
    model.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(data_loader):
        symptoms = batch['symptoms'].to(device)
        icd_codes = batch['icd_codes'].to(device)
        
        optimizer.zero_grad()
        
        # Alternate between directions
        if batch_idx % 2 == 0:
            output = model.forward_path(symptoms)
            loss, _ = loss_fn.forward_loss(output, symptoms, icd_codes)
        else:
            output = model.reverse_path(icd_codes, symptoms)
            loss, _ = loss_fn.reverse_loss(output, icd_codes, symptoms)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return {'loss': avg_loss}
