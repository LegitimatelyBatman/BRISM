"""
Training functions for BRISM model with alternating batch training.
"""

import torch
import os
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
from pathlib import Path
from .model import BRISM
from .loss import BRISMLoss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        else:
            if self.verbose and val_loss < self.best_loss:
                print(f"Validation loss improved: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_freq: int = 1,
                 verbose: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_loss' or 'train_loss')
            mode: 'min' or 'max' - whether lower or higher is better
            save_best_only: Only save when monitored metric improves
            save_freq: Save every N epochs (if not save_best_only)
            verbose: Whether to print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def _is_improvement(self, metric: float) -> bool:
        """Check if metric has improved."""
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric
    
    def save_checkpoint(self, 
                       model: BRISM,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': model.config
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            if self.verbose:
                print(f"Saved best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
            if self.verbose:
                print(f"Saved checkpoint to {path}")
        
        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
    
    def __call__(self, 
                 model: BRISM,
                 optimizer: torch.optim.Optimizer,
                 epoch: int,
                 metrics: Dict[str, float],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Check if checkpoint should be saved and save if needed.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
        """
        metric_value = metrics.get(self.monitor)
        
        if metric_value is None:
            if self.verbose:
                print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return
        
        # Check if this is the best model
        is_best = self._is_improvement(metric_value)
        
        if is_best:
            self.best_metric = metric_value
            self.save_checkpoint(model, optimizer, epoch, metrics, scheduler, is_best=True)
        
        # Save periodic checkpoint
        if not self.save_best_only and (epoch % self.save_freq == 0):
            self.save_checkpoint(model, optimizer, epoch, metrics, scheduler, is_best=False)


def load_checkpoint(checkpoint_path: str, 
                   model: BRISM,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: Optional[torch.device] = None) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


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
    callback: Optional[Callable[[int, int, Dict], None]] = None,
    checkpoint_dir: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    save_best_only: bool = True
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
        checkpoint_dir: Optional directory to save checkpoints
        early_stopping_patience: Optional patience for early stopping (None = no early stopping)
        save_best_only: Only save best model based on validation loss
        
    Returns:
        Dictionary of training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_metrics': []
    }
    
    # Initialize checkpoint and early stopping
    checkpointer = None
    early_stopping = None
    
    if checkpoint_dir is not None:
        checkpointer = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor='val_loss' if val_loader is not None else 'train_loss',
            mode='min',
            save_best_only=save_best_only,
            verbose=True
        )
    
    if early_stopping_patience is not None and val_loader is not None:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True
        )
    
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
            
            # Check early stopping
            if early_stopping is not None:
                if early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save checkpoint
        if checkpointer is not None:
            checkpoint_metrics = {
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss if val_loader is not None else avg_epoch_loss,
                **epoch_avg_metrics
            }
            checkpointer(model, optimizer, epoch+1, checkpoint_metrics, scheduler)
        
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
