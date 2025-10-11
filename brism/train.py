"""
Training functions for BRISM model with alternating batch training.
"""

import torch
import os
import logging
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
from pathlib import Path
from .model import BRISM
from .loss import BRISMLoss

# Configure logger
logger = logging.getLogger(__name__)


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
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
        else:
            if self.verbose and val_loss < self.best_loss:
                logger.info(f"Validation loss improved: {self.best_loss:.4f} -> {val_loss:.4f}")
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
                logger.info(f"Saved best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
            if self.verbose:
                logger.debug(f"Saved checkpoint to {path}")
        
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
                logger.warning(f"Monitored metric '{self.monitor}' not found in metrics")
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
                   device: Optional[torch.device] = None,
                   weights_only: bool = False,
                   strict: bool = True) -> Dict:
    """
    Load model checkpoint with validation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        weights_only: If True, only load model weights (ignore optimizer/scheduler)
        strict: If True, strictly enforce that keys match when loading state dict
        
    Returns:
        Dictionary with checkpoint information
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is incompatible or missing required keys
    """
    # Verify file exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
    
    # Detect checkpoint's original device and warn if it differs from requested device
    if device is not None and 'model_state_dict' in checkpoint:
        # Get device of first tensor in checkpoint to detect original device
        first_key = next(iter(checkpoint['model_state_dict'].keys()))
        checkpoint_device = checkpoint['model_state_dict'][first_key].device
        
        if checkpoint_device != device:
            logger.warning(
                f"Checkpoint was saved on {checkpoint_device} but loading to {device}. "
                f"Tensors will be moved to {device} using map_location."
            )
    
    # Validate required keys
    required_keys = ['model_state_dict', 'epoch']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}. Found keys: {list(checkpoint.keys())}")
    
    # Version check (if version info is available)
    if 'version' in checkpoint:
        checkpoint_version = checkpoint['version']
        # Simple version warning (can be extended)
        logger.info(f"Loading checkpoint from version: {checkpoint_version}")
        # Could add compatibility checks here
    
    # Validate model architecture compatibility
    model_state = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Check vocab sizes if config is saved
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if hasattr(model, 'config'):
            # Handle both dict and BRISMConfig object
            if hasattr(saved_config, 'symptom_vocab_size'):
                # It's a BRISMConfig object
                if saved_config.symptom_vocab_size != model.config.symptom_vocab_size:
                    raise ValueError(
                        f"Checkpoint symptom_vocab_size ({saved_config.symptom_vocab_size}) "
                        f"doesn't match model ({model.config.symptom_vocab_size})"
                    )
                if saved_config.icd_vocab_size != model.config.icd_vocab_size:
                    raise ValueError(
                        f"Checkpoint icd_vocab_size ({saved_config.icd_vocab_size}) "
                        f"doesn't match model ({model.config.icd_vocab_size})"
                    )
                if saved_config.latent_dim != model.config.latent_dim:
                    raise ValueError(
                        f"Checkpoint latent_dim ({saved_config.latent_dim}) "
                        f"doesn't match model ({model.config.latent_dim})"
                    )
            elif isinstance(saved_config, dict):
                # It's a dict
                if saved_config.get('symptom_vocab_size') != model.config.symptom_vocab_size:
                    raise ValueError(
                        f"Checkpoint symptom_vocab_size ({saved_config.get('symptom_vocab_size')}) "
                        f"doesn't match model ({model.config.symptom_vocab_size})"
                    )
                if saved_config.get('icd_vocab_size') != model.config.icd_vocab_size:
                    raise ValueError(
                        f"Checkpoint icd_vocab_size ({saved_config.get('icd_vocab_size')}) "
                        f"doesn't match model ({model.config.icd_vocab_size})"
                    )
                if saved_config.get('latent_dim') != model.config.latent_dim:
                    raise ValueError(
                        f"Checkpoint latent_dim ({saved_config.get('latent_dim')}) "
                        f"doesn't match model ({model.config.latent_dim})"
                    )
    
    # Check for shape mismatches
    shape_mismatches = []
    for key in model_state.keys():
        if key in model_dict:
            if model_state[key].shape != model_dict[key].shape:
                shape_mismatches.append(
                    f"{key}: checkpoint={model_state[key].shape}, model={model_dict[key].shape}"
                )
    
    if shape_mismatches and strict:
        raise ValueError(
            f"Model architecture mismatch. Shape differences:\n" + 
            "\n".join(shape_mismatches)
        )
    elif shape_mismatches:
        logger.warning("Shape mismatches found (loading in non-strict mode):")
        for mismatch in shape_mismatches:
            logger.warning(f"  {mismatch}")
    
    # Load model state
    try:
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except Exception as e:
        raise ValueError(f"Failed to load model state dict: {str(e)}")
    
    # Log summary of parameters not loaded (non-strict mode)
    if not strict:
        missing_keys = incompatible_keys.missing_keys if hasattr(incompatible_keys, 'missing_keys') else []
        unexpected_keys = incompatible_keys.unexpected_keys if hasattr(incompatible_keys, 'unexpected_keys') else []
        
        if missing_keys or unexpected_keys or shape_mismatches:
            logger.warning("=" * 60)
            logger.warning("CHECKPOINT LOADING SUMMARY (non-strict mode)")
            logger.warning("=" * 60)
            
            if shape_mismatches:
                logger.warning(f"Shape mismatches: {len(shape_mismatches)} parameters")
                for mismatch in shape_mismatches[:5]:  # Show first 5
                    logger.warning(f"  - {mismatch}")
                if len(shape_mismatches) > 5:
                    logger.warning(f"  ... and {len(shape_mismatches) - 5} more")
            
            if missing_keys:
                logger.warning(f"Missing in checkpoint: {len(missing_keys)} parameters")
                for key in missing_keys[:5]:  # Show first 5
                    logger.warning(f"  - {key}")
                if len(missing_keys) > 5:
                    logger.warning(f"  ... and {len(missing_keys) - 5} more")
            
            if unexpected_keys:
                logger.warning(f"Unexpected in checkpoint: {len(unexpected_keys)} parameters")
                for key in unexpected_keys[:5]:  # Show first 5
                    logger.warning(f"  - {key}")
                if len(unexpected_keys) > 5:
                    logger.warning(f"  ... and {len(unexpected_keys) - 5} more")
            
            logger.warning("=" * 60)
            logger.warning("Some parameters were not loaded. Model may need retraining.")
            logger.warning("=" * 60)
    
    # Load optimizer and scheduler (unless weights_only)
    if not weights_only:
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
    
    logger.info(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
    
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
    save_best_only: bool = True,
    max_grad_norm: Optional[float] = None,
    show_progress: bool = False
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
        max_grad_norm: Optional maximum gradient norm for clipping (None = no clipping).
                      Helps prevent gradient explosion, especially with focal loss.
        show_progress: If True and tqdm is available, show progress bar during training
        
    Returns:
        Dictionary of training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_metrics': [],
        'learning_rate': []
    }
    
    # Try to import tqdm if progress bars requested
    tqdm_available = False
    if show_progress:
        try:
            from tqdm import tqdm
            tqdm_available = True
        except ImportError:
            logger.warning("tqdm not installed, progress bars disabled. Install with: pip install tqdm")
    
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
    
    # Setup epoch iterator with optional progress bar
    epoch_range = range(num_epochs)
    if tqdm_available:
        epoch_range = tqdm(epoch_range, desc="Training", unit="epoch")
    
    for epoch in epoch_range:
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
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            # Track metrics (use .detach() to avoid keeping computation graphs)
            epoch_losses.append(loss.detach().item())
            for key, val in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(val.detach().item() if torch.is_tensor(val) else val)
            
            # Clean up to free memory
            del loss, output
            if direction in [2, 3]:  # Cycle computations use more memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Logging
            if global_step % log_interval == 0:
                avg_loss = sum(epoch_losses[-log_interval:]) / min(log_interval, len(epoch_losses))
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {avg_loss:.4f}")
                
                # Log memory usage on GPU
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                    logger.debug(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                if callback:
                    callback_metrics = {'loss': avg_loss}
                    for k, v in loss_dict.items():
                        callback_metrics[k] = v.detach().item() if torch.is_tensor(v) else v
                    callback(epoch, global_step, callback_metrics)
            
            global_step += 1
        
        # Average epoch metrics
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history['train_loss'].append(avg_epoch_loss)
        
        epoch_avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        history['epoch_metrics'].append(epoch_avg_metrics)
        
        logger.info(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        for key, val in epoch_avg_metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate_brism(model, val_loader, loss_fn, device)
            history['val_loss'].append(val_loss)
            logger.info(f"  Validation Loss: {val_loss:.4f}")
            
            # Check early stopping
            if early_stopping is not None:
                if early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save checkpoint
        if checkpointer is not None:
            checkpoint_metrics = {
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss if val_loader is not None else avg_epoch_loss,
                **epoch_avg_metrics
            }
            checkpointer(model, optimizer, epoch+1, checkpoint_metrics, scheduler)
        
        # Track current learning rate (before scheduler step)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Clean up after epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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


def configure_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """
    Configure logging for BRISM training.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file to write logs to (in addition to console)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured at {level} level")

