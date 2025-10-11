"""
Model export functionality for BRISM deployment.

Provides utilities for exporting BRISM models to various formats:
- ONNX for cross-platform inference
- TorchScript for optimized PyTorch deployment
- Quantized models for mobile/edge deployment
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .model import BRISM, BRISMConfig

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: BRISM,
    output_path: str,
    symptom_vocab_size: Optional[int] = None,
    max_symptom_length: Optional[int] = None,
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> None:
    """
    Export BRISM model to ONNX format for cross-platform deployment.
    
    Args:
        model: Trained BRISM model
        output_path: Path to save ONNX model
        symptom_vocab_size: Symptom vocabulary size (defaults to model config)
        max_symptom_length: Maximum symptom sequence length (defaults to model config)
        opset_version: ONNX opset version (default 14)
        dynamic_axes: Whether to use dynamic batch/sequence axes
        
    Raises:
        ImportError: If ONNX is not installed
        RuntimeError: If export fails
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError(
            "ONNX export requires 'onnx' and 'onnxruntime' packages. "
            "Install with: pip install onnx onnxruntime"
        )
    
    model.eval()
    
    # Get dimensions from model config
    symptom_vocab_size = symptom_vocab_size or model.config.symptom_vocab_size
    max_symptom_length = max_symptom_length or model.config.max_symptom_length
    
    # Create dummy input
    dummy_symptoms = torch.zeros(1, max_symptom_length, dtype=torch.long)
    
    # Define dynamic axes for variable batch size and sequence length
    if dynamic_axes:
        dynamic_axes_dict = {
            'symptoms': {0: 'batch_size', 1: 'seq_length'},
            'icd_probs': {0: 'batch_size'},
            'latent': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    try:
        # Export model
        torch.onnx.export(
            model,
            dummy_symptoms,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['symptoms'],
            output_names=['icd_probs', 'latent'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Successfully exported model to ONNX: {output_path}")
        logger.info(f"  Opset version: {opset_version}")
        logger.info(f"  Dynamic axes: {dynamic_axes}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def export_to_torchscript(
    model: BRISM,
    output_path: str,
    method: str = 'save',
    optimize: bool = True
) -> None:
    """
    Export BRISM model to TorchScript for optimized deployment.
    
    Note: Due to the complexity of BRISM's forward method returning dictionaries,
    this function saves the model's state_dict for loading with torch.load.
    For true TorchScript compilation, consider wrapping specific forward methods.
    
    Args:
        model: Trained BRISM model
        output_path: Path to save model
        method: Export method ('save' for state dict, or 'script' for experimental)
        optimize: Whether to optimize for inference (only for 'script' method)
        
    Raises:
        ValueError: If method is invalid
        RuntimeError: If export fails
    """
    model.eval()
    
    if method not in ['save', 'script']:
        raise ValueError(f"method must be 'save' or 'script', got {method}")
    
    try:
        if method == 'save':
            # Save as standard PyTorch checkpoint (most reliable)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config
            }, output_path)
            
        else:  # script (experimental)
            # This may fail due to dictionary returns and dynamic control flow
            try:
                scripted_model = torch.jit.script(model)
                
                if optimize:
                    scripted_model = torch.jit.optimize_for_inference(scripted_model)
                
                scripted_model.save(output_path)
            except Exception as e:
                logger.warning(f"TorchScript compilation failed: {e}")
                logger.info("Falling back to state_dict save")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': model.config
                }, output_path)
        
        logger.info(f"Successfully exported model: {output_path}")
        logger.info(f"  Method: {method}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to export model: {str(e)}")


def quantize_model(
    model: BRISM,
    output_path: str,
    quantization_type: str = 'dynamic',
    dtype: torch.dtype = torch.qint8
) -> BRISM:
    """
    Quantize BRISM model for reduced size and faster inference.
    
    Args:
        model: Trained BRISM model
        output_path: Path to save quantized model
        quantization_type: Type of quantization ('dynamic' or 'static')
        dtype: Data type for quantization (torch.qint8 or torch.float16)
        
    Returns:
        Quantized model
        
    Raises:
        ValueError: If quantization type is invalid
        RuntimeError: If quantization fails
    """
    if quantization_type not in ['dynamic', 'static']:
        raise ValueError(f"quantization_type must be 'dynamic' or 'static', got {quantization_type}")
    
    model.eval()
    
    try:
        if quantization_type == 'dynamic':
            # Dynamic quantization (easier, no calibration needed)
            # Note: Embeddings are not quantized by default in dynamic quantization
            # Only Linear and LSTM layers are quantized
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM},  # Don't quantize embeddings
                dtype=dtype
            )
        else:
            # Static quantization (requires calibration data)
            raise NotImplementedError(
                "Static quantization requires calibration data. "
                "Use dynamic quantization or implement calibration."
            )
        
        # Save quantized model
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': model.config,
            'quantization_type': quantization_type,
            'dtype': str(dtype)
        }, output_path)
        
        logger.info(f"Successfully quantized model: {output_path}")
        logger.info(f"  Quantization type: {quantization_type}")
        logger.info(f"  Data type: {dtype}")
        
        # Log model size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
        reduction = (1 - quantized_size / original_size) * 100
        logger.info(f"  Size reduction: {reduction:.1f}% ({original_size:.2f}MB -> {quantized_size:.2f}MB)")
        
        return quantized_model
        
    except Exception as e:
        raise RuntimeError(f"Failed to quantize model: {str(e)}")


def prune_model(
    model: BRISM,
    output_path: str,
    amount: float = 0.3,
    method: str = 'l1_unstructured'
) -> BRISM:
    """
    Prune BRISM model to reduce parameters and improve inference speed.
    
    Args:
        model: Trained BRISM model
        output_path: Path to save pruned model
        amount: Fraction of parameters to prune (0.0 to 1.0)
        method: Pruning method ('l1_unstructured', 'random_unstructured')
        
    Returns:
        Pruned model
        
    Raises:
        ValueError: If amount or method is invalid
        RuntimeError: If pruning fails
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be between 0 and 1, got {amount}")
    
    if method not in ['l1_unstructured', 'random_unstructured']:
        raise ValueError(f"method must be 'l1_unstructured' or 'random_unstructured', got {method}")
    
    try:
        import torch.nn.utils.prune as prune
        
        model.eval()
        
        # Prune linear layers
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if method == 'l1_unstructured':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        else:  # random_unstructured
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount
            )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Save pruned model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config,
            'pruning_amount': amount,
            'pruning_method': method
        }, output_path)
        
        logger.info(f"Successfully pruned model: {output_path}")
        logger.info(f"  Pruning amount: {amount * 100:.1f}%")
        logger.info(f"  Pruning method: {method}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to prune model: {str(e)}")


def export_for_deployment(
    model: BRISM,
    output_dir: str,
    formats: list = ['torchscript', 'onnx'],
    quantize: bool = False,
    prune: bool = False,
    prune_amount: float = 0.3
) -> Dict[str, str]:
    """
    Export BRISM model in multiple formats for deployment.
    
    Args:
        model: Trained BRISM model
        output_dir: Directory to save exported models
        formats: List of export formats ('torchscript', 'onnx')
        quantize: Whether to create quantized version
        prune: Whether to create pruned version
        prune_amount: Fraction of parameters to prune
        
    Returns:
        Dictionary mapping format names to file paths
        
    Examples:
        >>> from brism import BRISM, BRISMConfig
        >>> from brism.export import export_for_deployment
        >>> 
        >>> config = BRISMConfig()
        >>> model = BRISM(config)
        >>> 
        >>> # Export in multiple formats
        >>> paths = export_for_deployment(
        ...     model,
        ...     'deployment/',
        ...     formats=['torchscript', 'onnx'],
        ...     quantize=True
        ... )
        >>> print(paths)
        {'torchscript': 'deployment/model.pt', 'onnx': 'deployment/model.onnx', ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    # Export to requested formats
    for format_name in formats:
        try:
            if format_name == 'torchscript':
                path = str(output_dir / 'model.pt')
                export_to_torchscript(model, path)
                exported_files['torchscript'] = path
                
            elif format_name == 'onnx':
                path = str(output_dir / 'model.onnx')
                export_to_onnx(model, path)
                exported_files['onnx'] = path
                
            else:
                logger.warning(f"Unknown format: {format_name}")
                
        except Exception as e:
            logger.error(f"Failed to export {format_name}: {e}")
    
    # Create quantized version if requested
    if quantize:
        try:
            path = str(output_dir / 'model_quantized.pt')
            quantize_model(model, path)
            exported_files['quantized'] = path
        except Exception as e:
            logger.error(f"Failed to create quantized model: {e}")
    
    # Create pruned version if requested
    if prune:
        try:
            path = str(output_dir / 'model_pruned.pt')
            prune_model(model, path, amount=prune_amount)
            exported_files['pruned'] = path
        except Exception as e:
            logger.error(f"Failed to create pruned model: {e}")
    
    logger.info(f"Export complete. Files saved to: {output_dir}")
    return exported_files
