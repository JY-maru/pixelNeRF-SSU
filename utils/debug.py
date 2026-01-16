# utils/debug.py
"""
Debugging utilities
"""
import torch
from tqdm import tqdm

# Global debug flag - 학습 전 텐서 확인 용 
DEBUG_MODE = False


def set_debug_mode(enabled: bool):
    """
    Set global debug mode
    
    Args:
        enabled: True to enable debug mode, False to disable
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled
    if DEBUG_MODE:
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"[ DEBUG MODE ENABLED ]")
        tqdm.write(f"{'='*70}\n")


def get_debug_mode() -> bool:
    """
    Get current debug mode status
    
    Returns:
        bool: Current debug mode status
    """
    return DEBUG_MODE


def check_tensor(name, tensor, check_grad=False):
    """텐서 상태 확인"""
    if not DEBUG_MODE:
        return
    
    if tensor is None:
        tqdm.write(f"[!] {name}: None")
        return
    
    tqdm.write(f"\n▢ {name}:")
    tqdm.write(f"  Shape: {tensor.shape}")
    tqdm.write(f"  Dtype: {tensor.dtype}")
    tqdm.write(f"  Device: {tensor.device}")
    tqdm.write(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
    tqdm.write(f"  Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    
    # NaN/Inf 체크
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        tqdm.write(f"  [✕] Contains NaN! Count: {torch.isnan(tensor).sum().item()}")
    if has_inf:
        tqdm.write(f"  [✕] Contains Inf! Count: {torch.isinf(tensor).sum().item()}")
    
    if not has_nan and not has_inf:
        tqdm.write(f"  [✓] No NaN/Inf")
    
    # Gradient 체크
    if check_grad and tensor.requires_grad and tensor.grad is not None:
        tqdm.write(f"  Grad - Min: {tensor.grad.min().item():.6f}, Max: {tensor.grad.max().item():.6f}")