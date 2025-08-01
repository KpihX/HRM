"""
AdamATan2 replacement for systems without CUDA support.
This is a fallback implementation that uses standard PyTorch Adam optimizer.
"""

import torch
from torch.optim import Adam


class AdamATan2(Adam):
    """
    A fallback implementation of AdamATan2 that uses standard Adam optimizer.
    This is used when CUDA extensions are not available (e.g., on Intel Arc Graphics).
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        """
        Initialize the AdamATan2 optimizer with the same interface as the original.
        Falls back to standard Adam implementation.
        
        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate (default: 1e-3)
            betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay: weight decay (L2 penalty) (default: 0)
            amsgrad: whether to use the AMSGrad variant of this algorithm (default: False)
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        print("Warning: Using standard Adam optimizer as fallback for AdamATan2 (CUDA extensions not available)")
