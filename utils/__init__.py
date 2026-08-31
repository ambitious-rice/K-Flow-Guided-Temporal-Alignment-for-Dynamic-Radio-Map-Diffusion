"""
Utility functions and model wrappers
"""

__all__ = [
    'losses',
    'model_wrapper',
    'utils',
    'logger',
    'fp16_util',
    'nn',
    'cal_pinn',
    'cal_pinn_components',
    'cal_pinn_masked',
    'full_image_hessian_charbonnier',
    'masked_hessian_charbonnier',
    'masked_mean',
    'build_unet_from_config',
    'UNetWithTimeWrapper',
]


def __getattr__(name):
    if name == 'cal_pinn':
        from .losses import cal_pinn
        return cal_pinn
    elif name == 'cal_pinn_components':
        from .losses import cal_pinn_components
        return cal_pinn_components
    elif name == 'cal_pinn_masked':
        from .losses import cal_pinn_masked
        return cal_pinn_masked
    elif name == 'full_image_hessian_charbonnier':
        from .losses import full_image_hessian_charbonnier
        return full_image_hessian_charbonnier
    elif name == 'masked_hessian_charbonnier':
        from .losses import masked_hessian_charbonnier
        return masked_hessian_charbonnier
    elif name == 'masked_mean':
        from .losses import masked_mean
        return masked_mean
    elif name == 'build_unet_from_config':
        from .model_wrapper import build_unet_from_config
        return build_unet_from_config
    elif name == 'UNetWithTimeWrapper':
        from .model_wrapper import UNetWithTimeWrapper
        return UNetWithTimeWrapper
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
