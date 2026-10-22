"""MOSS mask assignment using the existing ComfyUI FreeFuse spatial regularizer."""
import torch
import torch.nn.functional as F

def linear_normalize(x, dim=1):
    """
    Normalize tensor to [0, 1] range along specified dimension.

    This is crucial for balanced_argmax to work properly - softmax normalized
    values have very small variance, while linear normalization preserves
    the relative differences.
    """
    x_min = x.min(dim=dim, keepdim=True)[0]
    x_max = x.max(dim=dim, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-08)

def stabilized_balanced_argmax(logits, h, w, target_count=None, max_iter=15, lr=1e-05, gravity_weight=1e-05, spatial_weight=4e-05, momentum=0.2, centroid_margin=0.0, border_penalty=0.0, anisotropy=1.3, debug=False):
    """
    Stabilized balanced argmax algorithm from diffusers FreeFuse implementation.

    Uses iterative bias adjustment with spatial constraints to produce
    balanced, spatially coherent mask assignments.

    Args:
        logits: (B, C, N) tensor where N = h * w
        h, w: Spatial dimensions
        target_count: Target pixel count per class (default: N/C)
        max_iter: Number of iterations
        lr: Learning rate for bias updates
        gravity_weight: Weight for centroid attraction
        spatial_weight: Weight for neighbor voting
        momentum: Momentum for probability smoothing

    Returns:
        (B, N) tensor of class assignments
    """
    (B, C, N) = logits.shape
    device = logits.device
    scale_h = 1
    scale_w = 1
    y_range = torch.linspace(-scale_h, scale_h, steps=h, device=device)
    x_range = torch.linspace(-scale_w, scale_w, steps=w, device=device)
    x_range = x_range * anisotropy
    (grid_y, grid_x) = torch.meshgrid(y_range, x_range, indexing='ij')
    flat_y = grid_y.reshape(1, 1, N)
    flat_x = grid_x.reshape(1, 1, N)
    max_dim = max(h, w)
    pixel_size = 2.0 / max_dim
    is_border = (flat_y.abs() > scale_h - pixel_size * 1.5) | (flat_x.abs() > scale_w - pixel_size * 1.5)
    border_mask = is_border.float()
    if target_count is None:
        target_count = N / C
    bias = torch.zeros(B, C, 1, device=device, dtype=logits.dtype)
    running_probs = linear_normalize(logits, dim=-1)
    logit_range = (logits.max() - logits.min()).item()
    logit_scale = max(logit_range, 0.0001)
    effective_lr = lr * logit_scale
    max_bias = logit_scale * 10.0
    if debug:
        print(f'\n[BalancedArgmax] Start. B={B}, C={C}, N={N}, Target={target_count:.1f}')
        print(f'Logits range: min={logits.min().item():.5f}, max={logits.max().item():.5f}')
        print(f'Logit scale: {logit_scale:.6f}, Effective LR: {effective_lr:.6f}')
    neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
    neighbor_kernel[:, :, 1, 1] = 0
    current_logits = logits.clone()
    for i in range(max_iter):
        probs = linear_normalize(current_logits - bias, dim=-1)
        running_probs = (1 - momentum) * probs + momentum * running_probs
        mass = running_probs.sum(dim=2, keepdim=True) + 1e-06
        center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
        center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass
        if centroid_margin > 0:
            limit_y = scale_h * (1.0 - centroid_margin)
            limit_x = scale_w * (1.0 - centroid_margin)
            center_y = torch.clamp(center_y, -limit_y, limit_y)
            center_x = torch.clamp(center_x, -limit_x, limit_x)
        dist_sq = (flat_y - center_y) ** 2 + (flat_x - center_x) ** 2
        hard_indices = torch.argmax(current_logits - bias, dim=1)
        hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)
        diff = hard_counts - target_count
        cur_lr = effective_lr * 0.95 ** i
        bias = bias + torch.sign(diff).unsqueeze(2) * cur_lr
        bias = torch.clamp(bias, -max_bias, max_bias)
        if spatial_weight > 0:
            probs_img = running_probs.view(B, C, h, w)
            probs_img_f32 = probs_img.float()
            kernel_f32 = neighbor_kernel.float()
            neighbor_votes = F.conv2d(probs_img_f32, kernel_f32, padding=1, groups=C)
            neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
        else:
            neighbor_votes = torch.zeros_like(logits)
        gravity_term = dist_sq * gravity_weight
        border_term = border_mask * border_penalty
        current_logits = logits - bias + neighbor_votes * spatial_weight - gravity_term - border_term
        if debug and (i == 0 or i == max_iter - 1 or i % 5 == 0):
            hard_counts_list = hard_counts[0].int().tolist()
            print(f'Iter {i:02d}: Counts={hard_counts_list}')
    if debug:
        final_assignment = torch.argmax(current_logits, dim=1)[0]
        final_counts = final_assignment.flatten().bincount(minlength=C).tolist()
        print(f'[BalancedArgmax] Final Counts: {final_counts}\n')
    return torch.argmax(current_logits, dim=1)

def morphological_clean_mask(mask, h, w, opening_kernel_size=3, closing_kernel_size=3):
    """
    Clean a binary mask using morphological operations.

    Applies:
    1. Opening (erosion + dilation): removes small foreground noise
    2. Closing (dilation + erosion): fills small holes in foreground

    Args:
        mask: (B, N) binary mask, 1 for foreground, 0 for background
        h, w: spatial dimensions (N = h * w)
        opening_kernel_size: kernel size for opening operation
        closing_kernel_size: kernel size for closing operation

    Returns:
        cleaned_mask: (B, N) cleaned binary mask
    """
    B = mask.shape[0]
    device = mask.device
    dtype = mask.dtype
    mask_2d = mask.view(B, 1, h, w).float()

    def dilate(x, kernel_size):
        padding = kernel_size // 2
        out = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out

    def erode(x, kernel_size):
        padding = kernel_size // 2
        out = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=padding)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
        return out
    if opening_kernel_size > 1:
        opened = erode(mask_2d, opening_kernel_size)
        opened = dilate(opened, opening_kernel_size)
    else:
        opened = mask_2d
    if closing_kernel_size > 1:
        closed = dilate(opened, closing_kernel_size)
        closed = erode(closed, closing_kernel_size)
    else:
        closed = opened
    cleaned_mask = closed.view(B, -1)
    return cleaned_mask.to(dtype)
def process_moss_maps(maps, h, w, bg_scale=0.95, clean=True):
    names = [name for name in maps if not name.startswith("__")]
    logits = torch.stack([maps[name][..., 0] for name in names], dim=1).float()
    labels = stabilized_balanced_argmax(logits, h, w)
    foreground = torch.ones_like(labels, dtype=torch.float32)
    if "__background__" in maps:
        background = maps["__background__"][..., 0].float() * bg_scale
        foreground = (logits.amax(1) >= background).float()
        if clean:
            foreground = morphological_clean_mask(foreground, h, w)
    return {name: ((labels == index).float() * foreground)[..., None]
            for index, name in enumerate(names)}
