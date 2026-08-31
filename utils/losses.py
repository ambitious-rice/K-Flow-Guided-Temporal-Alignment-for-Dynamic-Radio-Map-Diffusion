import torch
import torch.nn.functional as F


def masked_mean(values, mask, eps=1e-8):
    """Per-sample masked mean for tensors shaped ``(B, C, H, W)``.

    ``mask`` may have one channel and is broadcast over ``values`` channels.
    Returning a per-sample vector makes distributed aggregation and weighting
    explicit in callers.
    """
    if values.ndim != 4 or mask.ndim != 4:
        raise ValueError("values and mask must be shaped (B, C, H, W)")
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    if values.shape != mask.shape:
        raise ValueError(f"values shape {values.shape} and mask shape {mask.shape} are incompatible")
    numerator = (values * mask).flatten(1).sum(1)
    denominator = mask.flatten(1).sum(1).clamp_min(eps)
    return numerator / denominator


def cal_pinn_masked(cal, valid_mask, shooter, k=1.0):
    """PINN loss restricted strictly to the dynamic free-space domain.

    ``valid_mask`` is ``Ω = not building and not vehicle``.  Obstacle pixels
    are zeroed before the finite-difference stencil and never enter the PDE or
    source loss.  This intentionally does not add a loss on building/vehicle
    pixels: those are known excluded cells, not RSS supervision targets.
    """
    if cal.ndim == 3:
        cal = cal.unsqueeze(1)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if shooter.ndim == 3:
        shooter = shooter.unsqueeze(1)
    if cal.ndim != 4 or valid_mask.ndim != 4 or shooter.ndim != 4:
        raise ValueError("cal, valid_mask and shooter must be 3D or 4D tensors")

    valid_mask = (valid_mask > 0.5).to(device=cal.device, dtype=cal.dtype)
    shooter_mask = (shooter > 0.5).to(device=cal.device, dtype=cal.dtype) * valid_mask
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=cal.device,
        dtype=cal.dtype,
    ).view(1, 1, 3, 3)
    # Zeroing obstacle cells avoids propagating their arbitrary target values
    # into neighbouring PDE stencils while the residual itself remains masked.
    lap = F.conv2d(cal * valid_mask, lap_kernel, padding=1)
    residual = lap + float(k) ** 2 * cal
    loss_pde = masked_mean(residual.pow(2), valid_mask)

    # Tx pixels normally fall in free space.  Keep the term well defined if a
    # heatmap overlaps an excluded cell by returning zero for that sample.
    source_num = shooter_mask.flatten(1).sum(1)
    source_loss = ((cal - 1.0).pow(2) * shooter_mask).flatten(1).sum(1)
    source_loss = torch.where(source_num > 0, source_loss / source_num.clamp_min(1.0), torch.zeros_like(source_loss))
    return loss_pde + source_loss


def cal_pinn_components(cal, buildings, shooter, k=1.0, k_building=1.0):
    """Return the governing-equation and semantic-anchor terms separately.

    The legacy ``cal_pinn`` bundled all three terms.  Keeping the components
    explicit allows a controlled regularizer replacement in which obstacle
    and transmitter anchors remain unchanged and only the equation residual
    is exchanged.
    """
    if cal.ndim != 3 or buildings.ndim != 3 or shooter.ndim != 3:
        raise ValueError("cal, buildings and shooter must be shaped (B, H, W)")

    cal_t = cal.unsqueeze(1)
    buildings_t = buildings.unsqueeze(1)
    shooter_t = shooter.unsqueeze(1)

    device = cal_t.device
    dtype = cal_t.dtype

    lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0,-4.0, 1.0],
                               [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1,1,3,3)
    lap = F.conv2d(cal_t, lap_kernel, padding=1)

    buildings_mask = (buildings_t > 0.5)
    shooter_mask = (shooter_t > 0.5)

    k_tensor = torch.tensor(float(k), device=device, dtype=dtype)
    k_building_tensor = torch.tensor(float(k_building), device=device, dtype=dtype)
    k_map = torch.where(buildings_mask, k_building_tensor, k_tensor)

    residual = lap + (k_map ** 2) * cal_t
    equation_loss = residual.pow(2).flatten(1).mean(1)

    bc_num = buildings_mask.sum(dim=(1,2,3)).clamp_min(1)
    obstacle_loss = (cal_t.pow(2) * buildings_mask).sum(dim=(1,2,3)) / bc_num

    src_num = shooter_mask.sum(dim=(1,2,3)).clamp_min(1)
    source_loss = ((cal_t - 1.0).pow(2) * shooter_mask).sum(dim=(1,2,3)) / src_num
    return equation_loss, obstacle_loss, source_loss


def cal_pinn(cal, buildings, shooter, k=1.0, k_building=1.0):
    """
    Migrated from guided_diffusion.gaussian_diffusion.cal_pinn, behavior unchanged:
    - cal, buildings, shooter: (B, H, W)
    - Returns: (B,) per-sample PINN loss

    ``buildings`` is the original RMDM building mask.  Callers in the dynamic
    sparse protocol may pass a union of building and current-frame vehicle
    masks instead: both are zero-RSS obstacles and receive the same material
    coefficient and soft zero-field boundary treatment.
    """
    equation_loss, obstacle_loss, source_loss = cal_pinn_components(
        cal,
        buildings,
        shooter,
        k=k,
        k_building=k_building,
    )
    return equation_loss + obstacle_loss + source_loss


def full_image_hessian_charbonnier(image, *, epsilon=1.0e-3):
    """Full-image Hessian regularizer aligned with the legacy PDE domain.

    Like the legacy Helmholtz residual, this evaluates every pixel with
    one-cell zero padding.  Obstacle and transmitter semantics are deliberately
    absent here because their unchanged anchor losses are added separately by
    the training objective.
    """
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4:
        raise ValueError("image must be shaped (B, H, W) or (B, C, H, W)")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")

    dtype = image.dtype
    device = image.device
    dxx_kernel = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    dyy_kernel = dxx_kernel.transpose(-1, -2).contiguous()
    dxy_kernel = 0.25 * torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    dxx = F.conv2d(image, dxx_kernel, padding=1)
    dyy = F.conv2d(image, dyy_kernel, padding=1)
    dxy = F.conv2d(image, dxy_kernel, padding=1)
    hessian_frobenius_sq = dxx.square() + 2.0 * dxy.square() + dyy.square()
    charbonnier = torch.sqrt(hessian_frobenius_sq + float(epsilon) ** 2) - float(epsilon)
    return charbonnier.flatten(1).mean(1)


def masked_hessian_charbonnier(
    image,
    valid_mask,
    tx_heatmap,
    *,
    epsilon=1.0e-3,
    tx_threshold=0.5,
):
    """Second-order CV regularizer on complete free-space 3x3 stencils.

    The reconstruction remains full-image supervised.  This auxiliary term is
    evaluated only where the center and all eight neighbours are free space,
    and excludes a one-cell dilation of the Tx core.  It therefore never
    smooths across building/vehicle boundaries or through the source peak.
    """
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if tx_heatmap.ndim == 3:
        tx_heatmap = tx_heatmap.unsqueeze(1)
    if image.ndim != 4 or valid_mask.ndim != 4 or tx_heatmap.ndim != 4:
        raise ValueError("image, valid_mask and tx_heatmap must be 3D or 4D tensors")
    if image.shape != valid_mask.shape or image.shape != tx_heatmap.shape:
        raise ValueError("image, valid_mask and tx_heatmap must share shape")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")

    dtype = image.dtype
    device = image.device
    valid = (valid_mask > 0.5).to(dtype=dtype)
    neighbourhood_count = F.conv2d(
        valid,
        torch.ones((1, 1, 3, 3), device=device, dtype=dtype),
        padding=1,
    )
    interior = neighbourhood_count > 8.5
    tx_core = (tx_heatmap > float(tx_threshold)).to(dtype=dtype)
    tx_exclusion = F.max_pool2d(tx_core, kernel_size=3, stride=1, padding=1) > 0.5
    regularizer_mask = interior & ~tx_exclusion

    dxx_kernel = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    dyy_kernel = dxx_kernel.transpose(-1, -2).contiguous()
    dxy_kernel = 0.25 * torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    dxx = F.conv2d(image, dxx_kernel, padding=1)
    dyy = F.conv2d(image, dyy_kernel, padding=1)
    dxy = F.conv2d(image, dxy_kernel, padding=1)
    hessian_frobenius_sq = dxx.square() + 2.0 * dxy.square() + dyy.square()
    charbonnier = torch.sqrt(hessian_frobenius_sq + float(epsilon) ** 2) - float(epsilon)
    return masked_mean(charbonnier, regularizer_mask.to(dtype=dtype))
