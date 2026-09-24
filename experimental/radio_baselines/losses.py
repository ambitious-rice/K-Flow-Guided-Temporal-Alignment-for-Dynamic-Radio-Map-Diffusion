"""RME-GAN's two-phase objectives under normalized RSS and unknown Tx location."""
import torch
from torch import nn
from torch.nn import functional as F


class SparsePropagationTemplate(nn.Module):
    """Least-squares log-distance fit; candidate source and coefficients use observations only."""
    def __init__(self, size=128, grid=16):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        cy, cx = torch.meshgrid(torch.linspace(0, size - 1, grid),
                                torch.linspace(0, size - 1, grid), indexing='ij')
        distances = ((yy.flatten()[None] - cy.flatten()[:, None]).square() +
                     (xx.flatten()[None] - cx.flatten()[:, None]).square()).sqrt().clamp_min(1)
        self.register_buffer('log_distance', distances.log10(), persistent=False)

    @torch.no_grad()
    def forward(self, observed, mask):
        y, m = (observed * mask).flatten(1).float(), mask.flatten(1).float()
        count = m.sum(1, keepdim=True).clamp_min(1)
        d = self.log_distance.float()
        sx = m @ d.T
        sxx = m @ d.square().T
        sy = y.sum(1, keepdim=True)
        sxy = y @ d.T
        var_x = (sxx - sx.square() / count).clamp_min(1e-6)
        cov = sxy - sx * sy / count
        slope = (cov / var_x).clamp(max=0)
        intercept = (sy - slope * sx) / count
        # Residual sum of squares with the constrained, nonpositive slope.
        sse = (y.square().sum(1, keepdim=True) - sy.square() / count
               - 2 * slope * cov + slope.square() * var_x)
        chosen = sse.argmin(1)
        batch = torch.arange(len(y), device=y.device)
        template = intercept[batch, chosen, None] + slope[batch, chosen, None] * d[chosen]
        return template.view_as(observed).clamp(0, 1)


def multiscale_ssim_loss(x, y):
    """Five-scale SSIM with small windows suitable for the common 128x128 grid."""
    weights = (.0448, .2856, .3001, .2363, .1333)
    product = x.new_ones(x.shape[0])
    for level, weight in enumerate(weights):
        ux, uy = F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(y, 3, 1, 1)
        vx = (F.avg_pool2d(x.square(), 3, 1, 1) - ux.square()).clamp_min(0)
        vy = (F.avg_pool2d(y.square(), 3, 1, 1) - uy.square()).clamp_min(0)
        cov = F.avg_pool2d(x * y, 3, 1, 1) - ux * uy
        contrast = (2 * cov + .03**2) / (vx + vy + .03**2)
        if level == len(weights) - 1:
            contrast = contrast * (2 * ux * uy + .01**2) / (ux.square() + uy.square() + .01**2)
        product = product * contrast.mean((1, 2, 3)).clamp_min(1e-6).pow(weight)
        if level < len(weights) - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return 1 - product.mean()


class RMEObjectives(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.template = SparsePropagationTemplate(size)
        kernels = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype=torch.float32)
        self.register_buffer('sobel', kernels[:, None])
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        self.register_buffer('cells', ((yy * 10 // size) * 10 + xx * 10 // size).flatten())
        fy, fx = torch.meshgrid(torch.fft.fftfreq(size), torch.fft.rfftfreq(size), indexing='ij')
        self.register_buffer('high_frequency', (fy.square() + fx.square()).flatten().topk(100).indices)

    def geometry_loss(self, pred, observed, mask):
        values = observed.flatten(1).masked_fill(mask.flatten(1) == 0, -torch.inf)
        cells = self.cells[None].expand(len(pred), -1)
        maxima = pred.new_full((len(pred), 100), -torch.inf)
        maxima.scatter_reduce_(1, cells, values, reduce='amax', include_self=True)
        eligible = (values == maxima.gather(1, cells)) & (mask.flatten(1) > 0)
        indices = torch.arange(values.shape[1], device=pred.device)[None].expand_as(cells)
        chosen = indices.masked_fill(~eligible, values.shape[1])
        selected = torch.full((len(pred), 100), values.shape[1], device=pred.device, dtype=torch.long)
        selected.scatter_reduce_(1, cells, chosen, reduce='amin', include_self=True)
        present = selected < values.shape[1]
        error = pred.flatten(1).gather(1, selected.clamp_max(values.shape[1] - 1))
        reference = observed.flatten(1).gather(1, selected.clamp_max(values.shape[1] - 1))
        return ((error - reference).square() * present).sum() / present.sum().clamp_min(1)

    def forward(self, pred, target, x, phase, adversarial):
        pred, target, x = pred.float(), target.float(), x.float()
        mse = F.mse_loss(pred, target)
        tv = ((pred[..., 1:, :] - pred[..., :-1, :]).square().mean() +
              (pred[..., :, 1:] - pred[..., :, :-1]).square().mean())
        logs = dict(mse=mse, adversarial=adversarial, tv=tv)
        if phase == 'first':
            prior = self.template(x[:, 3:4], x[:, 4:5])
            gp = F.conv2d(pred, self.sobel, padding=1)
            gr = F.conv2d(prior, self.sobel, padding=1)
            valid = gr.square().sum(1) > 1e-8
            gradient = ((1 - F.cosine_similarity(gp, gr, dim=1)) * valid).sum() / valid.sum().clamp_min(1)
            logs['gradient'] = gradient
            total = .01 * adversarial + 10 * mse + .01 * tv + .01 * gradient
        else:
            geometry = self.geometry_loss(pred, x[:, 3:4], x[:, 4:5])
            fft_error = torch.fft.rfft2(pred - target, norm='ortho').flatten(1)
            high_frequency = fft_error[:, self.high_frequency].abs().square().mean()
            ssim = multiscale_ssim_loss(pred, target)
            logs.update(geometry=geometry, high_frequency=high_frequency, ms_ssim=ssim)
            total = .01 * adversarial + mse + .001 * tv + .1 * ssim + geometry + .01 * high_frequency
        return total, logs


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        import lpips
        from .vendor.radiodiff.discriminator import NLayerDiscriminator, weights_init
        self.perceptual = lpips.LPIPS(net='vgg', verbose=False).eval().requires_grad_(False)
        self.discriminator = NLayerDiscriminator(input_nc=1).apply(weights_init)

    def generator_loss(self, model, pred, target, posterior, use_discriminator):
        pred, target = pred.float(), target.float()
        perceptual = self.perceptual(pred.repeat(1, 3, 1, 1) * 2 - 1,
                                     target.repeat(1, 3, 1, 1) * 2 - 1).mean()
        reconstruction = (pred - target).abs().mean() + F.mse_loss(pred, target) + perceptual
        nll = reconstruction / model.logvar.exp() + model.logvar
        kl = posterior.kl().mean() / target[0].numel()
        total = nll + 1e-6 * kl
        if use_discriminator:
            adversarial = -self.discriminator(pred).mean()
            last = model.decoder.conv_out.weight
            nll_grad = torch.autograd.grad(nll, last, retain_graph=True)[0]
            adv_grad = torch.autograd.grad(adversarial, last, retain_graph=True)[0]
            weight = (.5 * (nll_grad.norm() / (adv_grad.norm() + 1e-4)).clamp(0, 1e4)).detach()
            total = total + weight * adversarial
        return total, dict(reconstruction=reconstruction, kl=kl)
