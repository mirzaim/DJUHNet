
import torch


class DPSHLoss(torch.nn.Module):
    def __init__(self, n_class, bit, num_train, device, alpha=0.1):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(num_train, bit).float().to(device)
        self.Y = torch.zeros(num_train, n_class).float().to(device)
        self.alpha = alpha

    def forward(self, u, y, ind):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y.float() @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + \
            inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = self.alpha * (u - u.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss


def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n


def quantization_swdc_loss(b, device='cuda', aggregate=True):
    real_b = torch.randn(b.shape, device=device).sign()
    bsize, dim = b.size()

    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss
