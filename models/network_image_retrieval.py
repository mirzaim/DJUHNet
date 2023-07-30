
import torch
import torch.nn as nn
import torchvision
import timm


class FeatureExNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('tresnet_l', num_classes=81)
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.head.fc.in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.backbone.reset_classifier(0)

        self.head = nn.Sequential(
            nn.Linear(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.neck(self.backbone(x))
        xq = self.head(x)
        out = self.classifier(xq)
        return x, torch.sigmoid(xq), out


class ResNet50(nn.Module):
    def __init__(self, hash_bit, pretrained=False):
        super().__init__()
        model_resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y

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

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

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
