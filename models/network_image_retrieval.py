
import torch
import torch.nn as nn
import timm


class FeatureExNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet50')
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.backbone.reset_classifier(0)

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.neck(self.backbone(x))
        xq = self.head(x)
        out = self.classifier(xq)
        return x, xq, out
