import torch
import cv2
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class FastSCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 64, stride=2)
        )
        self.global_feature = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, n_classes, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.downsample(x)
        g = self.global_feature(x)
        x = x + g
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

def prepare_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device), img

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FastSCNN(n_classes=2).to(device)
model.load_state_dict(torch.load("fast_scnn_pipe.pth", map_location=device))
model.eval()

img_tensor, orig_img = prepare_image("input0.png")

with torch.no_grad():
    out = model(img_tensor)
    pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(orig_img)

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="jet")
plt.show()