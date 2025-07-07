import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomPerspective

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

class PipeSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        mask = (mask > 127).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

random_perspective = RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=InterpolationMode.BILINEAR)

def simple_transform(img, mask):
    if torch.rand(1) < 0.5:
        img = torch.flip(img, dims=[2])
        mask = torch.flip(mask, dims=[1])

    if torch.rand(1) < 0.3:
        img = torch.flip(img, dims=[1])
        mask = torch.flip(mask, dims=[0])

    if torch.rand(1) < 0.3:
        angle = float(torch.empty(1).uniform_(-20, 20))
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask.unsqueeze(0).float(), angle, interpolation=InterpolationMode.NEAREST).squeeze(0).long()

    if torch.rand(1) < 0.3:
        brightness = float(torch.empty(1).uniform_(0.7, 1.3))
        contrast = float(torch.empty(1).uniform_(0.7, 1.3))
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)

    if torch.rand(1) < 0.2:
        img = TF.gaussian_blur(img, kernel_size=3)

    if torch.rand(1) < 0.2:
        seed = torch.seed()
        torch.manual_seed(seed)
        img = random_perspective(img)
        torch.manual_seed(seed)
        mask = random_perspective(mask.unsqueeze(0).float()).squeeze(0).long()

    return img, mask

def main():
    image_dir = "Original" # Orijinal dataset
    mask_dir = "Mask" # 0,1 bitlik kullanılan maske dataset

    dataset = PipeSegmentationDataset(image_dir, mask_dir, transform=simple_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FastSCNN(n_classes=2).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0.0
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "fast_scnn_pipe.pth")
    print("fast_scnn_pipe.pth")

    model.eval()
    traced = torch.jit.trace(model.cpu(), torch.randn(1, 3, 256, 256))
    traced.save("fast_scnn_pipe_traced.pt")
    print("fast_scnn_pipe_traced.pt")

    try:
        import coremltools as ct
        mlmodel = ct.convert(
            traced,
            inputs=[ct.ImageType(name="input_1", shape=(1, 3, 256, 256), scale=1/255.0)],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.macOS11
        )
        mlmodel.save("fast_scnn_pipe.mlmodel")
        print("CoreML:fast_scnn_pipe.mlmodel")
    except ImportError:
        print("coreML bulunamadı, işlem es geçiliyor")

if __name__ == "__main__":
    main()