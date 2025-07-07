import os
from glob import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


class LightDistanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv3(x)))  # 16 -> 8
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

class RealDistanceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, distances, transform_img=None, transform_mask=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.distances = distances
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        distance = torch.tensor(self.distances[idx], dtype=torch.float32)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask, distance

def extract_index(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1

def get_sorted_file_lists(original_dir, mask_dir, distance_dir):
    orig_files = glob(os.path.join(original_dir, "*.png"))
    orig_files = [(f, extract_index(os.path.basename(f), r"img_(\d+)\.png")) for f in orig_files]
    orig_files = sorted(orig_files, key=lambda x: x[1])

    mask_files = glob(os.path.join(mask_dir, "*.png"))
    mask_files = [(f, extract_index(os.path.basename(f), r"mask_(\d+)\.png")) for f in mask_files]
    mask_files = sorted(mask_files, key=lambda x: x[1])

    distance_files = glob(os.path.join(distance_dir, "*.txt"))
    distance_files = [(f, extract_index(os.path.basename(f), r"distance_(\d+)\.txt")) for f in distance_files]
    distance_files = sorted(distance_files, key=lambda x: x[1])

    mask_dict = {idx: f for f, idx in mask_files}
    dist_dict = {idx: f for f, idx in distance_files}

    file_list = []
    for orig_path, idx in orig_files:
        if idx in mask_dict and idx in dist_dict:
            file_list.append((orig_path, mask_dict[idx], dist_dict[idx]))
        else:
            print(f"Uyarı:Maske veya index bulunmuyor {idx}")

    distances = []
    for _, _, dist_path in file_list:
        with open(dist_path, 'r') as f:
            val = f.readline().strip()
            try:
                distances.append(float(val))
            except:
                print(f"Uyarı: Geçersiz değer-> {dist_path}")
                distances.append(0.0)

    image_paths = [x[0] for x in file_list]
    mask_paths = [x[1] for x in file_list]

    return image_paths, mask_paths, distances

def train_model(model, dataloader, criterion, optimizer, epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, masks, distances in dataloader:
            images, masks, distances = images.to(device), masks.to(device), distances.to(device)
            inputs = torch.cat([images, masks], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, distances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    original_dir = "../Original"
    mask_dir = "../Mask"
    distance_dir = "../Distance"

    image_paths, mask_paths, distances = get_sorted_file_lists(original_dir, mask_dir, distance_dir)
    print(f"{len(image_paths)} tane örnek yüklendi")

    transform_img = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_mask = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    dataset = RealDistanceDataset(image_paths, mask_paths, distances, transform_img, transform_mask)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightDistanceNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, dataloader, criterion, optimizer, epochs=50, device=device)

    torch.save(model.state_dict(), "../test/light_distance_net.pth")
    print("Model light_distance_net.pth diye kaydedildi")