import os
import re
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

from train.train_distance import LightDistanceNet

def extract_index(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1

def get_test_files(original_dir, mask_dir, distance_dir):
    img_files = os.listdir(original_dir)
    mask_files = os.listdir(mask_dir)
    dist_files = os.listdir(distance_dir)

    img_files = [(f, extract_index(f, r"img_(\d+)\.png")) for f in img_files]
    mask_dict = {extract_index(f, r"mask_(\d+)\.png"): f for f in mask_files}
    dist_dict = {extract_index(f, r"distance_(\d+)\.txt"): f for f in dist_files}

    file_list = []
    for img_name, idx in img_files:
        if idx in mask_dict and idx in dist_dict:
            file_list.append((img_name, mask_dict[idx], dist_dict[idx]))
    return file_list

def run_test(model_path, original_dir, mask_dir, distance_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightDistanceNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform_img = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_mask = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    file_list = get_test_files(original_dir, mask_dir, distance_dir)
    print(f"{len(file_list)} test örneği bulundu.")

    with torch.no_grad():
        for img_name, mask_name, dist_name in file_list:
            img_path = os.path.join(original_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            dist_path = os.path.join(distance_dir, dist_name)

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            image_tensor = transform_img(image).unsqueeze(0).to(device)
            mask_tensor = transform_mask(mask).unsqueeze(0).to(device)

            input_tensor = torch.cat([image_tensor, mask_tensor], dim=1)
            pred = model(input_tensor).cpu().item()

            with open(dist_path, 'r') as f:
                real_dist = float(f.readline().strip())

            print(f"{img_name} -> Tahmin: {pred:.2f} / Gerçek: {real_dist:.2f}")

            plot_result(image, mask, pred, real_dist, img_name)

def plot_result(image, mask, pred, real, title):
    plt.figure(figsize=(6, 6))

    image_resized = image.resize(mask.size)
    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    combined = Image.blend(image_resized, mask_rgb, alpha=0.4)

    plt.imshow(combined)
    plt.axis('off')
    plt.title(f"{title}\nTahmin: {pred:.2f} / Gerçek: {real:.2f}", fontsize=10)
    plt.show()

if __name__ == '__main__':
    run_test(
        model_path="alight_distance_net.pth",
        original_dir="Original",
        mask_dir="Mask",
        distance_dir="Distance"
    )