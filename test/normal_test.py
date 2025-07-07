import torch
import cv2
import matplotlib.pyplot as plt
from old_train import FastSCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FastSCNN(n_classes=2).to(device)
model.load_state_dict(torch.load("../fast_scnn_pipe.pth", map_location=device))
model.eval()

def prepare_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device), img

img_tensor, orig_img = prepare_image("test.png")

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