import cv2
import torch
import time

def test_fps_torch_realtime(traced_model_path, device="cpu"):
    model = torch.jit.load(traced_model_path).to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Gerçek zamanlı FPS testi başlıyor (Torch trace)...")
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame alınamadı.")
            break

        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)
        end_time = time.time()

        fps = 1.0 / (end_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        pred = output.argmax(1).squeeze().cpu().numpy()
        mask = cv2.resize((pred * 255).astype("uint8"), (frame.shape[1], frame.shape[0]))
        overlay = frame.copy()
        overlay[mask > 127] = (0, 255, 0)

        cv2.putText(overlay, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Torch Real-time Pipe Segmentation", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fps_torch_realtime("../fast_scnn_pipe_traced.pt", device="mps" if torch.backends.mps.is_available() else "cpu")