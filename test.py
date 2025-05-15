import os
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# === Config ===
class TestConfig:
    IMG_SIZE = (256, 256)
    MODEL_PATH = 'saved_models/fast_scnn_robust_final.h5'
    TEST_IMAGE_DIR = 'test'
    SAVE_RESULTS_DIR = 'test_results'
    THRESHOLD = 0.5
    SAVE_IMAGES = True  # False yaparsan çizim/kaydetme devre dışı olur


config = TestConfig()
os.makedirs(config.SAVE_RESULTS_DIR, exist_ok=True)


# === Yükleme ===
def load_trained_model(path):
    try:
        model = load_model(path, compile=False)
        print(f"Model yüklendi: {path}")
        return model
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        return None


# === Görüntü Ön İşleme ===
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Okunamayan görsel: {path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    img = img.astype('float32') / 255.0
    return img


# === Görselleştir ve Kaydet ===
def save_prediction_visual(original, prediction, filename):
    mask = (prediction > config.THRESHOLD).astype('uint8') * 255

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title('Raw Prediction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')

    save_path = os.path.join(config.SAVE_RESULTS_DIR, f"result_{filename}")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Kayıt: {save_path}")


# === Ana Test Fonksiyonu ===
def test_model():
    model = load_trained_model(config.MODEL_PATH)
    if model is None:
        return

    image_files = [f for f in os.listdir(config.TEST_IMAGE_DIR)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("Test klasöründe resim bulunamadı.")
        return

    total_time = 0.0

    for idx, file in enumerate(image_files, 1):
        img_path = os.path.join(config.TEST_IMAGE_DIR, file)
        img = preprocess_image(img_path)
        if img is None:
            continue

        input_tensor = tf.expand_dims(img, axis=0)

        # Ölçüm başlat
        start = time.time()
        prediction = model(input_tensor, training=False)[0].numpy()
        end = time.time()

        step_time = (end - start) * 1000
        total_time += step_time
        print(f"[{idx}/{len(image_files)}] {file} → {step_time:.2f} ms")

        # Görsel kaydetme opsiyonel
        if config.SAVE_IMAGES:
            save_prediction_visual(img, prediction, file)

    avg_time = total_time / len(image_files)
    print(f"\n✅ Ortalama inference süresi: {avg_time:.2f} ms / görsel")


# === Çalıştır ===
if __name__ == '__main__':
    # GPU kullanımı için eager mode kapat
    tf.config.run_functions_eagerly(False)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    test_model()