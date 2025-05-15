import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2


# ---------------------- Enhanced Configuration ----------------------
class Config:
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 20
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 1
    MODEL_NAME = 'fast_scnn_robust'
    SAVE_DIR = 'saved_models'
    AUGMENT = False
    ORIGINAL_DIR = 'Original'
    MASK_DIR = 'Mask'
    REGULARIZATION_FACTOR = 0.001
    DROPOUT_RATE = 0.3
    EARLY_STOPPING_PATIENCE = 3
    MIN_LEARNING_RATE = 1e-5
    MONITOR_METRIC = 'val_binary_mean_iou'


config = Config()
os.makedirs(config.SAVE_DIR, exist_ok=True)


# ---------------------- Visualization Functions ----------------------
def visualize_mask_distribution(masks):
    """Visualize the distribution of mask coverage percentages"""
    coverage = np.mean(masks, axis=(1, 2, 3))
    plt.figure(figsize=(10, 5))
    plt.hist(coverage, bins=50)
    plt.title('Distribution of Mask Coverage')
    plt.xlabel('Coverage Percentage')
    plt.ylabel('Number of Images')
    plt.show()


def visualize_prediction(model, image, true_mask):
    """Visualize model prediction compared to ground truth"""
    pred_mask = model.predict(np.expand_dims(image, axis=0))[0]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze() > 0.5, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    metrics = ['loss', 'precision', 'recall']
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        val_key = f'val_{metric}'
        if val_key in history.history:
            plt.plot(history.history[val_key], label=f'Validation {metric}')
        plt.title(metric.upper())
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------- Enhanced Data Loading ----------------------
def load_data(original_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(original_dir) if f.startswith('img_')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith('mask_')])

    images, masks = [], []
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(original_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, config.IMG_SIZE)
        img = img.astype('float32') / 255.0

        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, config.IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype('float32')
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


# ---------------------- Enhanced Model Architecture ----------------------
def conv_block(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_regularizer=regularizers.l2(config.REGULARIZATION_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def depthwise_separable_conv(x, filters, strides=1):
    x = layers.DepthwiseConv2D(3, strides=strides, padding='same',
                               kernel_regularizer=regularizers.l2(config.REGULARIZATION_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding='same',
                      kernel_regularizer=regularizers.l2(config.REGULARIZATION_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_robust_fast_scnn(input_shape=(256, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Learning to Downsample with reduced capacity
    x = conv_block(inputs, 24, strides=2)  # Reduced from 32
    x = depthwise_separable_conv(x, 36, strides=2)  # Reduced from 48
    ld_output = depthwise_separable_conv(x, 48, strides=2)  # Reduced from 64

    # Global Feature Extractor with additional dropout
    x = conv_block(ld_output, 48, 1)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = depthwise_separable_conv(x, 48, strides=2)
    for _ in range(2):  # Reduced from 3
        x = depthwise_separable_conv(x, 48)
        x = layers.Dropout(config.DROPOUT_RATE / 2)(x)

    x = depthwise_separable_conv(x, 72, strides=2)  # Reduced from 96
    for _ in range(2):  # Reduced from 3
        x = depthwise_separable_conv(x, 72)
        x = layers.Dropout(config.DROPOUT_RATE / 2)(x)

    gfe_output = x

    # Feature Fusion
    gfe_upsampled = layers.UpSampling2D(4, interpolation='bilinear')(gfe_output)
    fused = layers.Concatenate()([ld_output, gfe_upsampled])
    fused = conv_block(fused, 96)  # Reduced from 128
    fused = layers.Dropout(config.DROPOUT_RATE)(fused)

    # Classifier
    x = conv_block(fused, 96, 1)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.UpSampling2D(8, interpolation='bilinear')(x)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return models.Model(inputs=inputs, outputs=outputs)


# ---------------------- Enhanced Loss Functions ----------------------
def focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # Use binary crossentropy because output already has sigmoid
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Focal Loss
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0))

    # Dice Loss
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)

    return tf.reduce_mean(focal_loss) + dice_loss

def dice_coef(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_pred, axis=(1, 2))
    return (2. * intersection + 1e-6) / (union + 1e-6)


# ---------------------- Enhanced Data Generator ----------------------
class RobustMaskGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, augment=False):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        ) if augment else ImageDataGenerator()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.augment:
            aug_x, aug_y = [], []
            for x, y in zip(batch_x, batch_y):
                seed = np.random.randint(1000)
                aug_x.append(self.datagen.random_transform(x, seed=seed))
                aug_y.append(self.datagen.random_transform(y, seed=seed))
            return np.array(aug_x), np.array(aug_y)
        return batch_x, batch_y

def binary_mean_iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# ---------------------- Enhanced Training Pipeline ----------------------
def main():
    # Clear any existing session
    tf.keras.backend.clear_session()

    # Configure for M1/M2 Macs
    tf.config.run_functions_eagerly(True)  # For debugging
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

    # Load data
    images, masks = load_data(config.ORIGINAL_DIR, config.MASK_DIR)
    print(f"Data shape: {images.shape}, Mask shape: {masks.shape}")
    print(f"Unique mask values: {np.unique(masks)}")

    # Verify no NaN values
    assert not np.isnan(images).any(), "NaN values found in images"
    assert not np.isnan(masks).any(), "NaN values found in masks"

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks,
        test_size=0.2,
        random_state=42
    )

    # Create generators (fixed this section)
    train_gen = RobustMaskGenerator(X_train, y_train, config.BATCH_SIZE, augment=config.AUGMENT)
    val_gen = RobustMaskGenerator(X_val, y_val, config.BATCH_SIZE)

    # Debug the first batch
    print("\nDebugging first training batch...")
    first_batch_x, first_batch_y = train_gen[0]
    print(f"First batch - X shape: {first_batch_x.shape}, y shape: {first_batch_y.shape}")
    print(f"X range: {np.min(first_batch_x)} to {np.max(first_batch_x)}")
    print(f"y unique values: {np.unique(first_batch_y)}")

    # Build model
    model = build_robust_fast_scnn(
        input_shape=(*config.IMG_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )

    # Test forward pass
    print("\nTesting model with random input...")
    test_input = np.random.rand(1, *config.IMG_SIZE, 3).astype('float32')
    test_output = model.predict(test_input)
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: {np.min(test_output)} to {np.max(test_output)}")

    # Define callbacks (was missing in original)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.SAVE_DIR, f'{config.MODEL_NAME}_best.h5'),
            monitor='val_iou',
            mode='max',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou',
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.SAVE_DIR, 'training_log.csv')
        )
    ]

    # Compile model with legacy optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=focal_dice_loss,
        metrics=[binary_mean_iou] 
    )

    # Final verification
    print("\nFinal verification:")
    print(f"Train generator length: {len(train_gen)} batches")
    print(f"Val generator length: {len(val_gen)} batches")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Start training
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=1,
        use_multiprocessing=False
    )

    # Save and visualize results
    model.save(os.path.join(config.SAVE_DIR, f'{config.MODEL_NAME}_final.h5'))
    plot_training_history(history)
    visualize_prediction(model, X_val[0], y_val[0])


if __name__ == '__main__':
    main()