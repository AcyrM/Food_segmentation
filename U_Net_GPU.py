import os
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from keras.metrics import MeanIoU
from sklearn.metrics import confusion_matrix

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 18

parent_dir = r"C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Dados\Data"
model_path = r"C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Unet_GPU_test2.h5"
model_path_3classes = r"C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Unet_GPU_binary.h5"

def cbam_block(x, reduction=16):
    channels = x.shape[-1]

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared = layers.Dense(channels // reduction, activation='relu')
    avg_out = shared(avg_pool)
    max_out = shared(max_pool)

    channel_attention = layers.Add()([avg_out, max_out])
    channel_attention = layers.Dense(channels, activation='sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1,1,channels))(channel_attention)
    x = layers.Multiply()([x, channel_attention])

    # Spatial attention
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial = tf.concat([avg_pool, max_pool], axis=-1)
    spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial)
    x = layers.Multiply()([x, spatial])

    return x

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = cbam_block(x)  # Apply CBAM block
    return x

def upsample_concat_block(x, skip, filters):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x
    
def attention_gate(x, g, filters):
    # Simple attention gate with upsampled skip
    theta_x = layers.Conv2D(filters, 1)(x)
    phi_g = layers.Conv2D(filters, 1)(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, activation="sigmoid")(act)
    return layers.Multiply()([x, psi])

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Pretrained backbone
    backbone = ResNet50(input_tensor=inputs, include_top=False, weights='imagenet')

    # Skip connections
    skips = [
        backbone.get_layer("conv1_relu").output,   # 128x128
        backbone.get_layer("conv2_block3_out").output,  # 64x64
        backbone.get_layer("conv3_block4_out").output,  # 32x32
        backbone.get_layer("conv4_block6_out").output,  # 16x16
    ]
    bottleneck = backbone.get_layer("conv5_block3_out").output  # 8x8

    # Decoder
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.SpatialDropout2D(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.SpatialDropout2D(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def upsample_concat_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        skip = attention_gate(skip, x, filters)  # << ATTENTION GATE
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    x = conv_block(bottleneck, 512)
    x = upsample_concat_block(x, skips[3], 256)
    x = upsample_concat_block(x, skips[2], 128)
    x = upsample_concat_block(x, skips[1], 64)
    x = upsample_concat_block(x, skips[0], 32)

    x = layers.UpSampling2D((2, 2))(x)
    output = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='output')(x)
    return tf.keras.Model(inputs, output)

def load_image(image_path, mask_path):
    # Load and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # RGB
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # Load and decode mask (single channel, values 0–17)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # Important: channels=1
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')  # No interpolation artifacts
    mask = tf.cast(mask, tf.uint8)

    return image, mask

def get_dataset(image_dir, mask_dir, batch_size=2, shuffle=True):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# === COLOR MAP ===
color_map = {
    0:  (0, 0, 0),
    1:  (255, 255, 255),
    2:  (0, 255, 0),
    3:  (255, 0, 0),
    4:  (255, 255, 0),
    5:  (255, 0, 255),
    6:  (0, 255, 255),
    7:  (128, 0, 0),
    8:  (0, 128, 0),
    9:  (0, 0, 128),
    10: (128, 128, 0),
    11: (128, 0, 128),
    12: (0, 128, 128),
    13: (192, 192, 192),
    14: (128, 128, 128),
    15: (255, 165, 0),
    16: (255, 20, 147),
    17: (0, 0, 255)
}

train_dataset = get_dataset(
    image_dir=os.path.join(parent_dir, "Training", "Images"),
    mask_dir=os.path.join(parent_dir, "Training", "Masks"),
    batch_size=2,
    shuffle=True
)

val_dataset = get_dataset(
    image_dir=os.path.join(parent_dir, "Validation", "Images"),
    mask_dir=os.path.join(parent_dir, "Validation", "Masks"),
    batch_size=2,
    shuffle=False
)
class_weights = tf.constant([
    0.05,  # background
    0.1,   # plate
    5.0,   # 2
    15.0,  # 3 ← weak
    15.0,   # 4
    10.0,  # 5 ← weak
    5.0,   # 6
    12.0,  # 7 ← weak
    15.0,  # 8 ← weak
    15.0,  # 9 ← weak
    5.0,   # 10
    15.0,  # 11 ← weak
    10.0,  # 12
    5.0,  # 13
    5.0,  # 14
    5.0,  # 15
    5.0,  # 16
    5.0,  # 17
], dtype=tf.float32)

def weighted_sparse_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_flat = tf.reshape(y_true, [-1])             # (batch*H*W,)
    y_pred_flat = tf.reshape(y_pred, [-1, 18])         # (batch*H*W, 18)

    weights = tf.gather(class_weights, y_true_flat)    # (batch*H*W,)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(ce * weights)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_true = tf.one_hot(y_true, 18)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)  # (B,H,W)
    y_true = tf.one_hot(y_true, 18)  # (B,H,W,C)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    ce = -y_true * tf.math.log(y_pred)
    fl = alpha * tf.math.pow(1 - y_pred, gamma) * ce
    return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

def combined_loss(y_true, y_pred):
    return 0.7 * weighted_sparse_ce(y_true, y_pred) + 0.15 * dice_loss(y_true, y_pred) + 0.15*focal_loss(y_true, y_pred)

class MeanIoUWrapper(MeanIoU):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred: (batch, H, W, 18) → argmax → (batch, H, W)
        y_pred = tf.argmax(y_pred, axis=-1)
        # y_true: ensure shape (batch, H, W)
        y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == 4 else y_true
        return super().update_state(y_true, y_pred, sample_weight)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

def compute_per_class_iou(y_true, y_pred, num_classes=18):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - intersection
    iou = intersection / np.maximum(union, 1)
    return iou

def combined_focal_dice(y_true, y_pred):
    return 0.5 * focal_loss(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


model = build_unet(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss=combined_loss, metrics=[MeanIoUWrapper(num_classes=18)])
model.load_weights(model_path_3classes, by_name=True, skip_mismatch=True)

for layer in model.layers:
    if 'resnet' in layer.name or 'encoder' in layer.name:
        layer.trainable = False

# 2. Compile and train (frozen encoder)
model.compile(optimizer='adam', loss=combined_loss, metrics=[MeanIoUWrapper(num_classes=18)])
model.fit(train_dataset, validation_data=val_dataset, epochs=3, callbacks=[lr_callback])

print("Unfreezing all layers for fine-tuning...")
# 3. Unfreeze all layers
for layer in model.layers:
    layer.trainable = True

# 4. Recompile and continue training (fine-tuning)
model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=combined_loss, metrics=[MeanIoUWrapper(num_classes=18)])
model.fit(train_dataset, validation_data=val_dataset, initial_epoch=3, epochs=100, callbacks=[lr_callback])

model.save(model_path)

model = load_model(model_path, custom_objects={
    'DiceLoss': dice_loss,
    'CategoricalFocalLoss': focal_loss,
    'combined_loss': combined_loss,
    'weighted_sparse_ce': weighted_sparse_ce,
    'dice_loss': dice_loss,
    'MeanIoUWrapper': MeanIoUWrapper
})


input_folder = parent_dir + r"\Validation\Images"
output_folder = parent_dir + r"\Validation\Output"
os.makedirs(output_folder, exist_ok=True)

# === Convert class mask to RGB using the color_map ===
def colorize_mask(mask, color_map):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        color_mask[mask == class_idx] = color
    return Image.fromarray(color_mask)

# === Preprocess an image for the model ===
def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0
    return img

# === Process all PNG images in folder ===
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

for filename in image_files:
    base_name = os.path.splitext(filename)[0]
    img_path = os.path.join(input_folder, filename)

    # Preprocess and predict
    img = preprocess_image(img_path)
    img = tf.expand_dims(img, axis=0)  # (1, 256, 256, 3)
    pred = model.predict(img)[0]       # (256, 256, 18)

    # Convert to class mask
    pred_mask = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)  # (256, 256)

    # # Save raw mask (class values)
    # raw_mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
    # Image.fromarray(pred_mask).save(raw_mask_path)

    # Save colorized mask
    colored_mask = colorize_mask(pred_mask, color_map)
    colored_mask_path = os.path.join(output_folder, f"{base_name}_colored.png")
    colored_mask.save(colored_mask_path)

    # print(f"Saved: {raw_mask_path}, {colored_mask_path}")
all_y_true = []
all_y_pred = []

for images, masks in val_dataset:
    preds = model.predict(images)                        # shape: (B, 256, 256, 18)
    preds = tf.argmax(preds, axis=-1).numpy()            # → shape: (B, 256, 256)
    masks = tf.squeeze(masks, axis=-1).numpy()           # shape: (B, 256, 256)

    all_y_true.append(masks)
    all_y_pred.append(preds)

all_y_true = np.concatenate(all_y_true, axis=0)
all_y_pred = np.concatenate(all_y_pred, axis=0)

ious = compute_per_class_iou(all_y_true, all_y_pred, num_classes=18)
for i, score in enumerate(ious):
    print(f"Class {i:2d} IoU: {score:.3f}")