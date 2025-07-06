import os
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 3

parent_dir = r"C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Dados\Data_3classes"
model_path = r"C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Unet_GPU_binary.h5"


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def upsample_concat_block(x, skip, filters):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x
    
def build_unet_binary(input_shape=(256, 256, 3)):
    # Decoder
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def upsample_concat_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x
    
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

    x = conv_block(bottleneck, 512)
    x = upsample_concat_block(x, skips[3], 256)
    x = upsample_concat_block(x, skips[2], 128)
    x = upsample_concat_block(x, skips[1], 64)
    x = upsample_concat_block(x, skips[0], 32)

    x = layers.UpSampling2D((2, 2))(x)  # back to 256x256
    output = output = layers.Conv2D(3, (1, 1), activation='softmax')(x)

    return models.Model(inputs, output)

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
    0.05,  # 0 = background
    0.1,   # 1 = plate
    *[5.0] * 16  # 2 to 17 = food classes
], dtype=tf.float32)

def weighted_sparse_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_flat = tf.reshape(y_true, [-1])             # (batch*H*W,)
    y_pred_flat = tf.reshape(y_pred, [-1, 3])         # (batch*H*W, 18)

    weights = tf.gather(class_weights, y_true_flat)    # (batch*H*W,)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(ce * weights)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_true = tf.one_hot(y_true, 3)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    return 0.8 * weighted_sparse_ce(y_true, y_pred) + 0.2 * dice_loss(y_true, y_pred)

class_weights = tf.constant([
    0.5,  # 0 = background
    0.5,   # 1 = plate
    1.0  # 2 to 17 = food classes
], dtype=tf.float32)

def weighted_sparse_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_flat = tf.reshape(y_true, [-1])             # (batch*H*W,)
    y_pred_flat = tf.reshape(y_pred, [-1, 3])         # (batch*H*W, 18)

    weights = tf.gather(class_weights, y_true_flat)    # (batch*H*W,)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(ce * weights)

model = build_unet_binary(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss=weighted_sparse_ce, metrics=['accuracy'])


model.fit(train_dataset, validation_data=val_dataset, epochs=2)

model.save(model_path)

model = load_model(model_path, custom_objects={
    'combined_loss': combined_loss,
    'weighted_sparse_ce': weighted_sparse_ce,
    'dice_loss': dice_loss
})


input_folder = parent_dir + r"\Training\Images"
output_folder = parent_dir + r"\Training\Output"
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

    # Save raw mask (class values)
    raw_mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
    Image.fromarray(pred_mask).save(raw_mask_path)

    # Save colorized mask
    colored_mask = colorize_mask(pred_mask, color_map)
    colored_mask_path = os.path.join(output_folder, f"{base_name}_colored.png")
    colored_mask.save(colored_mask_path)

    print(f"Saved: {raw_mask_path}, {colored_mask_path}")