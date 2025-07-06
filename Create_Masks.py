import os
import cv2
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imshow
from scipy import ndimage
import albumentations as A

# Input and output base paths
input_base = r'C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Dados\Imagens_um_Alimentos'
output_base = r'C:\Users\acyrm\Documents\UTFPR\9Periodo\VisaoComputacional\Classification\Dados\Data'

for split in ['Training', 'Validation']:
    for sub in ['Images', 'Masks']:
        os.makedirs(os.path.join(output_base, split, sub),  exist_ok=True)


# Define thresholds per class (in HSV space)
CLASS_THRESHOLDS = {
    'Alface': {'h': (35, 85), 's': (100, 255), 'v': (50, 255), 'class_id': 2},              # Verde
    'Almondega': {'h': (0, 10), 's': (100, 255), 'v': (50, 255), 'class_id': 3},            # Marrom avermelhado
    'Arroz': {'h': (10, 50), 's': (0, 120), 'v': (140, 255), 'class_id': 4},                # Amarelo claro / Bege
    'BatataFrita': {'h': (10, 50), 's': (90, 255), 'v': (140, 255), 'class_id': 5},         # Amarelo forte
    'Beterraba': {'h': (140, 180), 's': (60, 255), 'v': (30, 255), 'class_id': 6},          # Roxo / Magenta escuro
    'BifeBovinoChapa': {'h': (0, 180), 's': (50, 255), 'v': (0, 255), 'class_id': 7},       # Marrom escuro
'CarneBovinaPanela': {'h': (0, 30), 's': (100, 255), 'v': (30, 200), 'class_id': 8},        # Marrom avermelhado
    'Cenoura': {'h': (0, 35), 's': (150, 255), 'v': (100, 255), 'class_id': 9},             # Laranja
    'FeijaoCarioca': {'h': (0, 45), 's': (80, 255), 'v': (50, 255), 'class_id': 10},        # Marrom claro
    'Macarrao': {'h': (0, 40), 's': (50, 255), 'v': (130, 255), 'class_id': 11},            # Amarelo claro / bege
    'Maionese': {'h': (0, 50), 's': (0, 170), 'v': (140, 255), 'class_id': 12},             # Quase branco/amarelado
    'PeitoFrango': {'h': (0, 50), 's': (0, 255), 'v': (0, 255), 'class_id': 13},            # Bege alaranjado
    'PureBatata': {'h': (10, 50), 's': (20, 255), 'v': (0, 230), 'class_id': 14},           # Bege
'StrogonoffCarne': {'h': (0, 40), 's': (80, 255), 'v': (120, 255), 'class_id': 15},         # Laranja rosado
'StrogonoffFrango': {'h': (0, 40), 's': (80, 255), 'v': (120, 255), 'class_id': 16},        # Laranja claro
    'Tomate': {'h': (0, 220), 's': (60, 255), 'v': (60, 255), 'class_id': 17}               # Vermelho forte
}

# Background and plate thresholds
PLATE_THRESHOLDS = {'h': (0, 180), 's': (0, 40), 'v': (130, 255)}         # Baixa saturação, alto brilho

# Color map for visualization
color_map = {
    0:  (0, 0, 0),         # Background - Preto
    1:  (255, 255, 255),   # Plate - Branco
    2:  (0, 255, 0),       # Alface - Verde
    3:  (255, 0, 0),       # Almondega - Vermelho
    4:  (255, 255, 0),     # Arroz - Amarelo
    5:  (255, 0, 255),     # BatataFrita - Magenta
    6:  (0, 255, 255),     # Beterraba - Ciano
    7:  (128, 0, 0),       # BifeBovinoChapa - Marrom escuro
    8:  (0, 128, 0),       # CarneBovinaPanela - Verde escuro
    9:  (0, 0, 128),       # Cenoura - Azul escuro
    10: (128, 128, 0),     # FeijaoCarioca - Oliva
    11: (128, 0, 128),     # Macarrao - Roxo
    12: (0, 128, 128),     # Maionese - Azul petróleo
    13: (192, 192, 192),   # PeitoFrango - Cinza claro
    14: (128, 128, 128),   # PureBatata - Cinza médio
    15: (255, 165, 0),     # StrogonoffCarne - Laranja
    16: (255, 20, 147),    # StrogonoffFrango - Rosa choque
    17: (0, 0, 255)        # Tomate - Azul
}

STANDARD_SIZE = (256, 256)  # Width, Height

def clear_mask_edges(mask):
    mask[0, :] = 0         # Top edge
    mask[-1, :] = 0        # Bottom edge
    mask[:, 0] = 0         # Left edge
    mask[:, -1] = 0        # Right edge
    return mask

def fill_small_holes(mask, target_class, max_hole_size):
    """
    Fill small holes (areas with different class inside target_class) below a size threshold.
    """
    # Create binary mask for the target class
    binary = (mask == target_class).astype(np.uint8)
    
    # Invert: we want to find "holes" (areas NOT target class inside target class)
    inverted = 1 - binary
    
    # Label connected components in the inverted area
    labeled, num_features = ndimage.label(inverted)
    
    # Measure size of each hole
    sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
    
    # Iterate through holes
    for i, size in enumerate(sizes):
        if size <= max_hole_size:
            # Fill hole: set back to target_class
            mask[labeled == (i + 1)] = target_class

    return mask

def save_image_and_mask(image_np, base_name, split, mask=None):
    if mask is None:
        mask = create_mask(image_np, params, params['class_id'])

    image_output_path = os.path.join(output_base, split, 'Images', f"{base_name}.png")
    Image.fromarray(image_np).save(image_output_path, format="PNG", quality=95)

    mask_output_path = os.path.join(output_base, split, 'Masks', f"{base_name}.png")
    Image.fromarray(mask.astype(np.uint8)).save(mask_output_path, format="PNG")

    # Sanity check
    unique_classes = np.unique(mask)
    saved_mask = np.array(Image.open(mask_output_path))
    saved_unique_classes = np.unique(saved_mask)
    if not np.array_equal(unique_classes, saved_unique_classes):
        print(f"[ERROR] Unique classes mismatch for {base_name}: {unique_classes} vs {saved_unique_classes}")

def augment_image_and_mask(image, mask):
    # Horizontal flip
    if random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Vertical flip
    if random.random() < 0.3:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # Rotation (90, 180, 270 degrees)
    if random.random() < 0.3:
        k = random.choice([1, 2, 3])
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    # Brightness jitter via HSV
    if random.random() < 0.4:
        factor = 0.7 + 0.6 * random.random()  # [0.7–1.3]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 2] *= factor
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Gamma correction
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

    # Gaussian noise
    if random.random() < 0.2:
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # CLAHE (contrast enhancement)
    if random.random() < 0.3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image, mask

def colorize_mask(mask, color_map):
    """
    Converte uma máscara de classes (valores inteiros) em uma imagem RGB com cores por classe.

    Parâmetros:
    - mask: np.ndarray (2D) contendo valores inteiros de classe (ex: 0, 1, 2, ...)
    - color_map: dict {classe_id: (R, G, B)} definindo a cor para cada classe

    Retorno:
    - color_mask: np.ndarray (H, W, 3) imagem RGB colorida
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color

    return color_mask

def create_mask(img_array, food_params, food_class_id):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)  # Convert to HSV

    h, s, v = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

    # Plate condition
    plate_cond = (h >= PLATE_THRESHOLDS['h'][0]) & (h <= PLATE_THRESHOLDS['h'][1]) & \
                 (s >= PLATE_THRESHOLDS['s'][0]) & (s <= PLATE_THRESHOLDS['s'][1]) & \
                 (v >= PLATE_THRESHOLDS['v'][0]) & (v <= PLATE_THRESHOLDS['v'][1])

    # Create mask
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    mask[plate_cond] = 1
    mask = clear_mask_edges(mask)  # Clear edges

    # Fill small black holes inside background (limit to small size)
    mask = fill_small_holes(mask, target_class=0, max_hole_size=5000)  # Example size threshold (adjust)

    # Fill small black holes inside plate (limit to small size)
    mask = fill_small_holes(mask, target_class=1, max_hole_size=15000)  # Example size threshold (adjust)

    # Food condition
    food_cond = (h >= food_params['h'][0]) & (h <= food_params['h'][1]) & \
                (s >= food_params['s'][0]) & (s <= food_params['s'][1]) & \
                (v >= food_params['v'][0]) & (v <= food_params['v'][1]) & \
                (mask == 1)  # Only consider food in plate areas
    
    mask[food_cond] = food_class_id
    
    # Optionally: small holes in food areas
    if food_class_id > 1:
        final_mask = fill_small_holes(mask, target_class=food_class_id, max_hole_size=1000)  # Optional
    # mask[mask == food_class_id] = 2

    return final_mask

train_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.3),
    A.Blur(p=0.2),
    A.GaussNoise(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])





for class_name, params in CLASS_THRESHOLDS.items():
    folder_path = os.path.join(input_base, class_name)
    if not os.path.isdir(folder_path):
        print(f"Skipping {class_name}: folder not found.")
        continue

    # Get image files
    print(f"Processing class: {class_name}")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_files, val_files = train_test_split(files, test_size=0.1, random_state=42)
    print(f"Found {len(files)} images for {class_name}. Training: {len(train_files)}, Validation: {len(val_files)}")


    def process_and_save(file, split):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(STANDARD_SIZE, Image.BILINEAR)
        img_np = np.array(img)

        # Generate original mask (unaugmented)
        base_name = f"{class_name}_{os.path.splitext(file)[0]}"
        mask = create_mask(img_np, params, params['class_id'])

        # Save the original image + mask
        save_image_and_mask(img_np, base_name, split, mask)

        # # If training, save 2 augmented versions
        # if split == 'Training':
        #     for i in range(2):
        #         # Apply albumentations to both image and mask
        #         augmented = train_transform(image=img_np, mask=mask)
        #         aug_img = augmented['image']
        #         aug_mask = augmented['mask']

        #         # Save the augmented pair
        #         save_image_and_mask(aug_img, f"{base_name}_aug{i+1}", split, aug_mask)



    for file in train_files:
        # print(f"Processing {file} for {class_name} in Training set")
        process_and_save(file, 'Training')
    for file in val_files:
        # print(f"Processing {file} for {class_name} in Validation set")
        process_and_save(file, 'Validation')