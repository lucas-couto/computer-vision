import os
import random
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_images(config):
    input_shape = config['model']['input_shape']
    valid_size = config['training']['valid_size']
    train_size = config['training']['train_size']
    input_dir, output_dir = config['paths']['images_dir'], config['paths']['datasets_dir']
    image_size = (input_shape[0], input_shape[1])

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    
    real_images_dir = os.path.join(input_dir, 'real')
    fake_images_dir = os.path.join(input_dir, 'fake')

    real_images = [os.path.join('real', f) for f in os.listdir(real_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [os.path.join('fake', f) for f in os.listdir(fake_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    images = real_images + fake_images
    random.shuffle(images)

    train_images, val_images = train_test_split(images, test_size=valid_size, train_size=train_size, random_state=42)

    for img in train_images:
        try:
            img_path = os.path.join(input_dir, img)
            image = Image.open(img_path)
            image = image.resize(image_size)
            output_path = os.path.join(output_dir, 'train', img)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {e}")

    for img in val_images:
        try:
            img_path = os.path.join(input_dir, img)
            image = Image.open(img_path)
            image = image.resize(image_size)

            output_path = os.path.join(output_dir, 'valid', img)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {e}")
