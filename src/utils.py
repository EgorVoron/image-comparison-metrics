from PIL import Image
import os


def get_image_from_dir(dataset_path: str):
    for img_filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_filename)
        yield Image.open(img_path)


def load_images(dataset_path: str, limit: int):
    images = []
    for idx, img_filename in enumerate(os.listdir(dataset_path)):
        if idx > limit:
            break
        img_path = os.path.join(dataset_path, img_filename)
        images.append(Image.open(img_path))
    return images
