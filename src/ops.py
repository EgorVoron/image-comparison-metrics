from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import typing as tp
import numpy as np
from math import sin, cos, pi, radians
from scipy import ndimage
import cv2
from tqdm import tqdm


def grayscale(img):
    return ImageOps.grayscale(img)


def distortion_factory(distortion_func: tp.Callable, params: list) -> dict[tp.Any, tp.Callable]:
    return {d: lambda img, d=d: distortion_func(img, d) for d in params}


def get_distortions(distorion_func: tp.Callable, params: list, images: list) -> dict[tp.Any, list]:
    return {d: [distorion_func(img, d) for img in tqdm(images)] for d in params}


def gaussian_blur(img: Image.Image, deviation: float = 7) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(radius=deviation))
    return blurred


def flip(img: Image.Image, where=0) -> Image.Image:
    if where == 0:
        return ImageOps.mirror(img)
    return ImageOps.flip(img)


def jpeg(img: Image.Image, quality=40) -> Image.Image:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", optimize=True, quality=quality)
    return Image.open(buffered)


def scale(img: Image.Image, factor) -> Image.Image:
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)))


def rotate(img: Image.Image, angle) -> Image.Image:
    rotated = img.rotate(angle, expand=True)
    w, h = rotated.size
    # return rotated.crop((int(w * cos(angle)), )))
    delta = int(h * sin(radians(angle)))
    coords = (delta, delta, w - delta, h - delta)
    # print(coords)
    return rotated.crop(coords)


def brighter(img: Image.Image, factor) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def sharper(img: Image.Image, factor) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def crop(img: Image.Image, percent) -> Image.Image:
    w, h = img.size
    return img.crop((percent * w, percent * h, (1 - percent) * w, (1 - percent) * h))


def sp_noise(image, prob):
    '''
    salt and pepper
    '''
    image = np.array(grayscale(image))
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return Image.fromarray(output)


def median_filter(image, size=7):
    image = np.array(image)
    # result = ndimage.median_filter(image, size=size)
    result = cv2.medianBlur(image, ksize=size)
    return Image.fromarray(result)

def add_watermark(background, foreground):
    # todo: different watermarks
    background.paste(foreground, (0, 0), foreground)
    background.show()