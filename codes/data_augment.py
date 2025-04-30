from draw_tools.plot import plot
import numpy as np
from skimage.transform import rotate, AffineTransform, warp

def augment_image(image):
    image = image.reshape(28, 28)
    # 平移操作
    tx = np.random.randint(-2, 3)
    ty = np.random.randint(-2, 3)
    transform = AffineTransform(translation=(tx, ty))
    translated = warp(image, transform, mode='constant', cval=0)

    # 旋转操作
    angle = np.random.choice([-20, -10, 10, 20])
    rotated = rotate(translated, angle, resize=False, mode='constant', cval=0)

    # 缩放操作
    scale = np.random.uniform(0.8, 1.2)
    scaled = warp(rotated, AffineTransform(scale=(scale, scale)), mode='constant', cval=0)

    return scaled.flatten()