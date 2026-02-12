import numpy as np
import cv2


def pixel_to_world(pixel_point, scale=0.5):
    """
    Простой перевод пикселей в мм.
    scale = мм на пиксель (калибруется отдельно)
    """
    x_pixel, y_pixel = pixel_point
    return x_pixel * scale, y_pixel * scale


def estimate_orientation_from_mask(mask):
    """
    Определение ориентации через PCA по маске
    """
    coords = np.column_stack(np.where(mask > 0))
    coords = np.float32(coords)

    if len(coords) < 5:
        return 0

    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return np.degrees(angle)
