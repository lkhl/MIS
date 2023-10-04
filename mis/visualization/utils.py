import cv2
import numpy as np


def draw_contour(img, mask, color=(253, 211, 106), thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img = cv2.drawContours(img, contours, -1, color=color, thickness=thickness)
    return img


def draw_mask(img, mask, opacity=0.6):
    mask = (mask > 0)[..., None]
    img = img * mask + img * ~mask * (1 - opacity)
    return img.astype(np.uint8)


def draw_click(img, clicks, radius=8):
    for click in clicks:
        color = (146, 208, 80) if click.is_positive else (192, 0, 0)
        coords = (click.coords[1], click.coords[0])
        img = cv2.circle(img.copy(), coords, int(radius * 1.5), (0, 0, 0), -1)
        img = cv2.circle(img, coords, radius, color, -1)
    return img


def apply_color_map(mask, color_map):
    return color_map[mask.flatten()].reshape(*mask.shape, 3)


def draw_region(img, mask, color_map, opacity=0.6):
    return cv2.addWeighted(img, 1 - opacity, apply_color_map(mask, color_map), opacity, 0)


def random_color_map(num_colors, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    color_map = np.random.rand(num_colors, 3) * 255
    return color_map.astype(np.uint8)
