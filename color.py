# Example Usage from terminal: python color.py ./pics/r1-min.jpg ./pics/r2-min.jpg 0.1

import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
from PIL import Image
import argparse

def is_too_bright(pixel):
    # Filters out too bright pixels which can skew mean RGB values
    if pixel[0] < 250 and pixel[1] < 250 and pixel[2] < 250:
        return False
    else:
        return True

def is_too_dark(pixel):
    # Filters out too dark pixels which can skew mean RGB values
    if pixel[0] > 5 and pixel[1] > 5 and pixel[2] > 5:
        return False
    else:
        return True

def color_info(image, normalize=False):
    # Returns mean values of reds, greens and blues in an image

    r = 0.
    g = 0.
    b = 0.
    total_color_pixels = 0.

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):

            pixel = image[row][col]

            if not is_too_bright(pixel) and not is_too_dark(pixel):
                total_color_pixels += 1
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]

    if normalize:
        total = r + g + b
        r = r / total
        g = g / total
        b = b / total
        return [r,g,b]

    else:
        return [r/total_color_pixels,g/total_color_pixels,b/total_color_pixels]

def chromatic_match(im1, im2, tolerance):
    # Executes the image color comparison pipeline

    if type(im1) == 'str':
        im1 = cv2.imread(im1)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.imread(im2)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

        im1 = np.array(Image.fromarray(im1).resize((100,120)))[20:]
        im2 = np.array(Image.fromarray(im2).resize((100,120)))[20:]
    
    else:
        im1 = np.array(im1.resize((100,120)))[20:]
        im2 = np.array(im2.resize((100,120)))[20:]

    r1,g1,b1 = color_info(np.array(im1))
    r2,g2,b2 = color_info(np.array(im2))
    
    tr = tg = tb = False

    rn1, gn1 , bn1 = color_info(np.array(im1), normalize=True)
    rn2, gn2 , bn2 = color_info(np.array(im2), normalize=True)

    rn = abs(rn1-rn2)/rn1
    gn = abs(gn1-gn2)/gn1
    bn = abs(bn1-bn2)/bn1
    total_deviation = rn+gn+bn

    match = total_deviation < tolerance

    return match


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process inputs')

    parser.add_argument('source_path', default='', help='Path to source(first) Image')
    parser.add_argument('sample_path', default='', help='Path to sample(second) Image')
    parser.add_argument('tolerance', type = float, default=float(0.1), help='Error tolerance')
    args=parser.parse_args()

    chrotmatic_match(args.source_path, args.sample_path, args.tolerance)