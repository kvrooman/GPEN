'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import pathlib
import numpy as np
import __init_paths

from PIL import Image
from face_model.face_gan import FaceGAN

class FaceColorization(object):
    def __init__(self, base_dir='./', size=1024, model=None, channel_multiplier=2):
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)

    # make sure the face image is well aligned. Please refer to face_enhancement.py
    def process(self, gray):
        # colorize the face
        out = self.facegan.process(gray)

        return out

if __name__=='__main__':
    model = {'name':'GPEN-Colorization-1024', 'size':1024}
    
    indir = pathlib.Path('examples') / 'grays'
    outdir = pathlib.Path('examples') / 'outs-colorization'
    os.makedirs(outdir, exist_ok=True)

    facecolorizer = FaceColorization(size=model['size'], model=model['name'], channel_multiplier=2)

    files = sorted(indir.glob('*.*g'))
    for n, file in enumerate(files):
        gray_img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        color_img = facecolorizer.process(gray_img)
        gray_img = cv2.resize(gray_img, colorf.shape[:2])
        combined_img = np.hstack((gray_img, color_img))

        filename = str(outdir / f"{file.stem}.jpg")
        cv2.imwrite(filename, combined_img)

        if n%10==0: print(n, file)
