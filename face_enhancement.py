'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import pathlib
import argparse
import numpy as np
import __init_paths

from tqdm import tqdm
from PIL import Image

from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from face_detect.retinaface_detection import RetinaFaceDetection
from align_faces import warp_and_crop_face, get_reference_facial_points

class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, use_sr=True, sr_model=None, channel_multiplier=2, narrow=1, device='cuda'):
        self.faceparser = FaceParse(base_dir, device=device)
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier, narrow, device=device)
        self.srmodel =  RealESRNet(base_dir, sr_model, device=device)

        self.use_sr = use_sr
        self.threshold = 0.9
        self.size = size

        # the mask for pasting restored faces back
        mask = np.zeros((512, 512), np.float32)
        mask = cv2.rectangle(mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(mask, (101, 101), 11)

        self.kernel = np.array(([0.0625, 0.125, 0.0625],
                                [0.1250, 0.250, 0.1250],
                                [0.0625, 0.125, 0.0625]), dtype=np.float32)

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points((self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, exterior_area=20):
        mask[:exterior_area, :] = 0
        mask[-exterior_area:, :] = 0
        mask[:, :exterior_area] = 0
        mask[:, -exterior_area:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = mask.astype(np.float32)
        return mask

    def process(self, img):
        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])

        facebs, landms = self.facedetector.detect(img)
        
        height, width = img.shape[:2]
        orig_faces, enhanced_faces = [], []
        final_img = np.zeros(img.shape, dtype=np.uint8)
        final_mask = np.zeros((height, width), dtype=np.float32)
        for face_box, facial5points in zip(facebs, landms):
            if face_box[4] > self.threshold:

                facial5points = np.reshape(facial5points, (2, 5))

                orig_face, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
                crop_enhanced_face = self.facegan.process(orig_face)

                orig_faces.append(orig_face)
                enhanced_faces.append(crop_enhanced_face)

                crop_face_mask = self.faceparser.process(crop_enhanced_face)[0] / 255.0
                crop_face_mask = self.mask_postprocess(crop_face_mask)
                crop_face_mask = cv2.resize(crop_face_mask, crop_enhanced_face.shape[:2][::-1])

                face_height = face_box[3] - face_box[1]
                face_width = face_box[2] - face_box[0]
                small_face = min(face_height, face_width) < 100
                if small_face:
                    crop_enhanced_face = cv2.filter2D(crop_enhanced_face, -1, self.kernel)

                frame_face_mask = cv2.warpAffine(crop_face_mask, tfm_inv, (width, height), flags=3)
                frame_enhanced_face = cv2.warpAffine(crop_enhanced_face, tfm_inv, (width, height), flags=3)

                face_area = (frame_face_mask - final_mask) > 0
                final_mask[face_area] = frame_face_mask[face_area]
                final_img[face_area] = frame_enhanced_face[face_area]

        final_mask = final_mask[:, :, None]
        img = img_sr if self.use_sr and img_sr is not None else img
        img = final_img * final_mask + img * (1.0 - final_mask)

        return img, orig_faces, enhanced_faces
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--size', type=int, default=512, help='resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    parser.add_argument('--use_cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--sr_model', type=str, default='rrdb_realesrnet_psnr', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--indir', type=str, default='examples/imgs', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-BFR', help='output folder')
    args = parser.parse_args()

    face_enhancer = FaceEnhancement(size=args.size,
                                    model=args.model,
                                    use_sr=args.use_sr,
                                    sr_model=args.sr_model,
                                    channel_multiplier=args.channel_multiplier,
                                    narrow=args.narrow,
                                    device='cuda' if args.use_cuda else 'cpu')

    os.makedirs(args.outdir, exist_ok=True)
    outdir = pathlib.Path(args.outdir)

    files = sorted(pathlib.Path(args.indir).glob('*.*g'))
    progress_bar = tqdm(files, desc="Enhancing Images", unit=" images")
    for index, file in enumerate(progress_bar):
        
        original_img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if not isinstance(original_img, np.ndarray):
            print(file.name, 'error'); continue

        updated_img, orig_faces, enhanced_faces = face_enhancer.process(original_img)
        # enhanced_filename = str(outdir / f"{file.stem}_enhanced.jpg")
        # cv2.imwrite(enhanced_filename, updated_img)

        comparision_filename = str(outdir / f"{file.stem}_comparision.jpg")
        updated_img = cv2.resize(updated_img, original_img.shape[:2][::-1])
        print(original_img.shape, updated_img.shape)
        comparision_img = np.concatenate((original_img, updated_img), axis=1)
        cv2.imwrite(comparision_filename, comparision_img)

        #for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
        #    of = cv2.resize(of, ef.shape[:2])
        #    cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'_face%02d'%m+'.jpg'), np.hstack((of, ef)))
