from skimage import io
from skimage import transform as sktransform
import numpy as np
import json
from PIL import Image
from imgaug import augmenters as iaa
from mask import facehull
import cv2
import tqdm
import pickle
import random
from datetime import datetime
from tqdm import tqdm
import time
import argparse


def total_euclidean_distance(a, b):
    assert len(a.shape) == 2
    try:
        distance = np.sum(np.linalg.norm(a-b, axis=1))
    except:
        distance = 99999999
    return distance


def random_get_hull(landmark, img1):
    mask = facehull(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
    return mask/255


def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
   
    maskIndices = np.where(mask != 0)
    
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class BIOnlineGeneration:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.landmarks_record_json = json.load(f)
            self.landmarks_record = {}
            for k, v in self.landmarks_record_json.items():
                self.landmarks_record[k] = np.array(v)
        
        self.data_list = list(self.landmarks_record.keys())   # number of images in the data
        print(len(self.data_list))
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        self.search_sample_k = int(len(self.data_list)/10)  # k = no. of images/10
        self.search_sample_k = int(len(self.data_list))  # coz abhi utni images nhi hai

    def gen_one_datapoint(self):
        background_face_path = random.choice(self.data_list)
        num = random.randint(0, 1)
        
        if not num:
            face_img, mask = self.get_blended_face(background_face_path)
            mask = (1 - mask) * mask * 4
        else:
            face_img = io.imread(background_face_path)
            mask = np.zeros((600, 600, 1))
        
        face_img = face_img[100:600,50:550,:]
        mask = mask[100:600,50:550,:]
            
        return face_img,mask,data_type
        
    def get_blended_face(self,background_face_path):
        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_path]
        
        foreground_face_path = self.search_similar_face(background_landmark,background_face_path)
        foreground_face = io.imread(foreground_face_path)

        foreground_face = sktransform.resize(foreground_face, background_face.shape, preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
       
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        # filte empty mask after deformation
        if np.sum(mask) == 0 :
            return foreground_face,mask

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
       
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face,(600,600),preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask,(600,600),preserve_range=True)
        mask = mask[:,:,0:1]
        return blended_face,mask
    
    def search_similar_face(self,this_landmark,background_face_path):
        
        # loop throungh all candidates frame to get best match             THIS will be needed
        all_candidate_path = random.sample( self.data_list, k=self.search_sample_k)
        all_candidate_path = list(all_candidate_path)
        min_path = background_face_path
        min_dist = 99999999

        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(np.float32)
            candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
            if (candidate_distance < min_dist) and (candidate_path != background_face_path):
                min_dist = candidate_distance
                min_path = candidate_path

        return min_path

def generate_blends(obj, path_suffix='', valid=False):
    list_size = len(obj.data_list)
    count = 1
    img_paths = []
    mask_paths = []
    img_labels = []
    if valid:
        extension = '_valid'
    else:
        extension = ''
    for i in tqdm(range(list_size*5)):
        img,mask,label = obj.gen_one_datapoint()
        mask = np.repeat(mask,3,2)
        mask = (mask*255).astype(np.uint8)
        # save masks
        mask_img = Image.fromarray(mask, 'RGB')
        filename = 'blend_dataset' + extension +'/' + label + '_masks' + str(count) + '.jpg'
        mask_img.save(filename)
        mask_paths.append(filename)

        # save imgs
        face_img = Image.fromarray(img, 'RGB')
        filename = 'blend_dataset' + extension + '/' + label + '_imgs' + str(count) + '.jpg'
        face_img.save(filename)
        img_paths.append(filename)

        img_labels.append(label)
        count += 1
    all_paths = {'image paths' : img_paths, 'mask paths' : mask_paths, 'image labels' : img_labels}
    if valid:
        save_name = 'blend_dataset' + extension + '/'+'all_paths_valid_'+path_suffix+'.pkl'
    else:
        save_name = 'blend_dataset' + extension + '/'+'all_paths_'+path_suffix+'.pkl'
    with open(save_name, 'wb') as alp:
        pickle.dump(all_paths, alp)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate face blends")
    parser.add_argument("--path", required=True, help="Path to landmarks.json file")
    args = parser.parse_args()
    t = time.time()
    ds = BIOnlineGeneration(args.path)
    generate_blends(ds, path_suffix=args.path[-6])
    print("Process completed in hours : " + str((time.time() - t) / (60*60)))
