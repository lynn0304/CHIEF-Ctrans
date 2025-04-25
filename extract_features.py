import openslide
import cv2 
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tifffile as tiff
import os
import shutil
import subprocess
import argparse


def calculate_background_ratio(image, threshold=220):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    total_pixels = binary.size
    background_pixels = np.count_nonzero(binary)
    background_ratio = (background_pixels / total_pixels) * 100

    return background_ratio



def process_patch(x, y, imgsz, org_img, wsi_id):
    cropped_image = np.asarray(
        org_img.read_region((x, y), 0, (imgsz, imgsz)), dtype=np.uint8)
    bg_ratio = calculate_background_ratio(cropped_image)
    if bg_ratio < 70 and cropped_image.shape == (imgsz, imgsz, 4):
        tiff.imwrite('./patches/'+wsi_id+'_'+str(x)+'_'+str(y)+'.tif', cropped_image, compression='deflate')

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_img_path', type=str, default='')
parser.add_argument('--wsi_feature_path', type=str, default='')
args = parser.parse_args()

#### CHANGE HERE ########## 
WSI_PATH = args.wsi_img_path # where wsi is 
PATCH_FEAT_COMBINE = './combined_patch_feature'
#####################################
if os.path.exists(PATCH_FEAT_COMBINE):
    pass
else:
	os.makedirs(PATCH_FEAT_COMBINE)
###########################
all_wsi_path = os.listdir(WSI_PATH)
for wsi_path_name in all_wsi_path:
    if os.path.exists('./patches'):
        shutil.rmtree('./patches')
    os.makedirs('patches')
    ## just in case if the process stopped unexpectedly it will start over skipping finished ##
    finish_wsi = os.listdir(PATCH_FEAT_COMBINE)
    finish_wsi_id = ['TCGA-UF-A71E-01Z-00-DX1.030F4733-5EBE-4E86-AC31-9A3C96A65BF5']
    for i in range(len(finish_wsi)):
        finish_wsi_id.append(finish_wsi[i].replace('.pt', ''))
        #print(finish_wsi_id)
    wsi_path = WSI_PATH + '/' + wsi_path_name
    print('processing...', wsi_path)
    if wsi_path_name.replace('.svs', '') in finish_wsi_id:
    	print('finished', wsi_path_name)
    	continue
    ## just in case if the process stopped unexpectedly it will start over skipping finished ##
    
    # process patches 2048*2048
    imgsz = 2048
    wsi_id = os.path.basename(wsi_path).replace('.svs', '')
    org_img = openslide.OpenSlide(wsi_path)
    org_img_w, org_img_h = org_img.level_dimensions[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_patch, x, y, imgsz, org_img, wsi_id)
            for y in range(0, org_img_h-imgsz, int(imgsz))
            for x in range(0, org_img_w-imgsz, int(imgsz))
        ]
    # GET patch level features
    ######### CHANGE WHERE THE py FILE IS ###############
    subprocess.run(["python", "-u", "/mnt/data/Desktop/CHIEF/CHIEF/Get_CHIEF_patch_feature.py"])
