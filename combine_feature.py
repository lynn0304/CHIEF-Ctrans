import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_img_path', type=str, default='')
args = parser.parse_args()

#### PATH FOR PATCH FEATURE #########
PATCH_FEAT_PATH = './patch_feature'
WSI_PATH = args.wsi_img_path
PATCH_FEAT_COMBINE = './combined_patch_feature'
#####################################
if os.path.exists(PATCH_FEAT_COMBINE):
    pass
else:
	os.makedirs(PATCH_FEAT_COMBINE)

file_paths = os.listdir(PATCH_FEAT_PATH)

patch_list = []
for patch in os.listdir(WSI_PATH):
    wsi_id = patch.replace('.svs', '')
    if wsi_id not in patch_list:
        patch_list.append(wsi_id)


for wsi_id in patch_list:
    combined_features = []
    exists = False
    for file_path in file_paths:
        # Load the tensor from the .pt file
        if file_path.split('_')[0] == wsi_id:
            exists = True
            features = torch.load(PATCH_FEAT_PATH + '/' + file_path)
            # Ensure that the features tensor has the shape (patches, 768)
            if features.shape[1] == 768:
                combined_features.append(features)
            else:
                print(f"Warning: {file_path} does not have the correct shape (patches, 768). Skipping this file.")
    
    # Concatenate all feature tensors along the first axis (i.e., the patches dimension)
    if exists:
        combined_features_tensor = torch.cat(combined_features, dim=0)

        # Save the combined tensor to a new .pt file
        torch.save(combined_features_tensor, f'{PATCH_FEAT_COMBINE}/{wsi_id}.pt')

        print(f"Features successfully combined and saved as '{wsi_id}.pt'")


