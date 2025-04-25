import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.ctran import ctranspath
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'/mnt/data/Desktop/CHIEF/CHIEF/model_weight/CHIEF_CTransPath.pth')
model.load_state_dict(td['model'], strict=True)
model.eval()
######## CHANGE PATCH PATH ##########
IMG_PATH = './patches'
DES_PATH = './patch_feature' # des folder for patch feature
#####################################
if os.path.exists('./patch_feature'):
    pass
else:
	os.makedirs('./patch_feature')

all_img_path = os.listdir(IMG_PATH)
for img_id in all_img_path:
	img_path = IMG_PATH + '/'+img_id
	print('extracting ',img_id) 
	pt_name = img_id.replace('.tif', '')
	try:
		image = Image.open(img_path).convert('RGB')
		image = trnsfrms_val(image).unsqueeze(dim=0)
		with torch.no_grad():
			patch_feature_emb = model(image)
			torch.save(patch_feature_emb, f'{DES_PATH}/{pt_name}.pt')
	except (IOError, OSError) as e:
		print('skipping ', img_id)

