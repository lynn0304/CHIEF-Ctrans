WSI_IMG_PATH='/path/for/your/wsi'
CONFIG_PATH='/path/for/your/config/file'

python extract_features.py --wsi_img_path $WSI_IMG_PATH
python combine_feature.py --wsi_img_path $WSI_IMG_PATH
python Get_CHIEF_WSI_level_feature_batch.py --config_path $CONFIG_PATH