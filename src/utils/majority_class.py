import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

labels_folder = {
    'double_plant': 1,
    'drydown': 2,
    'endrow': 3,
    'nutrient_deficiency': 4,
    'planter_skip': 5,
    'water': 6,
    'waterway': 7,
    'weed_cluster': 8
}

def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def get_major_class_label(data_root):
    basename = [img_basename(f) for f in os.listdir(os.path.join(data_root,'images/rgb'))]
    dataset_mapping = {'basename':[], 'origin_set':[], 'majority_class_label':[] }
    for file in tqdm(basename):
        dataset_mapping['basename'].append(file)
        dataset_mapping['origin_set'].append("train" if "train" in data_root else "val")    
        label_images = np.ndarray((9,512,512))
        background_array = np.zeros((512,512))
        label_images[0] = background_array
        for label_name, label_index in labels_folder.items():
            
            arr = np.array(cv2.imread(os.path.join(data_root, 'labels', label_name, f"{file}.png"), cv2.IMREAD_GRAYSCALE))
            label_images[label_index] = arr
        n_sum = np.argmax(np.sum(label_images,axis=(1,2)),axis=0)
        dataset_mapping['majority_class_label'].append(int(n_sum))
    return dataset_mapping

