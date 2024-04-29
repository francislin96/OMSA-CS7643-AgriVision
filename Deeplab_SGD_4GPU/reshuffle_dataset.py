import os
import random
import shutil
from argparse import ArgumentParser
from tqdm import tqdm
from src.utils.majority_class import get_major_class_label
from sklearn.model_selection import train_test_split
from collections import Counter
import json

random.seed(563798)

parser = ArgumentParser("Reshuffle the 2021 Agrivision Dataset to create labeled test set")
parser.add_argument('input_dir', type=str, help='The root directory containing the dataset')

# change DATASET ROOT to your dataset path
#DATASET_ROOT = "\\Users\\sshel\\OneDrive\\Documents\\OMSA\\CS 7643\\Project\\OMSA-CS7643-AgriVision\\data\\images_2021"

args = parser.parse_args()
DATASET_ROOT = args.input_dir
TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
NEW_TRAIN_ROOT = os.path.join(DATASET_ROOT, 'new_train')
NEW_VAL_ROOT = os.path.join(DATASET_ROOT, 'new_val')
NEW_TEST_ROOT = os.path.join(DATASET_ROOT, 'new_test')

print("Getting majority class labels for train data set")
train_dict = get_major_class_label(TRAIN_ROOT)
print("Getting majority class labels for val data set")
val_dict = get_major_class_label(VAL_ROOT)

merged_dict = {x:train_dict[x]+val_dict[x] for x in train_dict.keys()}
print(Counter(merged_dict['majority_class_label']))

with open("dataset_map.json", "w") as outfile:
        json.dump(merged_dict, outfile)


X = merged_dict['basename']
y = merged_dict['majority_class_label']

train_files, X_tt, _, y_tt = train_test_split(X, y, test_size=0.4, stratify=y)
val_files, test_files, _, _ = train_test_split(X_tt,y_tt, test_size=0.5, stratify=y_tt)


subfolders = ['boundaries','images/nir','images/rgb','labels/double_plant','labels/drydown',
              'labels/endrow','labels/nutrient_deficiency','labels/planter_skip','labels/storm_damage',
              'labels/water','labels/waterway','labels/weed_cluster','masks']


def create_dirtree_without_files(src, dst):
    src = os.path.abspath(src)    
    src_prefix = len(src) + len(os.path.sep)
    os.makedirs(dst,exist_ok=True)
    for root, dirs, _ in os.walk(src):
        for dirname in dirs:
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
            os.makedirs(dirpath,exist_ok=True)


def move_files(files, new_root):
    
    for filename in tqdm(files):
        #basename = filename.split(".")[0]
        for folder in subfolders:
            if "images" in folder:
                try:
                    shutil.copy(os.path.join(f"{TRAIN_ROOT}\{folder}", f"{filename}.jpg"), os.path.join(f"{new_root}\{folder}", f"{filename}.jpg"))
                except:
                    shutil.copy(os.path.join(f"{VAL_ROOT}\{folder}", f"{filename}.jpg"), os.path.join(f"{new_root}\{folder}", f"{filename}.jpg"))
            else:
                try:
                    shutil.copy(os.path.join(f"{TRAIN_ROOT}\{folder}", f"{filename}.png"), os.path.join(f"{new_root}\{folder}", f"{filename}.png"))
                except:
                    shutil.copy(os.path.join(f"{VAL_ROOT}\{folder}", f"{filename}.png"), os.path.join(f"{new_root}\{folder}", f"{filename}.png"))


def main():
    
    print("Creating the subfolder structure for new train, val and test sets")
    
    for i,v in zip([TRAIN_ROOT,VAL_ROOT,VAL_ROOT],[NEW_TRAIN_ROOT,NEW_VAL_ROOT,NEW_TEST_ROOT]):
        create_dirtree_without_files(i,v)
    
    print("Copying train files")
    move_files(train_files, NEW_TRAIN_ROOT)
    print("Copying val files")
    move_files(val_files, NEW_VAL_ROOT)
    print("Copying test files")
    move_files(test_files, NEW_TEST_ROOT)
    

if __name__ == '__main__':
    main()