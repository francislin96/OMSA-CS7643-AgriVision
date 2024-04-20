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

# change DATASET ROOT to your dataset path
#DATASET_ROOT = "\\Users\\sshel\\OneDrive\\Documents\\OMSA\\CS 7643\\Project\\OMSA-CS7643-AgriVision\\data\\images_2021"

args = parser.parse_args()
DATASET_ROOT = args.input_dir
TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
NEW_TRAIN_ROOT = os.path.join(DATASET_ROOT, 'new_train')
NEW_VAL_ROOT = os.path.join(DATASET_ROOT, 'new_val')
NEW_TEST_ROOT = os.path.join(DATASET_ROOT, 'new_test')


### First check if this script was already run and a file mapping the old datasets to classes exists
try:
    with open("dataset_map.json", "r") as file:
        in_dict = json.load(file)
        merged_dict = {'basename':[],'origin_set':[],'majority_class_label':[]}
        for i in in_dict.keys():
            merged_dict['basename'].append(i)
            merged_dict['origin_set'].append(in_dict[i]['origin_set'])
            merged_dict['majority_class_label'].append(in_dict[i]['majority_class_label'])
except:
    print("Getting majority class labels for train data set")
    train_dict = get_major_class_label(TRAIN_ROOT)
    print("Getting majority class labels for val data set")
    val_dict = get_major_class_label(VAL_ROOT)
    merged_dict = {x:train_dict[x]+val_dict[x] for x in train_dict.keys()}

    out_dict = {merged_dict['basename'][i]:{'origin_set':merged_dict['origin_set'][i], 'majority_class_label':merged_dict['majority_class_label'][i]} for i in range(len(merged_dict['basename']))}    

    with open("dataset_map.json", "w") as outfile:
            json.dump(out_dict, outfile)

reversed_keys = {v:k for k,v in labels_folder.items()}
print("Distribution of classes:")
c = Counter(merged_dict['majority_class_label'])
print({x[0]:x[1] for x in sorted({reversed_keys.get(i,'background'):round(c[i]/sum(c.values()),3) for i in c}.items(), key = lambda x:-x[1])})

X = merged_dict['basename']
y = merged_dict['majority_class_label']

### Uncomment if creating a small subset

# l = merged_dict['basename']
# i = list(enumerate(l))
# random.shuffle(i)
# idx, X = zip(*i[:2000]) 
# y = [merged_dict['majority_class_label'][i] for i in idx]

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