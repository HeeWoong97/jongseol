import os
from glob import glob
import json
from shutil import move
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class_names = ['Car', 'Pedestrian']

ORIGIN_PATH = "../../../data/car_pedestrian/origin/"
OUTPUT_PATH = "../../../data/car_pedestrian/dataset/"

jpg_list = glob(os.path.join(ORIGIN_PATH, '*', '*', '*.jpg'))
png_list = glob(os.path.join(ORIGIN_PATH, '*', '*', '*.png'))
img_list = jpg_list + png_list
print('[Origin dataset is ', len(img_list), 'images]')
print()

os.makedirs(os.path.join(OUTPUT_PATH, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'valid', 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'valid', 'labels'), exist_ok=True)

print('[Split dataset into train, valid set]')
train_img_list, val_img_list = train_test_split(img_list, test_size=0.4, random_state=2022)
print('Train:', len(train_img_list), 'Valid:' + len(val_img_list))
print()

print('[Convert train set]')
file_list = []
for img_path in tqdm(train_img_list):
    json_path = img_path.replace('.jpg', '.json') if img_path.find('.jpg') is not -1 else img_path.replace('.png', '.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    width = int(data["camera"]["resolution_width"])
    height = int(data["camera"]["resolution_height"])

    txt = ""

    try:
        for ann in data["annotations"]:
            points = ann["points"]

            top_left = points[0]
            top_right = points[1]
            bottom_right = points[2]
            bottom_left = points[3]

            center_x = (top_left[0] + top_right[0]) / 2. / width
            center_y = (top_left[1] + bottom_left[1]) / 2. / height
            bounding_width = (top_right[0] - top_left[0]) / width
            bounding_height = (bottom_left[1] - top_left[1]) / height

            label = 0 if ann["label"] == "보행자" else 1

            txt += '%d %.6f %.6f %.6f %.6f\n' % (label, center_x, center_y, bounding_width, bounding_height)

        with open(os.path.join(OUTPUT_PATH, "train", "labels", os.path.basename(json_path).replace(".json", ".txt")), "w") as f:
            f.write(txt)
        
        file_list.append(os.path.join(OUTPUT_PATH, "train", "images", os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)

with open(os.path.join(OUTPUT_PATH, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
print("Finish")
print()        

print('[Convert valid set]')
file_list = []
for img_path in tqdm(img_list):
    json_path = img_path.replace('.jpg', '.json') if img_path.find('.jpg') is not -1 else img_path.replace('.png', '.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    width = int(data["camera"]["resolution_width"])
    height = int(data["camera"]["resolution_height"])

    txt = ""

    try:
        for ann in data["annotations"]:
            points = ann["points"]

            top_left = points[0]
            top_right = points[1]
            bottom_right = points[2]
            bottom_left = points[3]

            center_x = (top_left[0] + top_right[0]) / 2. / width
            center_y = (top_left[1] + bottom_left[1]) / 2. / height
            bounding_width = (top_right[0] - top_left[0]) / width
            bounding_height = (bottom_left[1] - top_left[1]) / height

            label = 0 if ann["label"] == "보행자" else 1

            txt += '%d %.6f %.6f %.6f %.6f\n' % (label, center_x, center_y, bounding_width, bounding_height)

        with open(os.path.join(OUTPUT_PATH, "valid", "labels", os.path.basename(json_path).replace(".json", ".txt")), "w") as f:
            f.write(txt)
        
        file_list.append(os.path.join(OUTPUT_PATH, "valid", "images", os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)

with open(os.path.join(OUTPUT_PATH, "valid.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(file_list) + "\n")
print("Finish")