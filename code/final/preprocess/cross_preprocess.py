import os
from glob import glob
import json
from shutil import move
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class_names = ['Zebra_Cross', 'R_Signal', 'G_Signal']

ORIGIN_PATH = "../../../data/cross/origin/"
OUTPUT_PATH = "../../../data/cross/dataset/"

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

    try:
      with open(json_path, 'r') as f:
          data = json.load(f)
    except Exception as e:
      continue

    w = data['imageWidth']
    h = data['imageHeight']
    
    txt = ''
    
    try:
        for shape in data['shapes']:
            label = shape['label']

            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            cx = (x1 + x2) / 2. / w
            cy = (y1 + y2) / 2. / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label = class_names.index(shape['label'])

            txt += '%d %f %f %f %f\n' % (label, cx, cy, bw, bh)

        move(img_path, os.path.join(OUTPUT_PATH, 'train', 'images', os.path.basename(img_path)))

        with open(os.path.join(OUTPUT_PATH, 'train', 'labels', os.path.basename(json_path).replace('.json', '.txt')), 'w') as f:
            f.write(txt)
        
        file_list.append(os.path.join('train', 'images', os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)
    
with open(os.path.join(OUTPUT_PATH, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
print("Finish")
print()

print('[Convert valid set]')
file_list = []
for img_path in tqdm(val_img_list):
    json_path = img_path.replace('.jpg', '.json') if img_path.find('.jpg') is not -1 else img_path.replace('.png', '.json')

    try:
      with open(json_path, 'r') as f:
          data = json.load(f)
    except Exception as e:
      continue

    w = data['imageWidth']
    h = data['imageHeight']
    
    txt = ''
    
    try:
        for shape in data['shapes']:
            label = shape['label']

            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            cx = (x1 + x2) / 2. / w
            cy = (y1 + y2) / 2. / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label = class_names.index(shape['label'])

            txt += '%d %f %f %f %f\n' % (label, cx, cy, bw, bh)

        move(img_path, os.path.join(OUTPUT_PATH, 'valid', 'images', os.path.basename(img_path)))

        with open(os.path.join(OUTPUT_PATH, 'valid', 'labels', os.path.basename(json_path).replace('.json', '.txt')), 'w') as f:
            f.write(txt)
        
        file_list.append(os.path.join('valid', 'images', os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)
    
with open(os.path.join(OUTPUT_PATH, 'valid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
print("Finish")