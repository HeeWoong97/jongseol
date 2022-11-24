import os
from glob import glob
import json
from tqdm import tqdm

data_path = "../../../data/cross/dataset"
img_list = glob(os.path.join(data_path, "valid", "images", "*.png"))
file_list = []
print(len(img_list))

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

        move(img_path, os.path.join(DATA_PATH, 'valid', 'images', os.path.basename(img_path)))

        with open(os.path.join(DATA_PATH, 'valid', 'labels', os.path.basename(json_path).replace('.json', '.txt')), 'w') as f:
            f.write(txt)
        
        file_list.append(os.path.join('valid', 'images', os.path.basename(img_path)))
    except Exception as e:
        print(e, img_path)
    
with open(os.path.join(DATA_PATH, 'valid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_list) + '\n')
        
print(len(file_list))