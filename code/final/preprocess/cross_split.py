import os
from glob import glob
from shutil import move
from tqdm import tqdm
from tqdm import tqdm

# origin dataset
DATA_PATH = "../../data/cross/dataset"

TRAIN_PATH = os.path.join(DATA_PATH, "train")
TRAIN_IMG = os.path.join(TRAIN_PATH, "images")
TRAIN_LABEL = os.path.join(TRAIN_PATH, "labels")

VALID_PATH = os.path.join(DATA_PATH, "valid")
VALID_IMG = os.path.join(VALID_PATH, "images")
VALID_LABEL = os.path.join(VALID_PATH, "labels")

# dataset1
DATA_PATH1 = "../../data/cross/dataset1"

TRAIN_PATH1 = os.path.join(DATA_PATH1, "train")
TRAIN_IMG1 = os.path.join(TRAIN_PATH1, "images")
TRAIN_LABEL1 = os.path.join(TRAIN_PATH1, "labels")

VALID_PATH1 = os.path.join(DATA_PATH1, "valid")
VALID_IMG1 = os.path.join(VALID_PATH1, "images")
VALID_LABEL1 = os.path.join(VALID_PATH1, "labels")

# dataset2
DATA_PATH2 = "../../data/cross/dataset2"

TRAIN_PATH2 = os.path.join(DATA_PATH2, "train")
TRAIN_IMG2 = os.path.join(TRAIN_PATH2, "images")
TRAIN_LABEL2 = os.path.join(TRAIN_PATH2, "labels")

VALID_PATH2 = os.path.join(DATA_PATH2, "valid")
VALID_IMG2 = os.path.join(VALID_PATH2, "images")
VALID_LABEL2 = os.path.join(VALID_PATH2, "labels")

## train split
jpg_list = glob(os.path.join(TRAIN_IMG, '*.jpg'))
png_list = glob(os.path.join(TRAIN_IMG, '*.png'))
train_imgs = jpg_list + png_list
print(len(jpg_list))
print(len(png_list))
print(len(train_imgs))

train_labels = glob(os.path.join(TRAIN_LABEL, '*.txt'))
print(len(train_labels))

split_size = len(train_imgs) // 2 if len(train_imgs) < len(train_labels) else len(train_labels) // 2

# dataset1
split_imgs1 = train_imgs[:split_size]
for file_path in tqdm(split_imgs1):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(TRAIN_LABEL + label_name):
    move(TRAIN_IMG+img_name, TRAIN_IMG1+img_name)
    move(TRAIN_LABEL+label_name, TRAIN_LABEL1+label_name)
  
train_jpg_imgs = glob(os.path.join(TRAIN_IMG1, "*.jpg"))
train_png_imgs = glob(os.path.join(TRAIN_IMG1, "*.png"))
train_imgs = train_jpg_imgs + train_png_imgs

print(len(train_jpg_imgs))
print(len(train_png_imgs))
print(len(train_imgs))

with open(os.path.join(DATA_PATH1, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_imgs) + '\n')    

# dataset2
split_imgs2 = train_imgs[split_size:]
for file_path in tqdm(split_imgs1):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(TRAIN_LABEL + label_name):
    move(os.path.join(TRAIN_IMG, img_name), os.path.join(TRAIN_IMG2, img_name))
    move(os.path.join(TRAIN_IMG, label_name), os.path.join(TRAIN_IMG2, label_name))

train_jpg_imgs = glob(os.path.join(TRAIN_IMG2, "*.jpg"))
train_png_imgs = glob(os.path.join(TRAIN_IMG2, "*.png"))
train_imgs = train_jpg_imgs + train_png_imgs
print(len(train_jpg_imgs))
print(len(train_png_imgs))
print(len(train_imgs))

with open(os.path.join(DATA_PATH2, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_imgs) + '\n')

## test split
jpg_list = glob(os.path.join(VALID_IMG, '*.jpg'))
png_list = glob(os.path.join(VALID_IMG, '*.png'))
valid_imgs = jpg_list + png_list
print(len(jpg_list))
print(len(png_list))
print(len(valid_imgs))

valid_labels = glob(os.path.join(VALID_LABEL, '*.txt'))
print(len(valid_labels))

split_size = len(valid_imgs) // 2 if len(valid_imgs) < len(valid_labels) else len(valid_labels) // 2
print(split_size)

# dataset1
split_imgs1 = valid_imgs[:split_size]
for file_path in tqdm(split_imgs1):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(VALID_LABEL + label_name):
    move(os.path.join(VALID_IMG, img_name), os.path.join(VALID_IMG1, img_name))
    move(os.path.join(VALID_LABEL, label_name), os.path.join(VALID_LABEL1, label_name))

valid_jpg_imgs = glob(os.path.join(VALID_IMG1, "*.jpg"))
valid_png_imgs = glob(os.path.join(VALID_IMG1, "*.png"))
valid_imgs = valid_jpg_imgs + valid_png_imgs
print(len(valid_jpg_imgs))
print(len(valid_png_imgs))
print(len(valid_imgs))

with open(os.path.join(DATA_PATH1, 'valid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_imgs) + '\n')

# dataset2
split_imgs2 = valid_imgs[:split_size]
for file_path in tqdm(split_imgs2):
  img_name = file_path.split('/')[-1]
  label_name = img_name[:-4] + '.txt'

  if os.path.exists(VALID_LABEL + label_name):
    move(os.path.join(VALID_IMG, img_name), os.path.join(VALID_IMG2, img_name))
    move(os.path.join(VALID_LABEL, label_name), os.path.join(VALID_LABEL2, label_name))

valid_jpg_imgs = glob(os.path.join(VALID_IMG2, "*.jpg"))
valid_png_imgs = glob(os.path.join(VALID_IMG2, "*.png"))
valid_imgs = valid_jpg_imgs + valid_png_imgs
print(len(valid_jpg_imgs))
print(len(valid_png_imgs))
print(len(valid_imgs))

with open(os.path.join(DATA_PATH2, 'valid.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_imgs) + '\n')