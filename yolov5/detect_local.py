#%%
import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from datetime import datetime

current_time = datetime.now()

#%%
PED_MODEL_PATH = './runs/train/exp/weights/best.pt'
CROSS_MODEL_PATH = './runs/train/exp3/weights/best.pt'

TEST_VIDEO_PATH = '../test-video/'
TEST_VIDEO_SAVE_PATH = TEST_VIDEO_PATH + 'output/'
CURRENT_TIME = f'{current_time.year}-{current_time.month}-{current_time.day} {current_time.hour}:{current_time.minute}:{current_time.second}'

<<<<<<< HEAD
TEST_VIDEO = 'ewha4.mp4'
=======
TEST_VIDEO = 'acro2.mp4'
>>>>>>> c8901040dbfff62e62e19dbfe8533315ed066d4c
SAVE_VIDEO = CURRENT_TIME + '.mp4'

#%%
img_size = 640
ped_conf_thres = 0.37
cross_conf_thres = 0.57
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False

safe_overlap_thres = 0.5
cross_overlap_thres = 0.3

#%%
print('[Car, pedestrain model info]')
ped_device = torch.device('cpu')
print(ped_device)
ped_ckpt = torch.load(PED_MODEL_PATH, map_location = ped_device)
ped_model = ped_ckpt['ema' if ped_ckpt.get('cma') else 'model'].float().fuse().eval()
ped_class_names = ['보행자', '차량']
ped_stride = int(ped_model.stride.max())
ped_colors = ((50, 50, 50), (255, 0, 0))
print()

#%%
print('[Cross, traffic light model info]')
cross_device = torch.device('cpu')
print(cross_device)
cross_ckpt = torch.load(CROSS_MODEL_PATH, map_location = cross_device)
cross_model = cross_ckpt['ema' if cross_ckpt.get('cma') else 'model'].float().fuse().eval()
cross_class_names = ['횡단보도', '초록불', '빨간불']
cross_stride = int(cross_model.stride.max())
cross_colors = ((255, 0, 255), (0, 0, 255), (0, 255, 0))
print()

#%%
cap = cv2.VideoCapture(TEST_VIDEO_PATH + TEST_VIDEO)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(TEST_VIDEO_SAVE_PATH + SAVE_VIDEO, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# %%
# input image processing
def img_process(img, stride, device):
	img_input = letterbox(img, img_size, stride = stride)[0]
	img_input = img_input.transpose((2, 0, 1))[::-1]
	img_input = np.ascontiguousarray(img_input)
	img_input = torch.from_numpy(img_input).to(device)
	img_input = img_input.float()
	img_input /= 255.
	img_input = img_input.unsqueeze(0)

	return img_input

#%%
# predict classes
def pred_classes(pred, class_names:list, ignore_class_names:list, annotator, colors)->dict:
	assert class_names == ped_class_names or class_names == cross_class_names, 'given class names are not allowed'

	preds = {class_name:[] for class_name in class_names if class_name not in ignore_class_names}

	for p in pred:
		class_name = class_names[int(p[5])]
		# x1, y1, x2, y2
		position = p[:4]

		if class_name not in ignore_class_names:
			preds[class_name].append(position)
			if annotator is not None:
				annotator.box_label(position, '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

	return preds

#%%
def detect(img, stride, device, model, class_names, ignore_class_names, colors, annotator=None):
	global cross_x1, cross_y1, cross_x2, cross_y2
	
	img_input = img_process(img, stride, device)
	
	pred = model(img_input, augment = False, visualize = False)[0]
	
	if '보행자' in class_names:
		pred = non_max_suppression(pred, ped_conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det)[0]
	elif '횡단보도' in class_names:
		pred = non_max_suppression(pred, cross_conf_thres, iou_thres, classes, agnostic_nms, max_det = max_det)[0]
	else:
		raise Exception('Model doesn\'t exist')
	pred = pred.cpu().numpy()
	
	pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()
	
	preds = pred_classes(pred, class_names, ignore_class_names, annotator, colors)
	
	return preds

#%%
print('[Check the pedestrain safety range]')
_, img = cap.read()

cnt = 0
isClick = False
isFinish = False

safe_x1, safe_y1 = 0, 0
safe_x2, safe_y2 = 0, 0

def click_event(event, x, y, flags, param):
	global cnt, isClick, isFinish
	global safe_x1, safe_y1, safe_x2
	
	if isFinish:
		return
	
	if isClick is False:
		if cnt == 1:
			print('Click the right down position')
		elif cnt == 2:
			print('Click the upper position')
		elif cnt == 3:
			print('Finish... Please press any key')
			isFinish = True
			return
		isClick = True
	
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, ' ', y)
		if cnt == 0:
			safe_x1 = x
		elif cnt == 1:
			safe_x2 = x
		elif cnt == 2:
			safe_y1 = y
		cnt += 1
		isClick = False

print('Click the left down position')

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print()

#%%
print('[Run the model]')
# 횡단보도 찾고 고정
ret, img = cap.read()
preds_dict = detect(img, cross_stride, cross_device, cross_model, cross_class_names, ['빨간불', '초록불'], cross_colors)
cross_x1, cross_y1, cross_x2, cross_y2 = preds_dict['횡단보도'][0]
safe_y2 = cross_y1

#%%
cur_frame = 1
pbar = tqdm(total=frames)

green = cv2.imread('./green.png')
yellow = cv2.imread('./yellow.png')
red = cv2.imread('./red.png')
rows, cols, _ = green.shape
H, W, _ = img.shape
start_row = int(H / 45)
start_col = int(W / 2) - int(cols / 2)

# 사각형 겹침 확인
def is_overlap(rect1, rect2):
	return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[1] > rect2[3] or rect1[3] < rect2[1])

# 사각형 겹치는 영역 크기
def overlap_area(rect1, rect2):
	assert len(rect1) == 4 and len(rect2) == 4
	x_poses = [rect1[0], rect1[2], rect2[0], rect2[2]]
	y_poses = [rect1[1], rect1[3], rect2[1], rect2[3]]
	x_poses.sort()
	y_poses.sort()

	overlap_w, overlap_h = x_poses[2] - x_poses[1], y_poses[2] - y_poses[1]

	# width, height
	return overlap_w * overlap_h

while cap.isOpened():
	ret, img = cap.read()
	if not ret:
		break

	pbar.update(cur_frame)

	annotator = Annotator(img.copy(), line_width = 3, example = '한글', font = 'data/malgun.ttf')

	preds1 = detect(img, ped_stride, ped_device, ped_model, ped_class_names, [], ped_colors, annotator)
	preds2 = detect(img, cross_stride, cross_device, cross_model, cross_class_names, ['빨간불', '초록불'], cross_colors, annotator)
	img = annotator.result()
	result_img = img.copy()
	cv2.rectangle(result_img, (int(safe_x1), int(safe_y1)), (int(safe_x2), int(safe_y2)), (255, 255, 255), 3)

	peds = preds1['보행자']
	cars = preds1['차량']
	crosses = preds2['횡단보도']

	# 자동차가 안전범위 가리는지 check
	# 사람이 횡단보도를 건너고 있어 빨간불이라면 no check
	is_safe_hide = False
	if len(cars):
		for car in cars:
			car_x1, car_y1, car_x2, car_y2 = car
			_cross = [safe_x1, safe_y1, safe_x2, safe_y2]
			_safe_area = (safe_x2 - safe_x1) * (safe_y2 - safe_y1)
			if is_overlap(car, _cross):
				_overlap_area = overlap_area(car, _cross)
				_overlap_area_ratio = _overlap_area / _safe_area
				if _overlap_area_ratio >= safe_overlap_thres:
					is_safe_hide = True
					break

	# 횡단보도가 가려지는지 check
	is_cross_hide = False
	init_cross_H = cross_y2 - cross_y1
	if len(crosses):
		cur_cross_x1, cur_cross_y1, cur_cross_x2, cur_cross_y2 = crosses[0]
		cur_cross_H = cur_cross_y2 - cur_cross_y1
		overlap_ratio = cur_cross_H / init_cross_H
		if overlap_ratio <= cross_overlap_thres:
			is_cross_hide = True
	else:
		is_cross_hide = True

	# safety 체크 알고리즘
	in_safety, in_cross = False, False		
	is_hide = is_safe_hide or is_cross_hide	
	if len(peds):
		for ped in peds:
			ped_x1, ped_y1, ped_x2, ped_y2 = ped

			_in_safety = is_overlap([safe_x1, safe_y1, safe_x2, safe_y2], [ped_x1, ped_y2, ped_x2, ped_y2])
			_in_cross = is_overlap([cross_x1, cross_y1, cross_x2, cross_y2], [ped_x1, ped_y2, ped_x2, ped_y2])
			in_safety, in_cross = in_safety or _in_safety, in_cross or _in_cross

			# red : stop!; yellow : stop and go; green : drive slowly
			if in_cross:
				result_img[start_row:start_row+rows, start_col:start_col+cols] = red
			elif is_hide or in_safety:
				result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
			else:
				result_img[start_row:start_row+rows, start_col:start_col+cols] = green
	elif is_hide:
		result_img[start_row:start_row+rows, start_col:start_col+cols] = yellow
	else:
		# no ped
		result_img[start_row:start_row+rows, start_col:start_col+cols] = green

	out.write(result_img)
	if cv2.waitKey(1) == ord('q'):
		break
print("Output saved as " + TEST_VIDEO_SAVE_PATH + SAVE_VIDEO)

#%%
cap.release()
out.release()
