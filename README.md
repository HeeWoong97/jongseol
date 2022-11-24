# TRIP(Turn Right If Possible)
## 팀원
* 20161166 양희웅
* 20181320 송지우
***
## 주제
### 차량에게 안전한 비보호 회전 안내
* 사람이 횡단보도에 접근하면 일단 멈춘 후 상황을 파악해야 한다
* 사람이 횡단보도를 건너고 있으면 무조건 멈춰야 한다
* 이는 비보호 좌회전 시에도 동일한데, 운전자가 쉽게 확인할 수 없는 시야각이 있어서 보행자를 발견하지 못할 확률이 있다
* https://www.youtube.com/watch?v=kNRo2DryF58&t=1s
### Goal
* 횡단보도, 신호등, 보행자의 정보를 통해 차량에게 안전한 좌, 우회전 시기를 안내해주는 시스템
***
## 사용한 dataset
* 보행자, 차 (AIHub 차량 및 사람 인지 영상 데이터셋) - 26000장
    * https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=195
    * 13000장씩 나누어서 저장
        * data/car_pedestrian/dataset1
        * data/car_pedestrian/dataset2
* 횡단보도, 신호등 (SelectStar 교차로 및 화폐 정보 데이터셋 데이터셋) - 35000장
    * https://open.selectstar.ai/data-set/wesee
    * 17500장씩 나누어서 저장
        * data/cross/dataset1
        * data/cross/dataset2
* 데이터의 용량이 너무 커서 github에 업로드 하지 못했습니다
* Google Drive 링크
    * https://drive.google.com/drive/folders/1-kA-24ls4H_zT9m31wy0f8EVAqB3xAbM?usp=share_link
***
## 학습 weight
* 차량, 보행자 모델
    * code/final/weights/car_pedestrian.pt
* 횡단보도, 신호등 모델
    * code/final/weights/cross.pt
***
## 코드의 구조
* code/
    * original/
        * 참고한 코드가 위치
        * cross.py
            * 빵형의 개발도상국 '세계최초? 교차로 우회전 위반 적발하는 인공지능'으로부터 참조
            * https://github.com/kairess/crosswalk-traffic-light-detection-yolov5
        * yolov5
            * TRIP이 사용한 오픈소스 물체인식 라이브러리
    * final/
        * 구현한 코드가 위치
        * **TRIP.py**
            * 메인 알고리즘 코드
            * **prediction 관련 주요코드**
                * 이미지 프로세싱 관련
                ``` python
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
                ```
                * class 검출 관련
                ``` python
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
                ```
                * detect 메인 알고리즘
                ``` python
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
                ```
            * **안전범위 설정 관련 코드**
                ``` python
                safe_x1, safe_y1 = 0, 0
                safe_x2, safe_y2 = 0, 0

                def click_event(event, x, y, flags, param):
                    global cnt, isClick, isFinish
                    global safe_x1, safe_y1, safe_x2, safe_y2
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
                            safe_y1, safe_y2 = y, y
                        cnt += 1
                        isClick = False
                ```
            * **안전범위를 통한 메인 알고리즘**
            * bounding box끼리 겹치는지 확인
                ``` python
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
                ```
            * 자동차가 안전범위를 가리는지 확인
                ``` python
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
                ```
            * 자동차가 횡단보도를 가리는지 확인
                ``` python
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
                ```
            * 운전자에게 안내하는 메인 부분
                ``` python
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
                ```
        * detect_cross.py
            * 영상처리로서 횡단보도 인식을 시도했던 코드
            * HSV, dilate, canny, houghlinesp의 방법을 사용
        * preprocess/
            * 데이터를 전처리 하는 코드
            * car_ped_preprocess.py
                * 차량, 보행자 데이터셋을 train, valid로 나눔
                * YOLO에서 인식하는 labeling 형식으로 변환
            * cross_preprocess.py
                * 횡단보도, 신호등 데이터셋을 train, valid로 나눔
                * YOLO에서 인식하는 labeling 형식으로 변환
            * car_ped_split.py
                * 차량, 보행자 데이터셋을 dataset1, dataset2로 나눔
                * 모든 데이터를 한번에 메모리에 캐싱할 수 없어서 2개의 부분으로 나누었습니다.
            * cross_split.py
                * 횡단보도, 신호등 데이터셋을 dataset1, dataset2로 나눔
        * train/
            * 모델을 학습하는 코드
            * car_ped_train.py
                * 차량, 보행자 모델을 학습
            * cross_train.py
                * 횡단보도, 신호등 모델을 학습
        * metadata/
            * yolo를 통한 학습 시 데이터들의 정보를 알려주는 파일들
            * 각 데이터셋의 경로가 담긴 yaml형식의 파일들
        * weights/
            * 학습 결과
            * car_pedestrian.pt
                * 차량, 보행자 모델 가중치값
            * cross.pt
                * 횡단보도, 신호등 모델 가중치값
        * video/
            * 처리하고 처리된 영상이 저장되는 곳
            * target/
                * 처리할 영상을 저장
            * output/
                * 처리가 완료된 영상이 저장
***
## 실행 방법
* 모델 학습 방법
    1. AI-HUB, 셀렉트스타에서 데이터셋을 직접 다운받을 경우
        1. code 폴더 내에 data 폴더를 만든다.
            ```
            $ mkdir code/data
            ```
        2. data 폴더 내에 car_pedestrian, cross 폴더를 만든다.
            ```
            $ mkdir code/data/car_pedestrian
            $ mkdir code/data/cross
            ```
        3. 각 폴더에 데이터를 다운받는다.
        4. 각 데이터셋에 대해 YOLO 형식으로 labeling을 변환한다.
            ```
            $ cd code/final/preprocess
            $ python car_ped_preprocess.py
            $ python cross_preproces.py
            ```
        5. 변환된 데이터셋을 dataset1, dataset2로 나눈다.
            ```
            $ python car_ped_split.py
            $ python cross_split.py
            ```
    2. 구글 드라이브에서 데이터셋을 직접 다운받을 경우
        1. 구글 드라이브에 접속한다.
            * https://drive.google.com/drive/folders/1-kA-24ls4H_zT9m31wy0f8EVAqB3xAbM?usp=share_link
        2. data 폴더를 code 폴더에 다운받는다. 
            * 예상되는 폴더 구조
            * code/data/car_pedestrian/
            * code/data/cross/
    3. 이후 공통사항
        * 차량, 보행자 모델 학습
            ```
            $ cd code/final/train
            $ python car_ped_train.py
            ```
        * 횡단보도, 신호등 모델 학습
            ```
            $ cd code/final/train
            $ python cross_train.py
            ```
* 모델 실행 방법
    1. 분석할 영상을 code/final/video/target에 넣는다

    2. 코드 실행
        ```
        $ cd code/final
        $ python TRIP.py target.mp4
        ```

    3. 안전범위 설정
        ```
        [Check the pedestrian safety range]
        Click the left down position
        351   407
        Click the right down position
        1049   401
        Click the upper position
        541   345
        Finish... Please press any key...
        ```
        1. 안전범위의 왼쪽 아래 클릭
        2. 안전범위의 오른쪽 아래 클릭
        3. 안전범위의 위쪽 부분 클릭
        4. 아무 키를 입력하여 안전범위 설정 종료

    4. 모델이 실행된다
        ```
        [Run the model]
        31%|█████████████████████████████████████████                                                                                            | 738/2390 [01:51<04:06,  6.70it/s]
        ```

    5. 실행 결과는 code/final/video/output에 저장되어 있다
***
## 실행 결과
* 테스트 장소
    * 이화약국 앞
    * 아크로리버하임 앞
    * 중대병원 앞1
    * 중대병원 앞2
* 왼쪽 위부터 이화약국, 아크로리버하임, 중대병원 앞1, 중대병원 앞2 순서로 배치했음
1. 보행자가 횡단보도를 건너는 경우: 빨간불
![red](https://user-images.githubusercontent.com/53477646/203051255-27a622c3-dd85-4f5a-9b4e-d6112211b41f.jpg)

2. 보행자가 안전범위에 있는 경우: 노란불
![yellow](https://user-images.githubusercontent.com/53477646/203051263-f66303d2-eea9-40e6-baf3-125c0fbf22db.jpg)

3. 보행자가 없거나 안전범위 밖에 있는 경우: 초록불
![green](https://user-images.githubusercontent.com/53477646/203051228-d59e6d28-0489-4f3a-9751-928f5727ab9a.jpg)

4. 안전범위가 가려지는 경우: 노란불
![yellow1](https://user-images.githubusercontent.com/53477646/203051278-9f4a360e-4e5a-47d8-826e-849f9cb5c64f.jpg)

5. 횡단보도가 가려지는 경우: 노란불
![yellow2](https://user-images.githubusercontent.com/53477646/203051289-a0d21c9d-c37a-419a-a079-f62e6461d93a.jpg)