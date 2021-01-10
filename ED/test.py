import os
import sys
from pathlib import Path

import cv2
import numpy as np
import csv

try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    # sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import eddef
except ImportError:
    print('Library Module Can Not Found')


data_eye_cnt = 0

for data_eye in os.listdir('./data/'):
    path = './data/' + data_eye
    filename = data_eye[:-4]    # 확장자를 뺀 이미지 이름

    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(filename)
    data_eye_cnt += 1

### for data_eye in os.listdir('./data/'):

print(f'\n### 총 {data_eye_cnt}개의 좌표 파일 추출')