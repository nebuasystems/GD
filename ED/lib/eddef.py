import numpy as np
import matplotlib.pyplot as plt
import cv2


def imshow(title, image):

    plt.title(title)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')      #, extent=(0, image.shape[1]*10, 0, image.shape[0]*10))

    plt.show()

# 전체 이미지 사이즈의 상하좌우 외부에서 안쪽으로 옮기며 값이 존재하는 행과 열을 찾아서 파라메터 산출 < 상하우좌는 모두 실제 그림 기준
def paraglass(img):

    ctr_edged = 255 - img

    data_x = ctr_edged.shape[0]
    data_y = ctr_edged.shape[1]

    # 1. 상한 경계 행 찾기
    for i in range(data_x):                # 실제 그림에서는 상하 y방향
        nowValue = 0
        for j in range(data_y):
            nowValue += ctr_edged[i][j]
        if nowValue > 255*2:               # 각 상황별로 어떤 값이 적절한가??
            break
    i = i - 5
    # cv2.line(lining_img, (0, i), (data_y, i), (0, 255, 0), 1) # y, x 방향 주의
    top_line_x = i

    # 2. 하한 경계 행 찾기
    for i in range(data_x-1, -1, -1):       # 실제 그림에서는 상하 y방향
        nowValue = 0
        for j in range(data_y):
            nowValue += ctr_edged[i][j]
        if nowValue > 255*2:                 # 각 상황별로 어떤 값이 적절한가??
            break
    i = i + 5
    # cv2.line(lining_img, (0, i), (data_y, i), (0, 255, 0), 1) # y, x 방향 주의
    bottom_line_x = i

    # 3. 우한 경계 열 찾기
    for j in range(data_y-1, -1, -1):         # 실제 그림에서는 상하 x방향
        nowValue = 0
        for i in range(data_x):
            nowValue += ctr_edged[i][j]
        if nowValue > 255*2:                  # 각 상황별로 어떤 값이 적절한가??
            break
    i = i - 5
    # cv2.line(lining_img, (j, 0), (j, data_x), (0, 255, 0), 1) # y, x 방향 주의
    right_line_y = j

    # 4. 좌한 경계 열 찾기
    for j in range(data_y):                     # 실제 그림에서는 상하 x방향
        nowValue = 0
        for i in range(data_x):
            nowValue += ctr_edged[i][j]
        if nowValue > 255*2:                    # 각 상황별로 어떤 값이 적절한가??
            break
    i = i + 5
    # cv2.line(lining_img, (j, 0), (j, data_x), (0, 255, 0), 1) # y, x 방향 주의
    left_line_y = j

    # 5. 좌측 안구 중앙점 산출
    # ctrL_x = int((top_line_x + bottom_line_x)/2) - 30    # 좌측 안구 중심 y   > 안경 디자인의 디테일은 상단에 많다. 중앙점을 위쪽으로 올리자  + alpha > 얼마만큼??? (30 > 50 > 30)
    print(int((bottom_line_x - top_line_x)/10))
    ctrL_x = int((top_line_x + bottom_line_x) / 2) - int((bottom_line_x - top_line_x)/10)    # 좌측 안구 중심 y > 안경 디자인의 디테일은 상단에 많다. 중앙점을 위쪽으로 올리자 : 이미지 영역의 ?%
    ctrL_y = int((left_line_y*3 + right_line_y)/4)- 0    # 좌측 안구 중심 x   > 안경 중심 기준으로 계산하여서 약간 우측으로 쏠림
    print(f'실제그림에서 안구 중앙 y좌표={ctrL_x}, x좌표={ctrL_y}')

    # lensR = int((right_line_y - left_line_y)/4) + 32   # 렌즈 반경 :  좌우측 경계 기준 + alpha
    # lensR =  int(np.sqrt((top_line_x - ctrL_x)**2 +  (left_line_y - ctrL_y)**2))   # 렌즈 반경 : 이미지상 좌상귀와 좌측 안구 중심 거리
    lensR =  int(np.sqrt((bottom_line_x - ctrL_x)**2 +  (left_line_y - ctrL_y)**2))   # 렌즈 반경 : 이미지상 좌하귀와 좌측 안구 중심 거리 < 중앙점을 위로 올려서 수정
    print(f'렌즈 반경 = {lensR}')
    # center = (ctrL_y, ctrL_x)
    # cv2.circle(img, center, lensR, (0, 255, 0), 1, cv2.LINE_8)
    # cv2.circle(img, center, 3, (0, 255, 0), 1, cv2.LINE_8)

    centerAll_y = int((left_line_y + right_line_y)/2)
    print(f'전체 중심 = {centerAll_y}')
    # cv2.line(lining_img, (centerAll_y, 0), (centerAll_y, data_x), (0, 255, 0), 1) # y, x 방향 주의

    return [top_line_x, bottom_line_x, right_line_y, left_line_y, ctrL_x, ctrL_y, lensR, centerAll_y]


# 1. bridge 문제 : 반으로 가르는 가상의 중간선을 실제 데이터에 적용 > 중간선을 그을 수 없는 경우 대응(bridge 미검출 이거나 bridge 연결 간격이 없는 경우)
def halfline(img, centerAll_y, bottom_line_x, ctrL_x):

    data_x = img.shape[0]
    center_list = []
    for i in range(data_x):  # 이미지 사이즈 위에서 부터 조사
        if img[i][centerAll_y - 1] == 0:
            center_list.append(i)
            break
    print(f'중간선 {int((bottom_line_x + ctrL_x) / 2)}~{data_x - 1}')
    for i in range(int((bottom_line_x + ctrL_x) / 2), -1, -1):  # 아래에서 부터 조사 < 이미지 사이즈 기준이 아니라 안경 영역인 하한 경계와 중앙 라인 사이 기준
        if img[i][centerAll_y - 1] == 0:
            center_list.append(i)
            break

    # print(f'center_list={center_list}')
    if len(center_list) >= 2:               # bridge가 2개이거나 하는 문제!!
        cv2.line(img, (centerAll_y - 1, center_list[0]), (centerAll_y - 1, center_list[1]), (0, 255, 0), 1)  # y, x 방향 주의

    return img


########################################################################################################
# 대각선 방향으로 위치 추출시 데이터 사이를 통과해서 놓치 거나 엉뚱한 위치를 인식하는 문제 대응
# 지금 좌표(edged[x+ctrL_x][y+ctrL_y])에서 가까운(1칸) 좌표에 값이 있다면 그 지점을 선택
#   < 데이터 선정의 오류는 없으며 중앙점 기준으로 추출된 데이터간의 미세한 각도 간격이 균일하지 않는 정도의 차이
# 1. 방사형의 진행 방향 기준(ccw, theta) 4분면으로 나눈다
# 2. 현 지점에서 인접한 두 귀의 자리에 값이 있는지 확인 : 진행 방향 우선 : 이미지에서 보여지는 좌표 기준 > (y, x) , 0~90도는 4/4분면
# 중복 고려, 지저분한 데이터가 걸릴지도 모른다. 중앙 연결 부위 문제
# 90도에서 이상하다. 왜 내곽에 영향을 주는가.!!!!!!!!
def findhiddenpoint(edged, theta, x, y):

    if (x - 1) < 0 or (x + 1) > (edged.shape[0]-1) or (y - 1) < 0 or (y + 1) > (edged.shape[1]-1):
        return [0, 0]

    edgex_theta = 0
    edgey_theta = 0

    if 0 <= theta < 90:
        if edged[x][y + 1] == 0:
            edgex_theta = x
            edgey_theta = y + 1
        elif edged[x - 1][y] == 0:
            edgex_theta = x - 1
            edgey_theta = y
    if 90 <= theta < 180:
        if edged[x - 1][y] == 0:  # + 1 ???
            edgex_theta = x - 1
            edgey_theta = y
        elif edged[x][y + 1] == 0:
            edgex_theta = x
            edgey_theta = y + 1
    if 180 <= theta < 270:
        if edged[x][y - 1] == 0:
            edgex_theta = x
            edgey_theta = y  - 1
        elif edged[x - 1][y] == 0:
            edgex_theta = x - 1
            edgey_theta = y
    if 270 <= theta < 360:
        if edged[x + 1][y] == 0:
            edgex_theta = x + 1
            edgey_theta = y
        elif edged[x][y - 1] == 0:
            edgex_theta = x
            edgey_theta = y - 1

    return [edgex_theta, edgey_theta]


def findhiddenpoint_in(edged, theta, x, y):

    if (x - 1) < 0 or (x + 1) > (edged.shape[0]-1) or (y - 1) < 0 or (y + 1) > (edged.shape[1]-1):
        return [0, 0]

    edgex_in = 0
    edgey_in = 0

    if 0 <= theta < 90:
        if edged[x + 1][y] == 0:
            edgex_in = x + 1
            edgey_in = y
        elif edged[x][y - 1] == 0:
            edgex_in = x
            edgey_in = y - 1
    if 90 <= theta < 180:
        if edged[x][y - 1] == 0:
            edgex_in = x
            edgey_in = y - 1
        elif edged[x + 1][y] == 0:  # - 1 ?
            edgex_in = x + 1
            edgey_in = y
    if 180 <= theta < 270:
        if edged[x - 1][y] == 0:
            edgex_in = x - 1
            edgey_in = y
        elif edged[x][y + 1] == 0:
            edgex_in = x
            edgey_in = y + 1
    if 270 <= theta < 360:
        if edged[x][y + 1] == 0:
            edgex_in = x
            edgey_in = y + 1
        elif edged[x + 1][y] == 0:
            edgex_in = x + 1
            edgey_in = y

    return [edgex_in, edgey_in]