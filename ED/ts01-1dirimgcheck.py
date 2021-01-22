# 안경 테두리 좌표 추출 : 20210108
# 01 : ts10arrange에서 이전
# 01-1seqdataviewing : 디 렉토리에 저장된 데이터를 하나씩 처리하며 확인
#       : 테두리 추적 방향을 ccw, cw 가변 : 내부 다리 문제 해결을 위하여...> ts02에 반영해야 함
#       : 내곽 추출시 좌우 상단 모서리에서는 범위를 넓힌다. < cw는 중간 부위가 이상함
#       : cw로 하면 불리한 경우 발생(balmain 2003). 추출 진입각에 따라서 다르다(momentum 고려?)

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np
import csv

try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    # sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import eddef
except ImportError:
    print('Library Module Can Not Found')


for data_eye in os.listdir('./data/'):

    # for test
    # if data_eye != "MK women's panama 2004 (1).jpg":
    #     continue


    path = './data/' + data_eye
    img = cv2.imread(path)

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_g = cv2.GaussianBlur(img_g, (3, 3), 0)

    img_gb = img_g

    _, img_g = cv2.threshold(img_g, 255, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)   # 일부 영역에만 적용이 가능하나? : 브릿지, 테두리 등이 잘 날라감

    print(f'\n\n# {data_eye[:-4]} ------------------------------------------')

    # edged = 255 - cv2.Canny(img_g, 20, 400)
    edged = 255 - cv2.Canny(img_g, 10, 250) # 200 ?
    print(f'image size(y,x) = {edged.shape}')

    img_tc = edged

    data_x = edged.shape[0]
    data_y = edged.shape[1]

    point_list_outer = []
    point_list_inner = []
    point_list_inner_auto = []  # 외곽에서 자동으로 산출되는 내곽

    lining_img = 255 - np.zeros_like(edged)    # 저장 좌표 데이터를 표시할 이미지(행렬)

    cnt = 0

    # 전체 이미지 사이즈의 상하좌우 외부에서 안쪽으로 옮기며 값이 존재하는 행과 열을 찾아서 파라메터 산출 < 상하우좌는 모두 실제 그림 기준
    top_line_x, bottom_line_x, right_line_y, left_line_y, ctrL_x, ctrL_y, lensR, centerAll_y = eddef.paraglass(edged)


    ####################################################################################################################
    # 이상치 전처리
    #
    # 1. bridge 문제 : 반으로 가르는 가상의 중간선을 실제 데이터에 적용 > 브리지가 한 줄로 나오는 등 다른 경우의 고려 사항??
    edged = eddef.halfline(edged, centerAll_y, bottom_line_x, ctrL_x)
    # eddef.imshow('half line', edged)
    # 2. bridge가 사라지는 경우에는 ? > 가상으로 선을 그어준다?


    ########################################################################################################################
    # 좌측 안구 중앙점에서 방사형 라인으로 교차하는 위치 좌표의 데이터 유무로 필요한 좌표 데이터를 찾는다

    # 이전 선정 지점 : 이전 방사선
    edgex_pre = 0
    edgey_pre = 0
    r_pre = 0

    edgex_in_pre = 0
    edgey_in_pre = 0

    ########################################################################################################################
    # cw로 회전 : 시작을 코끝 방향에서 : 45 ~ 0, 360 ~ 46
    theta1 = np.arange(45, 0, -1)
    theta2 = np.arange(360, 45, -1)
    theta_range = np.concatenate((theta1, theta2), axis=0)
    ########################################################################################################################
    # ccw로 회전 : 시작을 코끝 방향에서 : 45 ~ 359, 0 ~ 44
    # theta1 = np.arange(45, 360)
    # theta2 = np.arange(0, 45)
    # theta_range = np.concatenate((theta1, theta2), axis=0)

    # for theta in range(0, 360, 1):  # ccw. 몇개를 선택하는 것이 적절한가. < 돌출 문제로 근접점을 선정하려면 촘촘해야 한다
    for theta in theta_range:  # ccw. 몇개를 선택하는 것이 적절한가. < 돌출 문제로 근접점을 선정하려면 촘촘해야 한다

        # 검출 지점
        edgex_theta = 0
        edgey_theta = 0

        # 앞선 검출점 : 검출 지점 이전에 가장 적절하여 저장한 지점
        edgex_dis = 0
        edgey_dis = 0
        r_dis = 0

        # 이전 검출된 지점과 검출 지점과의 거리
        now_distance = 0
        now_distance_in = 0

        # 방사 라인을 거치며 측정된 최소 거리 지점 : now_distance중 최소
        saved_distance = 0
        saved_distance_in = 0

        # 선택 지점
        edgeSELx_theta = 0
        edgeSELy_theta = 0

        for r in range(lensR):          # 조사 반경(lensR) : 데이터 영역 좌하귀와 좌측 안구 중심 간격

            edgex_theta = 0
            edgey_theta = 0

            radian = theta * np.pi/180
            x = int(np.cos(radian) * r)
            y = int(np.sin(radian) * r)

            #########################################
            # 좌표가 좌상단과 중앙선+하단을 넘지 않도록 한다

            if (top_line_x-5 < (x+ctrL_x) < bottom_line_x+5) and (left_line_y-5 < (y+ctrL_y) < centerAll_y):

                xedge = x+ctrL_x
                yedge = y+ctrL_y

                if xedge >= data_x:
                    xedge = data_x - 1
                elif xedge < 0:
                    xedge = 0

                if yedge >= data_y:
                    yedge = data_y - 1
                elif yedge < 0:
                    yedge = 0

                if edged[xedge][yedge] == 0:          # 가장 바깥에서 검출된 좌표가 남길 바라

                    edgex_theta = xedge
                    edgey_theta = yedge

                else:
                    # 데이터 틈새로 지나가는 경우를 보완
                    edgex_theta, edgey_theta = eddef.findhiddenpoint(edged, theta, xedge, yedge)


                ########################################################################################################
                # 검출 후, 외곽 돌출 문제
                #   > 이전 선정 지점과의 거리를 비교하여 가장 가까운 지점 선정
                if (edgex_theta != 0 and edgey_theta != 0):
                    if (edgex_pre == 0 and edgey_pre == 0):             # 이전 선정점이 없다 or 최소 거리 지점이 없다 < 초기 시작 지점에 그림자 등이 있으면 문제!!
                        edgex_dis = edgex_theta
                        edgey_dis = edgey_theta
                        r_dis = r
                    else:                                               # 이전 선택점과의 거리 비교 : 가까우면 선택됨
                        saved_distance = np.sqrt((edgex_pre - edgex_dis)**2 + (edgey_pre - edgey_dis)**2)
                        now_distance = np.sqrt((edgex_pre - edgex_theta)**2 + (edgey_pre - edgey_theta)**2)

                        if ( (now_distance - 1) < saved_distance):        # (now_distance-alpha) : alpha 내부로 타지 않고 가길 바라
                                edgex_dis = edgex_theta
                                edgey_dis = edgey_theta
                                r_dis = r


                    # print(f'---theta={theta}, now_distance={now_distance}, saved_distance={saved_distance}, edgex={edgex_dis}, edgey={edgey_dis}, edgex_pre={edgex_pre}, edgey_pre={edgey_pre}')

        ### for r in range(lensR):

        ####################################################################################################################
        # 외곽 추출
        if edgex_dis != 0 and edgey_dis != 0:   # 최외곽 데이터가 있다면 기록
            edgeSELx_theta = edgex_dis
            edgeSELy_theta = edgey_dis

            point_list_outer.append((edgeSELx_theta, edgeSELy_theta))
            # cv2.circle(lining_img, (edgeSELy_theta, edgeSELx_theta), 1, (0, 0, 255), -1)

            edgex_pre = edgeSELx_theta  # 중복 느낌이 들어도 이해하기 쉽도록 선택된 위치를 다음에 사용할 경우에 이전(pre)으로 전달
            edgey_pre = edgeSELy_theta
            r_pre = r_dis

            cnt += 1

            ################################################################################################################
            # 임시 내곽 추출 : 검출된 외곽 기준 산출
            r_i = r_dis - lensR / 30  # 수치 기준 조사 필요
            x_i = int(np.cos(radian) * r_i) + ctrL_x
            y_i = int(np.sin(radian) * r_i) + ctrL_y
            point_list_inner_auto.append((x_i, y_i))


            ####################################################################################################################
            # 내곽 추출 : 추출된 외곽을 기준으로 진행
            edgex_in = 0
            edgey_in = 0
            edgex_in_dis = 0
            edgey_in_dis = 0

            # for r in range(lensR):
            # for r in range(2, 70):            # 조금(2) 띄우고 시작, 외곽에서 내곽이 얼마나 떨어져 있나? : 중앙 브릿지 분리선에서 가장 멀 것이다. < cucci 15, ray
            # for r in range(2, 200):              # 이미지 크기에 따라 다르다. 특히 브릿지에서 200이 넘을 수도.
            inner_range = 0
            if (45 < theta < 135) or (225 < theta < 270):
                inner_range = 200
            else:
                inner_range = 70
            for r in range(2, inner_range):

                radian = theta * np.pi / 180
                x = int(np.cos(radian) * r)
                y = int(np.sin(radian) * r)

                # out of bound 방지
                if data_x <= (edgeSELx_theta - x):
                    x = edgeSELx_theta - data_x + 1
                if data_y <= (edgeSELy_theta - y):
                    y = edgeSELy_theta - data_y + 1

                if edged[edgeSELx_theta - x][edgeSELy_theta - y] == 0:
                    edgex_in = edgeSELx_theta - x
                    edgey_in = edgeSELy_theta - y

                else:
                    # 데이터 틈새로 지나가는 경우를 보완
                    edgex_in, edgey_in = eddef.findhiddenpoint_in(edged, theta, edgeSELx_theta - x, edgeSELy_theta - y)

                # 이전 선정 지점과의 거리를 비교하여 가장 가까운 지점 선정
                if (edgex_in != 0 and edgey_in != 0):
                    if (edgex_in_pre == 0 and edgey_in_pre == 0):       # 이전 선정점이 없다 or 최소 거리 지점이 없다.
                        edgex_in_dis = edgex_in
                        edgey_in_dis = edgey_in
                    else:                                               # 이전 선택점과의 거리 비교 : 가까우면 선택됨
                        saved_distance_in = np.sqrt((edgex_in_pre - edgex_in_dis)**2 + (edgey_in_pre - edgey_in_dis)**2)
                        now_distance_in = np.sqrt((edgex_in_pre - edgex_in)**2 + (edgey_in_pre - edgey_in)**2)

                        if ( (now_distance_in - 1) <= saved_distance_in):        # (now_distance_in-alpha) : alpha 다른 라인 타지 않고 가길 바라
                                edgex_in_dis = edgex_in
                                edgey_in_dis = edgey_in


                    # print(f'---theta={theta}, now_distance_in={now_distance_in:.3f}, saved_distance_in={saved_distance_in:.3f}, edgex={edgex_in_dis}, edgey={edgey_in_dis}, edgex_pre={edgex_in_pre}, edgey_pre={edgey_in_pre}')

            ### for r in range(50): #inner

            if edgex_in_dis != 0 and edgey_in_dis != 0:  # 내곽 데이터가 있다면 기록
                point_list_inner.append((edgex_in_dis, edgey_in_dis))

                edgex_in_pre = edgex_in_dis
                edgey_in_pre = edgey_in_dis
            # else:
            #     edgex_in_pre = 0
            #     edgey_in_pre = 0

    ### for theta in range(0, 360, 3):


    ####################################################################################################################
    # 이상치 후처리
    # 외곽의 경우 어떤 돌출로 인해 중심점 기준으로 더 멀리 갔다는 소리
    # 다리 연결부 돌출 : 다리 연결부 위치가 안구 테두리 최외곽 대역이 아니라면, 최외곽 끝과 비교하여 더 먼 위치의 좌표 추출시 의심 가능 또는 다리의 위치 정보로 그 부분 집중 공략 <동서양 다리 위치가 다르다

                #   1. 기울기, 거리가 급변하는 구간 조사
                #   2. 문제 구간 양 점을 각각 제외하고 남은 한점에서 양쪽의 기울기 차이를 구한다
                #   3. 기울기 차이가 크지 않은 점만 남긴다.
                #   4. 점 몇 개로 라인 추정하는 기술은 여기서 필요한가? < 이걸로 추정해서 데이터를 갖추는 의미와 학습시의 처리와 이해상충하는 여부?
                #   5. 같은 값이 있다면 제거????
    # del(point_list_outer[41])   # 문제 데이터 삭제


    # for i in range(len(point_list)-1, -1, -1):  # 역순으로 진행


    # 검출된 내곽이 부적절하다면 자동 검출 내곽을 적용 : 판정 기준?
    if len(point_list_inner) < 300:
        point_list_inner = point_list_inner_auto

    # 정리된 리스트의 위치 정보를 이미지에 표시
    for i in range(len(point_list_outer)):
        cv2.circle(lining_img, (point_list_outer[i][1], point_list_outer[i][0]), 1, (0, 0, 255), -1)

    for i in range(len(point_list_inner)):
        cv2.circle(lining_img, (point_list_inner[i][1], point_list_inner[i][0]), 1, (0, 0, 255), -1)


    # print(f'count={cnt}')
    # print(f'point_list_outer={point_list_outer}')
    # print(f'len(point_list_outer_list)={len(point_list_outer)}')


    # csv 파일로 데이터 저장
    # with open(r'./data_csv/n1.csv', 'w', newline='\n') as f:                             # 파일명, 저장 위치에 대한 정책 필요!!!!!!!!!!!!!!!!!
    #     write = csv.writer(f)           # plt.plot(df[1], 512-df[0]) < 제대로 보려면...
    #     write.writerows(point_list_outer)     # 외곽
    #     write.writerows(' ')
    #     write.writerows(point_list_inner)    # 내곽

    # now_filename = os.path.basename(sys.argv[0])
    # eddef.imshow(now_filename[:-3], lining_img)
    # eddef.imshow(now_filename[:-3], np.hstack([img_gb, img_tc, lining_img]))

    # cv2.imshow( data_eye[:-4], np.hstack([img_gb, img_tc, lining_img]))
    # cv2.waitKey(0)

    # 변환 결과만 확인
    # plt.imshow(lining_img, cmap='gray')
    # plt.xlabel(data_eye[:-4])
    # plt.show()

    # 내외곽 획득 좌표수
    print(f'\n외곽 : {len(point_list_outer)}, 내곽 : {len(point_list_inner)}')

    # Gausianblur, canny, 최종결과 같이 확인
    eddef.imshow(data_eye[:-4], np.hstack([img_gb, img_tc, lining_img]))