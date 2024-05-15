import numpy as np
import zmq
import time
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import yaml


file_path = "./标定文件.yaml"

with open(file_path, "r") as file:
    parameter = yaml.load(file.read(), Loader=yaml.Loader)
    mtx = parameter['camera_matrix']
    dist = parameter['dist_coefficients']
    camera_u = parameter['camera_u']
    camera_v = parameter['camera_v']
    mtx = np.array(mtx)
    dist = np.array(dist)

# 打开笔记本摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

pipeline = rs.pipeline()
config = rs.config()
# 使用默认的配置,设置彩色流
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
# 启动图像
pipeline.start(config)

context = zmq.Context()
socket = context.socket(zmq.PUB)
host = "127.0.0.1"
port = 5000
address = f"tcp://{host}:{port}"
socket.bind(address)

# num = 0
while True:
    start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    # ret, frame = color_frame.read()
    # operations on the frame come here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)

    #    if ids != None:
    if ids is not None:

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # Estimate pose of each marker and return the values rvet and tvec---different
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        #        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #Draw Axis
        #        aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers

        for i in range(rvec.shape[0]):
            rvec_degree = np.rad2deg(rvec[i, :, :])
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
        # 显示ID，rvec,tvec, 旋转向量和平移向量
        cv2.putText(frame, "Id: " + str(ids), (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "rvec: " + str(rvec_degree[ :, :]), (10, 160), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "tvec: " + str(tvec[i, :, :]), (10, 260), font, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        data = tvec[i, :, :]
        # print("tvec:", tvec)
        # print("rvec_degree:", rvec_degree)

        # Correctly accessing the first element from each nested array and converting each to string
        # message = ",".join(map(str, [tvec[i, 0, 0], tvec[i, 0, 1], tvec[i, 0, 2], rvec_degree[0, 0], rvec_degree[0, 1], rvec_degree[0, 2]]))
        #
        # socket.send_string(message)
        # time.sleep(1)


    else:

        cv2.putText(frame, "No Ids", (10, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    end = time.time()
    # 计算帧率并显示
    cv2.putText(frame, "rate: " + str(1 / (end - start)), (10, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        #        num = num + 1
        #        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)