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


font = cv2.FONT_HERSHEY_SIMPLEX  # 设置显示字体

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

    # 列出所有的id和检测出来的边框角点
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)  # 0.05是
        for i in range(rvec.shape[0]):
            # 对每个检测到的标记处理其旋转向量
            R, _ = cv2.Rodrigues(rvec[i, 0, :])  # 现在 rvec[i, 0, :] 是一个1x3的旋转向量

            # 从旋转矩阵计算欧拉角 ZYX
            yaw = np.rad2deg(float(np.arctan2(R[1, 0], R[0, 0])))
            pitch = np.rad2deg(float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
            roll = np.rad2deg(float(np.arctan2(R[2, 1], R[2, 2])))

            x = f"{roll:.3f}"
            y = f"{pitch:.3f}"
            z = f"{yaw:.3f}"

            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, 0, :], tvec[i, 0, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)

            # 显示ID，rvec,tvec, 旋转向量和平移向量
            cv2.putText(frame, "Id: " + str(ids[i]), (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "rvec: " + x + '  ' + y + '  ' + z + '  ', (10, 160), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "tvec: " + str(tvec[i, 0, :]), (10, 260), font, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            # 发送数据到socket
            message = ",".join(map(str, [tvec[i, 0, 0], tvec[i, 0, 1], tvec[i, 0, 2], x, y, z]))
            socket.send_string(message)


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
