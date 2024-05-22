import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np
import yaml
import time
import cv2
import zmq


file_path = "./calibration_result.yaml"

with open(file_path, "r") as file:
    parameter = yaml.load(file.read(), Loader=yaml.Loader)
    mtx = parameter['camera_matrix']
    dist = parameter['dist_coefficients']
    camera_u = parameter['camera_u']
    camera_v = parameter['camera_v']
    mtx = np.array(mtx)
    dist = np.array(dist)

font = cv2.FONT_HERSHEY_TRIPLEX

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)

device = pipeline_profile.get_device()
sensor = device.query_sensors()[1]
exposure = 150  # 曝光
gain = 0  # 增益
sensor.set_option(rs.option.exposure, exposure)
sensor.set_option(rs.option.gain, gain)

window_name = "frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

context = zmq.Context()
socket = context.socket(zmq.PUB)
host = "127.0.0.1"
port = 5000
address = f"tcp://{host}:{port}"
socket.bind(address)

while True:
    start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    # 列出所有的id和检测出来的边框角点
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)  # 0.05是实际大小
        for i in range(rvec.shape[0]):
            # 对每个检测到的标记处理其旋转向量
            R, _ = cv2.Rodrigues(rvec[i, 0, :])  # 现在 rvec[i, 0, :] 是一个1x3的旋转向量
            # 从旋转矩阵计算欧拉角 ZYX
            yaw = np.rad2deg(float(np.arctan2(R[1, 0], R[0, 0])))
            pitch = np.rad2deg(float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
            roll = np.rad2deg(float(np.arctan2(R[2, 1], R[2, 2])))
            rx = f"{roll:.3f}"
            ry = f"{pitch:.3f}"
            rz = f"{yaw:.3f}"
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, 0, :], tvec[i, 0, :], 0.03)
            # aruco.drawDetectedMarkers(frame, corners)  # 调整不了边框粗细
            for corner in corners:
                cv2.polylines(frame, [np.int32(corner)], True, (0, 255, 0), 3)
            X = tvec[i, 0, :][0] * 1000
            Y = tvec[i, 0, :][1] * 1000
            Z = tvec[i, 0, :][2] * 1000
            cv2.putText(frame, "ID: " + str(ids[i]),
                        (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "RPY(Degree): " + rx + ', ' + ry + ', ' + rz,
                        (10, 160), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, " TRANS(MM): " + f"{X:.3f}" + ', ' + f"{Y:.3f}" + ', ' + f"{Z:.3f}",
                        (10, 210), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # 使用ZMQ发布数据
            message = ",".join(map(str, [tvec[i, 0, 0], tvec[i, 0, 1], tvec[i, 0, 2], rx, ry, rz]))
            socket.send_string(message)

    else:
        cv2.putText(frame, "No Ids", (10, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    end = time.time()
    # 计算帧率并显示
    cv2.putText(frame, "rate: " + str(1 / (end - start)), (10, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:  # 按esc键退出
        print('ESC BREAK...')
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
