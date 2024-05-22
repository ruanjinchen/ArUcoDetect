# ArUco二维码检测
该工程完成了使用chessboard对相机进行内参标定，然后使用标定数据，ArUco的二维码进行检测，返回二维码在相机坐标系下的位姿
## 步骤一：内参标定
先完成相机内参标定，可以直接使用手眼标定拍摄的照片，程序会自动完成标定并生成内参
```commandline
python 1.Calibration.py
```
## 步骤二：检测
可以修改[ArUco Detect](./2.ArUco_Detect.py)第30和31行的曝光和增益来调整画面的亮度和对比度。
修改[ArUco Detect](./2.ArUco_Detect.py)第59行的marker length，这个参数是二维码的实际边长，单位是米，然后将Realsense连接到电脑
```commandline
python 2.ArUco_Detect.py
```