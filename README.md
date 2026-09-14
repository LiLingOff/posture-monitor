# posture-monitor

雙目對極幾何即時三維坐姿監測系統。

## 安裝

```
pip install -r requirements.txt
```

## 測試

```
python -m pytest tests/ -v
```

## 拍標定影像

正面相機：

```
python -m src.calibration.capture mono --camera 0 --target-count 40
```

雙目模組如果是左右兩顆各自獨立的 USB 相機：

```
python -m src.calibration.capture stereo --left-camera 1 --right-camera 2 --target-count 20
```

雙目模組如果是自帶雙目、單一 USB 裝置輸出一張合併畫面（常見規格），改用 `--single-device`，`--left-camera` 這時填那顆裝置的 index：

```
python -m src.calibration.capture stereo --single-device --left-camera 1 --target-count 20
```

畫面是上下拼接不是左右拼接的話加 `--vertical-split`，左右眼顛倒加 `--swap-lr`。

視窗會即時畫出偵測到的棋盤格角點，按空白鍵存檔、ESC 結束，拍完會停在最後一幀等按鍵才關。相機 index 每台機器不一樣，Linux 上用 `v4l2-ctl --list-devices` 查（沒裝先 `sudo apt install v4l-utils`），順便確認 `groups` 裡有沒有 `video`，沒有的話 `cv2.VideoCapture` 開不了。

棋盤格預設 9×6 內角點、25mm 方格，用 `--cols --rows --square-size-mm` 調整。

## 算標定參數

```
python -m src.calibration.cli mono --images data/calibration_images/front
python -m src.calibration.cli stereo --left-images data/calibration_images/stereo_left --right-images data/calibration_images/stereo_right
```

結果存成 `.npz`，之後三角測量模組直接讀。單眼目標重投影誤差 <0.3px，雙目 <0.5px，超過會印警告。

## 人體關鍵點偵測 + TensorRT優化

用 [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)（NVIDIA為Jetson做的TensorRT pose函式庫），不是原版OpenPose——OpenPose在JetPack 6/CUDA 12上已經不維護，裝不太起來。

這裡（Windows開發機）沒有GPU，只能寫、測純邏輯的部分（`python -m pytest`不需要裝torch）。實際推論、TensorRT engine轉換要在Jetson上做。

Jetson上手動安裝：

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt && python3 setup.py install --user

git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose && python3 setup.py install --user

cp trt_pose/tasks/human_pose/human_pose.json <repo>/data/pose_models/
```

PyTorch要裝NVIDIA官方Jetson wheel，不是`pip install torch`（查 https://forums.developer.nvidia.com/t/pytorch-for-jetson ）。checkpoint（`.pth`）從trt_pose的model zoo手動下載，放到`data/pose_models/`。

跑基準測試（第一次跑fp16會花幾分鐘建TensorRT engine，之後讀快取）：

```
python -m src.pose.cli benchmark --precision fp32 --front-camera 1 --stereo-camera 3 --single-device
python -m src.pose.cli benchmark --precision fp16 --front-camera 1 --stereo-camera 3 --single-device
```

比對FP32/FP16關鍵點精度（RMSE超過門檻只印警告，不會擋著不給跑，代表建議退回FP32）：

```
python -m src.pose.cli compare-precision --camera 1 --samples 30 --rmse-threshold-px 3.0
```

有個地方要注意：`topology.py`裡假設trt_pose的`human_pose.json`有第18個`neck`關鍵點，這是抓trt_pose官方預設拓樸的印象，實際裝起來後要核對一下`json.load(open("human_pose.json"))["keypoints"]`，不一樣的話改那個tuple就好，其他地方不用動。

## 目錄

```
src/calibration/
  chessboard.py         角點偵測
  mono_calibration.py   正面相機張氏標定
  stereo_calibration.py stereoCalibrate + stereoRectify
  capture.py            接相機拍照
  cli.py                跑標定計算
src/pose/
  topology.py           關鍵點命名
  keypoints.py           PersonKeypoints + RMSE計算
  preprocess.py           畫面前處理（不碰torch）
  engine.py              PoseEngine介面 + trt_pose實作（torch延遲import）
  benchmark.py            延遲量測、FP32/FP16比對
  cli.py                跑基準測試/精度比對
tests/
  synthetic.py           合成測試影像用（標定）
  pose_fakes.py           假引擎，測pose模組不需要GPU
```
