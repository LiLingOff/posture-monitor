# posture-monitor

高中科展「基於雙目對極幾何之即時三維坐姿監測系統」的程式部分。用一顆正面相機加一組雙目模組拍攝坐姿，標定相機後做3D三角測量算出坐姿角度，即時給回饋。

目前完成度：**相機標定**（含應付雙目視野重疊區域小的ChArUco方案）、**人體關鍵點偵測骨架**（trt_pose + TensorRT，尚未在真實硬體上跑過）、**3D三角測量與坐姿角度計算**（θ_CA、θ_sym已實作並用合成資料驗證，θ_KA公式未定）。個人校正基準、閾值判定、LED/蜂鳴器即時回饋這幾塊還沒開始。

## 兩台機器、兩種角色

開發在一台沒有NVIDIA GPU的Windows機器上，實際部署跑在Jetson Orin Nano（JetPack 6、Python 3.10、CUDA 12）。這件事直接影響了程式怎麼寫：

- 標定模組全部是CPU運算（OpenCV），兩邊都能跑，測試也用合成影像驗證，不需要接相機。
- 人體關鍵點偵測需要torch+TensorRT+trt_pose，這些只在Jetson上裝得起來。程式裡把「不需要GPU的邏輯」（關鍵點資料結構、RMSE計算、延遲統計）跟「真的要跑推論的部分」切開，torch的import全部延遲到方法內部才執行，所以就算這台機器沒裝torch，`import pose` 也不會壞掉，測試照樣能跑。

看到程式裡有些地方特別繞（例如`engine.py`的lazy import），原因就是這個。

## 安裝

```
pip install -r requirements.txt
```

## 測試

```
python -m pytest tests/ -v
```

全部測試不需要接相機、不需要GPU。標定的準確度是這樣驗證的：已知一組相機內參和基線長度，用homography把一張正面棋盤格/ChArUco板圖片「合成」成該相機在特定姿態下拍到的樣子（這在數學上等價於真實相機拍平面標的物），餵進標定演算法，檢查算出來的內參/基線跟原本設定的差多少。這樣才是真的在測演算法本身，不是只測「有沒有丟例外」。

## 標定

### 拍照

正面相機：

```
python -m src.calibration.capture mono --camera 0 --target-count 40
```

雙目模組是左右兩顆各自獨立的USB相機：

```
python -m src.calibration.capture stereo --left-camera 1 --right-camera 2 --target-count 20
```

雙目模組是單一USB裝置、左右眼合併輸出成一張畫面（常見的低價雙目模組規格）：

```
python -m src.calibration.capture stereo --single-device --left-camera 1 --target-count 20
```

`--left-camera`這時填的是那顆裝置的index。畫面是上下拼接不是左右拼接的話加`--vertical-split`，左右眼顛倒加`--swap-lr`。

拍攝視窗會即時畫出偵測到的角點，按空白鍵存檔、ESC結束；拍完會停在最後一幀等按鍵才關，不會拍完瞬間視窗就消失。

相機index每台機器都不一樣，重灌系統或換USB孔位常常會變。Linux上先跑：

```
v4l2-ctl --list-devices
```

沒裝先`sudo apt install v4l-utils`。同時檢查一下`groups`裡有沒有`video`，沒有的話`cv2.VideoCapture`直接開不了，要`sudo usermod -aG video $USER`後重新登入。

棋盤格預設9×6內角點、25mm方格，用`--cols --rows --square-size-mm`調整。

### 雙目視野重疊區域太小，兩顆鏡頭永遠拍不到完整棋盤格

一般棋盤格標定要求整塊板子同時完整入鏡，如果雙目模組兩顆鏡頭本身視野重疊就小（基線寬、或鏡頭本身向內聚焦），這個條件不管怎麼調距離都達不到。

解法是換成ChArUco板：板上每個角點都有獨立ID（靠嵌在方格裡的ArUco標記辨識），標定時只要左右畫面有**足夠多共同角點**就能用，不需要整塊板子同時入鏡。這不是理論上的權宜之計——`tests/test_charuco_calibration.py`裡有一個測試專門驗證這件事：合成的兩台相機在整個測試過程中都沒有任何一幀能看到完整的63個角點，標定依然把基線長度跟旋轉角度精確地還原出來。

自己列印用：

```
python -m src.calibration.capture board --out data/charuco_board.png --squares-x 10 --squares-y 8 --square-size-mm 25 --marker-size-mm 18
```

列印時選「實際尺寸」，不要讓印表機自動縮放去符合紙張——`--square-size-mm`這個數字要跟印出來的板子量測結果一致，標定才會準。

如果是買現成的ChArUco板（例如FRC機器人界常用的AndyMark，規格通常印在包裝上），直接照包裝上的Checker Size、Marker Size、字典填參數就好，不用自己印：

```
python -m src.calibration.capture stereo --single-device --left-camera 1 --charuco --squares-x 15 --squares-y 11 --square-size-mm 15 --marker-size-mm 11 --dictionary DICT_4X4_250
```

拍攝時加`--charuco`（跟其餘旗標如`--single-device`並用）。畫面會顯示目前偵測到幾個角點、左右共同角點數；共同角點數要達到門檻（預設6，`--min-shared-corners`調整）才允許存檔。

如果畫面上完全沒有偵測到任何角點（棋盤格看得到、但沒有綠色標記疊上去），大概是板子用的是OpenCV 4.6以前的舊版ArUco標記排列——現成板子常有這問題。兩邊指令都加`--legacy-pattern`再試。

### 算標定參數

```
python -m src.calibration.cli mono --images data/calibration_images/front
python -m src.calibration.cli stereo --left-images data/calibration_images/stereo_left --right-images data/calibration_images/stereo_right
```

用ChArUco拍的要加`--charuco`，並且帶上跟拍攝時同一組`--squares-x --squares-y --square-size-mm --marker-size-mm --dictionary`（跟拍照時對不上，角點ID跟物理座標會對錯，標定結果會是錯的但不一定報錯，這種錯誤不容易發現）：

```
python -m src.calibration.cli stereo --charuco --squares-x 15 --squares-y 11 --square-size-mm 15 --marker-size-mm 11 --dictionary DICT_4X4_250 --left-images data/calibration_images/stereo_left --right-images data/calibration_images/stereo_right
```

結果存成`.npz`（相機內參、畸變係數，雙目的話還有`R/T/R1/R2/P1/P2/Q`），供之後三角測量模組直接讀。單眼目標重投影誤差<0.3px、雙目<0.5px，超過只會印警告、照樣把結果存下來——不是因為無所謂，是因為機械式擋著不給用反而會讓人略過警告字面、改用更寬鬆的門檻重跑，不如把選擇權留給人看數字判斷。

## 人體關鍵點偵測 + TensorRT優化

原研究文件寫的是OpenPose Body_25，但OpenPose（Caffe架構）大約2021年後就沒再更新，官方最高只支援到CUDA 10，在JetPack 6的CUDA 12環境下裝不太起來。改用[trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)——NVIDIA自己為Jetson做的TensorRT pose函式庫，方法學上一樣（部署基準測試、FP16加速、精度比對、必要時退回FP32），只是換了實際的偵測模型。

這裡沒有Jetson可以跑，所以`src/pose/`能寫、能測的只有不碰GPU的部分：關鍵點資料結構、RMSE計算、延遲統計、CLI參數解析，用假的引擎（`tests/pose_fakes.py`）餵資料驗證邏輯。真正的推論、TensorRT engine轉換要在Jetson上跑，程式碼寫好了但沒人親自驗證過會不會動——這點要老實講清楚，不是「應該可以」就當作完成。

Jetson上手動安裝：

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt && python3 setup.py install --user

git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose && python3 setup.py install --user

cp trt_pose/tasks/human_pose/human_pose.json <repo>/data/pose_models/
```

PyTorch要裝NVIDIA官方的Jetson專用wheel，不是`pip install torch`（到 https://forums.developer.nvidia.com/t/pytorch-for-jetson 查JetPack 6對應版本）。checkpoint（`.pth`）從trt_pose的model zoo手動下載，放到`data/pose_models/`。

跑基準測試（第一次跑fp16會花幾分鐘建TensorRT engine並存快取，之後直接讀）：

```
python -m src.pose.cli benchmark --precision fp32 --front-camera 1 --stereo-camera 3 --single-device
python -m src.pose.cli benchmark --precision fp16 --front-camera 1 --stereo-camera 3 --single-device
```

比對FP32/FP16的關鍵點座標精度：

```
python -m src.pose.cli compare-precision --camera 1 --samples 30 --rmse-threshold-px 3.0
```

RMSE超過門檻只印警告、不會擋著不給用，代表這種情況下建議退回FP32——`--rmse-threshold-px`現在的3.0px是憑經驗猜的，不是照實際影像解析度算出來的，跑完實測數字後應該要調整。

## 3D三角測量與坐姿角度

`src/geometry/`把雙目標定輸出（`R1/R2/P1/P2/Q`）跟pose模組輸出（`PersonKeypoints`）接起來，算出3D關鍵點跟坐姿角度。

三角測量流程：`cv2.undistortPoints`用`R1/R2`+`P1/P2`把原始像素點去畸變並套用校正轉換，再丟進`cv2.triangulatePoints`。這一步很容易寫錯——直接把原始像素座標丟進`triangulatePoints`，因為`P1/P2`是定義在校正後的座標系裡，不是原始相機座標系，算出來的3D點會是錯的但通常不會報錯，錯誤不容易發現。`tests/test_triangulation.py`用已知3D點反推驗證這條路徑。

角度公式來自前一屆科展的前作（《整合多視角姿態估測與幾何特徵量化之即時坐姿監測系統研究》）。前作只有兩顆獨立單眼相機（正面0°＋側面45°），沒有真正的立體視覺，靠信心度加權融合兩邊2D結果；θ_CA（頸椎前傾角）原本要用`/sin(45°)`補償單一45°相機缺乏深度資訊造成的透視壓縮。這次升級成真正的雙目對極幾何後，這個補償係數不需要了——直接算耳朵→肩膀的3D向量，投影到矢狀面，跟垂直軸算帶號夾角就是θ_CA；θ_sym（肩膀水平角）同理，左右肩連線投影到冠狀面跟水平軸算夾角。

```python
from geometry import triangulate_person_keypoints
from geometry.posture_angles import theta_ca, theta_sym

keypoints_3d = triangulate_person_keypoints(stereo_calib, left_keypoints, right_keypoints)
theta_ca(keypoints_3d)   # 頸椎前傾角（度），side="right"/"left"
theta_sym(keypoints_3d)  # 肩膀水平角（度）
```

θ_KA前作完全沒提到，公式還沒定案，`geometry.posture_angles.theta_ka()`目前是`NotImplementedError`。個人校正基準（θ_offset）跟10°/5°/20px這類判定門檻是前作拿來觸發警示用的執行期邏輯，不屬於幾何計算，這裡沒做，留給之後的監測/回饋模組。

## 目錄

```
src/calibration/
  chessboard.py         棋盤格角點偵測
  charuco.py            ChArUco板角點偵測（容許只看到板子一部分）
  mono_calibration.py   正面相機張氏標定
  stereo_calibration.py stereoCalibrate + stereoRectify
  capture.py            接相機互動式拍照
  cli.py                讀取已拍影像、跑標定計算
src/pose/
  topology.py           關鍵點命名（COCO 18點，含neck）
  keypoints.py          PersonKeypoints資料結構 + RMSE計算
  preprocess.py          畫面前處理（純numpy，不碰torch）
  engine.py              PoseEngine介面 + trt_pose實作（torch延遲import）
  benchmark.py           延遲量測、FP32/FP16精度比對
  cli.py                跑基準測試/精度比對
src/geometry/
  triangulation.py       雙目3D三角測量核心
  keypoints3d.py         PersonKeypoints3D + 單人關鍵點三角測量
  angles.py              通用角度數學（跟研究無關的純幾何）
  posture_angles.py     θ_CA/θ_sym實作，θ_KA是NotImplementedError stub
tests/
  synthetic.py           合成測試影像（棋盤格標定用）
  charuco_synthetic.py   合成測試影像（ChArUco標定用）
  geometry_synthetic.py 合成雙目標定+3D點投影（三角測量用）
  pose_fakes.py          假引擎，測pose模組不需要GPU
```

## 待確認的假設

寫這些的時候手上沒有真實硬體/裝好的trt_pose可以核對，先記在這裡：

- `pose/topology.py`假設trt_pose預設的`human_pose.json`有第18個`neck`關鍵點。裝好trt_pose後用`json.load(open("human_pose.json"))["keypoints"]`核對一下，順序或有無不同的話改那個tuple就好，其他模組不受影響。
- `pose/engine.py`裡`torch2trt`/`trt_pose`的呼叫方式（`fp16_mode`參數、`TRTModule`存讀、`ParseObjects`用法）是照公開資料寫的，沒有實機驗證過，到Jetson上八成要對照實際clone下來的原始碼調整。
- ChArUco的`--min-shared-corners`預設6、pose的`--rmse-threshold-px`預設3.0，都是憑經驗抓的起始值，不是算出來的，跑過實機數據後應該回頭調整。
- `geometry/posture_angles.py`假設雙目校正後的座標系符合OpenCV慣例（X右、Y下、Z深度）且相機大致水平架設、沒有明顯翻滾角，沒有額外做座標系旋轉校正。如果實際架設角度偏差較大，θ_CA/θ_sym算出來的角度會系統性地偏移，需要額外處理（或比照最初實驗步驟文件的做法，把殘餘傾角記錄下來當統計分析的共變量）。
