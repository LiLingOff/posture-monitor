# posture-monitor

高中科展「基於雙目對極幾何之即時三維坐姿監測系統」的程式部分。用一組雙目模組拍攝坐姿，完成相機標定後把左右兩眼看到的關鍵點還原成真實3D座標，再計算頸椎前傾角與肩膀水平角。前作的正面相機已決定不接，理由見「相機擺哪裡」一節。

| 模組 | 狀態 | 內容 |
|---|---|---|
| `src/calibration` | 可用 | 棋盤格／ChArUco，單眼＋雙目，已用合成資料驗證準確度 |
| `src/pose` | 可用（fp32） | Lightweight OpenPose。Jetson 實測 123.7ms/幀，權重載入與拓樸核對都已確認；fp16/TensorRT 尚未啟用 |
| `src/geometry` | 可用 | 三角測量、θ_CA／θ_sym；θ_KA 公式未定 |
| 個人基準／平均／記錄 | 可用 | θ_offset 校正、移動平均、逐幀 CSV |
| 閾值判定與回饋 | 未開始 | 判定邏輯、LED／蜂鳴器 |

資料沿著這條路徑改變形態：標定先算出相機參數存成 `.npz`，拍攝取得 BGR 影格，關鍵點偵測把影格轉成18個關節的像素座標，三角測量再用標定參數把左右兩組像素座標還原成 mm 為單位的3D座標，最後由3D向量算出角度。判定與回饋接在角度之後，目前還沒開始。

## 環境

| | 開發機 | 部署機 |
|---|---|---|
| 平台 | Windows，無NVIDIA GPU | Jetson Orin Nano |
| 系統 | — | JetPack 6、Ubuntu 22.04、Python 3.10、CUDA 12 |
| 能執行的範圍 | 標定、三角測量、全部測試 | 上述全部＋關鍵點偵測推論 |

這個環境差異直接影響程式寫法：`src/pose/engine.py` 裡所有 `import torch` 都放在方法內部，而非檔案開頭。開發機無法安裝這些套件，卻仍需要能 `import pose` 執行測試，因此把 import 延後到真正使用時才執行——沒有安裝 torch 的機器上 `import pose.engine` 依然會成功，只有實際建立 `LightweightOpenPoseEngine` 實例時才需要這些套件。`tests/test_pose_engine_import.py` 以測試鎖定這項性質。程式中若有寫法看起來迂迴，原因通常在此。

## 安裝與測試

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

226個測試，全部使用合成資料，不需要相機或GPU。

標定準確度的驗證方式如下：給定一組已知的相機內參與基線長度，用單應變換把正面棋盤格圖合成為該相機在特定姿態下拍到的畫面，輸入標定演算法，再檢查還原出來的參數與真值相差多少。這在數學上是嚴格等價而非近似——標定板是平面，`Z=0` 讓投影方程式退化成單應變換。三角測量的測試改用 `cv2.projectPoints` 直接投影已知3D點（單應變換的前提是共平面，而三角測量要驗證的正是非共平面的點），合成資料下還原誤差 0.000mm。

判準是還原出來的焦距誤差 <5%、基線長度誤差 <10%，而非只檢查有沒有拋出例外。

## 資料型別

| 型別 | 欄位 | 形狀 | 單位 |
|---|---|---|---|
| `StereoCalibrationResult` | `camera_matrix_left/right`、`dist_coeffs_*` | (3,3)、(5,) | — |
| | `R`、`T` | (3,3)、(3,1) | T 為 mm |
| | `R1/R2`、`P1/P2`、`Q` | (3,3)、(3,4)、(4,4) | 三角測量使用 `P1/P2` |
| `PersonKeypoints` | `points`、`confidences` | (18,2)、(18,) | 像素 |
| `PersonKeypoints3D` | `points` | (18,3) | **mm** |

三個貫穿全系統的慣例：

- **未偵測到的關鍵點一律填 NaN**，不用0也不用-1——0是合法座標，拿它表示缺失會讓錯誤資料混入後續計算而不被察覺
- **長度單位全系統統一為 mm**，源頭是標定時的 `--square-size-mm`。這個數字填錯，整個系統的長度尺度就跟著錯
- **精度未達標只印出警告，資料明顯異常才拋出例外**。前者代表結果可用但品質有待評估，後者代表這個結果沒有意義

## 標定：拍攝

| 情境 | 指令 |
|---|---|
| 單眼（本專案用不到，工具保留） | `python -m src.calibration.capture mono --camera 0 --target-count 40` |
| 雙目＝兩顆獨立USB相機 | `python -m src.calibration.capture stereo --left-camera 1 --right-camera 2` |
| 雙目＝單一USB裝置合併輸出 | `python -m src.calibration.capture stereo --single-device --left-camera 1` |
| 產生可列印的ChArUco板 | `python -m src.calibration.capture board --out data/charuco_board.png --squares-x 10 --squares-y 8 --square-size-mm 25 --marker-size-mm 18` |
| 查詢相機支援哪些解析度 | `python -m src.calibration.capture probe --camera 0` |

| 旗標 | 用途 |
|---|---|
| `--charuco` | 改用ChArUco板（可與其餘旗標並用） |
| `--single-device` | 雙目為單一裝置、左右眼合併在同一張畫面；`--left-camera` 填該裝置index |
| `--vertical-split` | 合併畫面為上下拼接而非左右並排 |
| `--swap-lr` | 左右眼顛倒 |
| `--width` / `--height` | 指定擷取解析度，雙目模組必填，理由見下節 |
| `--legacy-pattern` | 板子使用OpenCV 4.6以前的ArUco排列（市售現成板子常見） |
| `--min-shared-corners` | 左右需有幾個共同角點才允許存檔，預設6 |

操作方式：空白鍵存檔、ESC結束。視窗會即時繪出偵測到的角點與左右共同角點數量，拍攝完成後會停在最後一幀等待按鍵。資料夾中已有足夠張數時會直接跳過，要整批重拍請先清空資料夾；只刪掉其中幾張再重跑的話會從現有的最大編號往後接，不會蓋到留下來的影像。

有指定 `--width/--height` 時，程式會先核對既有影像的尺寸與這次要拍的是否一致，對不上直接拒絕並要求清空資料夾。換過解析度卻沿用舊影像的話，標定會算出一組綁在錯誤解析度上的內參，而且完全不會有徵兆。

拍攝要領：板子要涵蓋畫面中心與四個角落，並變換距離與傾斜角度；若都在同一個位置拍攝，畸變係數會求解不準。列印時選擇「實際尺寸」不要自動縮放，印完用尺量出一格的實際長度，以量到的數字填入 `--square-size-mm`。

市售現成板子依包裝規格填寫即可，例如 AndyMark 那款是 `--squares-x 15 --squares-y 11 --square-size-mm 15 --marker-size-mm 11 --dictionary DICT_4X4_250`。

| 症狀 | 處理 |
|---|---|
| 看得到板子但沒有標記疊上去 | 加上 `--legacy-pattern` |
| **`stereo left`／`stereo right` 顯示不同場景** | 取得的畫面是單眼影像被切成兩半。用 `--width/--height` 指定並排模式的解析度（見下節） |
| 共同角點數量一直不足 | 先確認上一項；再把板子移向兩鏡頭視野的重疊區，或調低 `--min-shared-corners` |
| `can't open camera by index` | 程式會當場去查 `/dev/videoN` 在不在、權限夠不夠、本使用者有沒有別的程序開著它，把查到的直接寫在錯誤訊息裡。三者都正常時才需要 root 權限的檢查（`sudo fuser -v`、`dmesg`）。Jetson 上已知的成因是 PhotonVision 服務獨佔相機、重新插拔後節點編號移位，以及上一次執行剛結束、核心還沒放掉 USB 介面 |

### 雙目模組必須指定解析度

OpenCV 在 V4L2 下若不指定解析度就會採用驅動的預設模式，通常是 640×480。雙目模組的左右眼並排輸出往往只存在於特定的寬解析度模式（例如 2560×720），預設模式提供的是單眼或裁切後的畫面——把它切成兩半會得到兩塊**不重疊**的裁切區域，看起來像兩個不同場景，標定永遠無法取得足夠的共同角點。

先查詢相機支援哪些模式：

```
python -m src.calibration.capture probe --camera 0
```

這個指令會逐一嘗試常見解析度，印出實際取得的尺寸、寬高比與量測到的張數率，並標示哪些是驅動退回的模式而非原生支援。`v4l2-ctl -d /dev/video0 --list-formats-ext` 也能查詢，但某些UVC裝置不會完整回報格式清單。

挑選模式時有三個考量：優先選擇驅動原生支援的模式，不要選退回來的；張數率要足夠即時監測使用，解析度再高，關鍵點偵測也會先等比例縮放到高度 256 才輸入網路；最後一點最重要——**標定與執行時必須使用同一個解析度**。內參 `fx/fy/cx/cy` 的單位是像素，數值綁定於當時的解析度，在 3840×1200 完成標定卻用 2560×720 執行，整組參數就失效，而且不會有任何錯誤訊息。

選定後拍攝時帶上該解析度：

```
python -m src.calibration.capture stereo --single-device --left-camera 0 --width 2560 --height 720 --charuco ...
```

程式開啟相機後會印出實際取得的解析度，未能達到要求時會明確說明。單一裝置模式下還會檢查第一幀的長寬比，不符合並排輸出的比例就發出警告。這個警告是為了在拍攝當下擋下來——真要等到標定階段才發現，整批影像都得重拍。

### 為什麼必須使用 ChArUco

一般棋盤格標定要求整塊板子完整入鏡。`findChessboardCorners` 尋找的是完整的 N×M 網格，缺少一角整張就作廢，因為棋盤格的角點外觀完全相同，沒有完整網格就無法判斷哪個角點對應哪個位置。雙目的條件更嚴苛：同一時刻左右兩張影像都必須看到完整板子。

這組雙目模組兩顆鏡頭的視野重疊區很小，無論如何調整距離都達不到這個條件。ChArUco 板在每個白格內嵌一個獨一無二的 ArUco 標記，角點因此擁有獨立ID，左右影像只要有足夠多共同ID的角點就能建立對應關係，不需要整塊板子入鏡。

這並非理論上的權宜之計：`tests/test_charuco_calibration.py` 的測試中，合成的兩台相機在整個過程中都沒有任何一幀看到完整的63個角點，標定依然精確還原出基線長度與相對旋轉。

## 標定：計算參數

| 目標 | 指令 |
|---|---|
| 單眼 | `python -m src.calibration.cli mono --images data/calibration_images/front` |
| 雙目 | `python -m src.calibration.cli stereo --left-images data/calibration_images/stereo_left --right-images data/calibration_images/stereo_right` |
| ChArUco | 上述指令加上 `--charuco` 與拍攝時**完全相同**的板子參數 |
| 檢視已存的標定結果 | `python -m src.calibration.cli inspect` |

輸出 `.npz` 檔：內參、畸變係數；雙目另含 `R/T/E/F/R1/R2/P1/P2/Q`。目標重投影誤差為單眼 <0.3px、雙目 <0.5px。

板子參數與拍攝時不一致的話，角點ID會對應到錯誤的物理座標——標定結果是錯的，但通常不會出現錯誤訊息，這類錯誤很難發現。

算完會接著印一份合理性驗算（也可以之後用 `inspect` 單獨看）。RMS 低只代表標定在內部自洽，說不出參數物理上合不合理，所以另外拿結果去對照剛性雙目模組應有的樣子：兩顆鏡頭應幾乎平行、平移應幾乎只有 X 分量、像素應接近正方形、光心應在畫面中央附近、`|P2[0,3]|` 應等於 `fx × baseline`。最後一項串起內參、外參與校正三者，對不上代表某一段接錯了。

這些檢查抓不到的是**尺度**。`--square-size-mm` 填錯的話所有參數仍然自洽，只是整個系統的長度單位跟著錯。唯一的外部驗證是把基線長度拿去跟雙目模組的規格書對照。

誤差超標只印出警告、照常存檔。這樣設計並非認為誤差無所謂，而是因為機械式地擋著不讓使用，反而會讓人略過警告內容、改用更寬鬆的門檻重新執行，不如把數字完整呈現讓人自行判斷。

### 重投影誤差的計算曾經出錯

印出來的 RMS 與 `cv2.calibrateCamera` 自己回傳的數字一致，`tests/test_reprojection_error.py` 鎖定這項性質。

會特別針對這點寫測試，是因為這裡曾經算錯：原本沿用 OpenCV 官方教學的寫法 `cv2.norm(...) / len(projected)`，但 `cv2.norm` 回傳的是所有點誤差平方和再開根號，要得到 RMS 應該除以 `sqrt(N)` 而非 `N`。除以 `N` 會把誤差低估 `sqrt(N)` 倍——35個角點就是5.9倍，真實的 0.13px 會印成 0.02px。

後果是整個品質門檻失去作用：真實誤差 1.5px 的劣質標定會顯示成 0.25px，通過 <0.3px 的檢查。這類錯誤沒有任何外顯症狀，若不刻意與 OpenCV 自己的回傳值對照就永遠不會發現。

## 關鍵點偵測 + TensorRT

原研究文件寫的是 OpenPose Body_25。OpenPose 本體（Caffe架構）約2021年後就不再更新、官方最高支援到 CUDA 10，在 JetPack 6 的 CUDA 12 環境下難以安裝，所以改用 [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)——論文《Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose》的官方實作，是 OpenPose 方法本身的最佳化版本，同樣是 Bottom-up + PAF、同樣輸出含 `neck` 的 COCO 18 點。

中間一度打算用 trt_pose，最後放棄：它的權重掛在 Google Drive，連結長期處於無權限狀態（repo 內搜尋 download/permission 有 16 個 issue），換多台機器都下載不到，而該專案實質已停止維護。Lightweight OpenPose 的權重由 Intel 的伺服器直接提供，不需要登入：

```
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
```

方法學不變（部署基準測試、FP16加速、精度比對、必要時退回FP32）。相對於 trt_pose，這個選擇反而更貼近原研究文件寫的 OpenPose。

fp32 推論已在 Jetson 上跑通（2026-09-21）：權重載入沒有任何一層退回隨機初始值，上游的 `Pose.kpt_names` 與 `pose/topology.py` 逐一相符，單幀 123.7ms、單眼 456x256 約 78ms。fp16 與 TensorRT 轉換還沒啟用。

開發機這一側能測的仍然只有不依賴GPU的部分：關鍵點資料結構、前處理與座標還原、RMSE計算、延遲統計、拓樸核對、CLI參數解析，以假引擎（`tests/pose_fakes.py`）驗證邏輯。

Jetson 上的安裝步驟：

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt && python3 setup.py install --user && cd ..

mkdir -p third_party && cd third_party
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
cd ..

mkdir -p data/pose_models && cd data/pose_models
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
cd ../..
```

Lightweight OpenPose 沒有 `setup.py`，不能 pip 安裝，所以程式是把 clone 下來的目錄加進 `sys.path`（用 `--repo-dir` 指定，預設 `third_party/lightweight-human-pose-estimation.pytorch`）。

### PyTorch 一定要裝 Jetson 專用 wheel

一般的 `pip install torch` 在 Jetson 上**裝得起來、import 得進去，但一碰 CUDA 就失敗**：

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060).
Please update your GPU driver ...
```

這個訊息會誤導人去更新驅動。Jetson 的 GPU 驅動綁在 JetPack 的 BSP 裡，**不能單獨升級**——問題出在 PyPI 上的 wheel 編譯時對應的 CUDA 比 JetPack 提供的新。

解法是換成 NVIDIA 為 Jetson 編譯的 wheel（到 https://forums.developer.nvidia.com/t/pytorch-for-jetson 查詢 JetPack 版本對應的檔案），先移除既有的：

```
pip3 uninstall -y torch torchvision
```

在那之前，用 `--device cpu --precision fp32` 就能驗證整條流程。Lightweight OpenPose 本來就是為 CPU 設計的（論文標題是 Real-time 2D Multi-Person Pose Estimation on CPU），速度雖然慢，但關鍵點座標、三角測量、角度計算的正確性都驗得出來。

| 用途 | 指令 |
|---|---|
| 驗證安裝 | `python3 -c "import torch, torch2trt; print('ok')"` |
| 核對關鍵點順序 | `python3 -c "import sys; sys.path.insert(0,'third_party/lightweight-human-pose-estimation.pytorch'); from modules.pose import Pose; print(Pose.kpt_names)"` |
| CPU 驗證（不需要CUDA） | `python -m src.pose.cli benchmark --precision fp32 --device cpu --front-camera 0 --stereo-camera 0 --single-device --width 2560 --height 720 --frames 5 --warmup 2` |
| 延遲基準（FP32） | `python -m src.pose.cli benchmark --precision fp32 --front-camera 1 --stereo-camera 0 --single-device --width 2560 --height 720` |
| 延遲基準（FP16） | 同上改為 `--precision fp16`；第一次執行會花數分鐘建立TensorRT engine並存入快取 |
| 精度比對 | `python -m src.pose.cli compare-precision --camera 0 --samples 30 --rmse-threshold-px 3.0` |

### 換模型時最容易錯的一件事

Lightweight OpenPose 的關鍵點順序與 trt_pose **完全不同**：`neck` 在索引 1 而非 17，而且左右是先右後左。沿用舊索引不會出現任何錯誤訊息，耳朵的座標會被當成肩膀用，而算出來的角度仍然落在正常範圍。

所以 `engine.py` 在載入模型之前，會先拿上游 `modules/pose.py` 的 `Pose.kpt_names` 跟 `pose/topology.py` 逐一比對，對不上就直接拋例外。這個核對刻意排在 `import torch` 之前，開發機沒裝 torch 也測得到（`tests/test_pose_engine_topology_guard.py`）。

### 前處理

照上游 `demo.py::infer_fast` 實作：依高度等比例縮放到 256，再補邊到 stride 8 的整數倍。有兩點與直覺相反，寫錯不會報錯但結果全錯：

- **不做 BGR 轉 RGB。** 上游是直接把 OpenCV 讀進來的 BGR 餵給網路的。
- **正規化是 `(img - 128) / 256`**，不是 ImageNet 的 mean/std。

熱圖座標要換算回原始畫面：`(x * stride / upsample_ratio - pad) / scale`。這一段抽成純 numpy 的 `restore_keypoint_coordinates()`，用往返還原的方式測（`tests/test_pose_preprocess.py`）——漏掉任何一步都只會讓三角測量拿到系統性偏移的2D點，不會有徵兆。

等比例縮放順帶解決了先前記在待辦裡的問題：配合 trt_pose 時是直接把畫面拉成正方形，模型看到的人體是變形的。

### TensorRT 的輸入尺寸

補邊後的寬度取決於畫面長寬比，所以網路輸入尺寸由相機解析度唯一決定。2560×720 的合併畫面是 912×256，切開後的單眼 1280×720 是 456×256——**同一次執行就會同時用到兩種**，正面相機再加一種。

fp32 的網路是全卷積的，同一個模型吃任何尺寸。fp16 的 TensorRT engine 綁定單一輸入尺寸，所以每個尺寸各建一個、各存一份快取（檔名帶上尺寸，例如 `lightweight_openpose_fp16_912x256.pth`）。第一次遇到新尺寸要花數分鐘建置，`warmup(frame)` 可以在計時迴圈外先做掉，避免把建置時間算進延遲。

## 三角測量與坐姿角度

`src/geometry/` 將雙目標定的輸出（`P1/P2` 等）與 pose 模組的輸出（`PersonKeypoints`）銜接起來。

端到端量測用 repo 根目錄的 `posture.py`：

| 用途 | 指令 |
|---|---|
| 量測一次並印出完整診斷 | `python posture.py once --width 2560 --height 720` |
| 持續量測，逐幀更新 | `python posture.py live --width 2560 --height 720` |
| 取個人基準 | `python posture.py baseline --subject chenyue --out data/baselines/chenyue.json` |
| 量測並記錄 | `python posture.py live --baseline data/baselines/chenyue.json --log data/sessions/0923.csv` |
| 調整平均視窗 | 加上 `--window 60`（預設 30 幀，約 5 秒）|

`once` 會列出每個關鍵點的左右像素座標、3D座標、校正後的垂直視差，以及深度範圍、單幀誤差與兩個角度。加 `--all-keypoints` 可以看全部18點。

`live` 印的是**平均值**，單幀值放在括號裡：

```
θ_CA  +1.7±1.7(單幀 -2.7)  θ_sym  -5.6±0.4(單幀 -7.1)   735mm  8°  30/30幀  略過7
```

要看的是平均。2026-09-23 實機量測中受試者保持不動，θ_CA 的單幀值掃過 50 度（標準差 ±11.5°），而判定門檻是 10 度——單幀結果是雜訊。平均 30 幀之後是 ±1.7°。同一段資料的 θ_sym 單幀標準差只有 ±2.1°，本來就堪用；差別在 θ_CA 靠深度、θ_sym 不靠。

偵測失誤的幀會被擋在平均之外並計入「略過」。**這件事非做不可**：誤差公式只看距離，距離愈近算出來愈小，所以失誤跑到近處時印出的是「距離 74mm，誤差 ±0.1°」——資料最壞的時候誤差數字最好看，靠數值大小完全擋不住。判斷依據是深度是否落在合理範圍、角度用到的點左右配對對不對、共同關鍵點夠不夠。

Ctrl-C 結束時會印出整段的統計，那才是可以記錄下來的數字。

放在根目錄而不是 `python -m src.geometry.cli`，是因為 `src/geometry` 用絕對匯入（`from calibration...`）需要 `src/` 在 sys.path 上，而 `-m src.geometry.cli` 的 sys.path[0] 是 repo 根目錄。`posture.py` 先補路徑再匯入，與 `conftest.py` 給測試用的做法一致。

程式介面：

```python
from geometry.pipeline import measure_posture

m = measure_posture(stereo_calib, left_keypoints, right_keypoints)
m.theta_ca_deg           # 頸椎前傾角，頭往前伸為正
m.theta_sym_deg          # 肩膀水平角，右肩較高為正
m.reference_depth_mm     # 耳肩的深度，用來換算角度精度
m.theta_ca_precision_deg # 這個距離下的單幀誤差
m.depth_range_mm
m.max_abs_vertical_disparity_px
```

### 相機架在哪裡都可以

唯一的架設要求是**相機大致水平、沒有明顯翻滾角**，因為程式直接把相機的 Y 軸當作重力方向。**方位角不受限制**：矢狀面與冠狀面的法向量取自受試者自己的雙肩連線（`anatomical_axes`），不是相機的 X 軸與 Z 軸，所以雙目模組架在正面、側面或任何斜角都得到同一個角度。雙肩有一邊沒偵測到時退回相機軸，那等同於假設受試者正對相機。

這一步不是細節。只補償深度壓縮而仍用相機軸定義解剖平面的話，方位角會換個位置回來：雙肩連線在相機 X 軸上的分量被 `cos(方位角)` 壓縮而垂直分量不會，θ_sym 就被放大 `1/cos(方位角)` 倍，45° 斜角多報 41%。這跟前作除以 `sin(45°)` 是同一類的係數。2026-09-22 的實測中模組位於正面偏 29.8°，θ_CA 用相機軸算出 −11.44°（頭往後仰）、改用解剖軸是 +9.59°（頭往前傾），判定結論相反，而兩個數值看起來都很合理。

實際擺放要考慮的只剩遮擋：θ_CA 用的耳朵與肩膀從側面看最清楚，但太偏側面時遠側肩膀會被身體擋住，而雙肩正是解剖平面的來源。斜前方 30° 左右兩者都看得到。

### 角度的正負號代表什麼

用 `atan2` 而非 `arccos(內積)` 的唯一理由就是保留方向，所以方向的意義必須定義清楚。

| 角度 | 正 | 負 |
|---|---|---|
| θ_CA | 頭往前伸 | 頭往後仰 |
| θ_sym | 右肩較高 | 左肩較高 |

前作以 `θ_CA > 10°` 判定頭部前傾，對應的就是正值這一側。正負號搞反的話，後仰會被判成前傾，而數值大小完全一樣、看不出異常——`tests/test_posture_angles.py` 用已知幾何把兩個方向都鎖住。

θ_sym 的向量取 `left_shoulder - right_shoulder`，不是反過來。受試者面向相機時，**解剖學上的右肩會出現在影像的左半邊**（X較小）、左肩在右半邊（X較大），所以 `left - right` 才會指向 +X、雙肩等高時得到 0°。取反的話算出來永遠接近 ±180°。這個方向與前作的 `arctan[(y_L−y_R)/(x_L−x_R)]` 一致。

這一點原本在程式與合成測試裡同時弄反了——測試把右肩放在 +X，正好印證了程式的同一個誤解，所以全數通過。**是實機資料才抓出來的**（實測 `right.x=219 < left.x=418`，程式卻算出 +157°）。那組座標已經寫成回歸測試。

### 深度精度：這個系統最緊的一個環節

θ_CA 量的是耳朵與肩膀的**深度差**，而前傾 15° 時這個差只有約 31mm。能不能量到，取決於視差的解析度：

```
Z = fx × baseline / disparity        深度誤差 = Z² × 視差誤差 / (fx × baseline)
```

實機參數（fx 568、baseline 60mm、Z 約 1000mm）下，**視差每差 1px，深度就差 29mm**——跟要量的訊號同一個量級。

問題出在熱圖的解析度。上游 `extract_keypoints` 取的是整數 argmax，所以關鍵點只能落在熱圖網格上，換算回原始畫面是 `(stride / upsample_ratio) / scale` 個像素：

| 輸入高度 | 次像素精修 | 量化(原始px) | Z=1000mm 的深度跳動 |
|---|---|---|---|
| 256 | 關 | 5.63 | 164 mm |
| 256 | **開（預設）** | 0.23 | **7 mm** |
| 384 | 開 | 0.15 | 4 mm |
| 512 | 開 | 0.11 | 3 mm |

沒有次像素精修的話，量化本身就比訊號大 5 倍，深度只能跳著走，算出來的角度沒有意義。精修用的是拋物線內插（熱圖式關鍵點偵測的標準後處理），實測誤差從 0.5 格降到 0.02 格。

### 相機擺哪裡，比坐多近重要得多

量化修掉之後，剩下的誤差來自關鍵點本身的抖動（約 1px）。θ_CA 量的是耳朵相對肩膀往前伸多少，而「往前」在相機座標系裡落在哪個軸，取決於雙目模組架在受試者的哪個方位。**這兩個軸的精度差了一個數量級**：

```
深度軸      σ_Z = Z² × σ_d / (fx × B)     隨距離平方成長    620mm 下 16 mm
影像平面軸  σ_X = Z × σ_px / fx           隨距離線性成長    620mm 下 1.1 mm

比值 = Z / B                              620mm 配 60mm 基線 = 10 倍
```

其中 `σ_d = √2 × σ_px`，因為視差是左右兩次像素量測的差。方位角 α 的組合誤差是 `√((cos α·σ_Z)² + (sin α·σ_X)²)`，再乘 `√2` 因為耳朵與肩膀各有一份誤差。

實機參數（fx 568、基線 60mm、耳肩距 170mm）下的 θ_CA 單幀誤差：

| 距離 \ 方位角 | 0°（正面） | 30° | 45° | 60° | 75° | 90°（正側） |
|---|---|---|---|---|---|---|
| 500 mm | ±4.9° | ±4.3° | ±3.5° | ±2.5° | ±1.3° | ±0.4° |
| 620 mm | ±7.6° | ±6.6° | ±5.4° | ±3.8° | ±2.0° | ±0.5° |
| 800 mm | ±12.7° | ±11.0° | ±9.0° | ±6.4° | ±3.3° | ±0.7° |
| 1000 mm | ±19.8° | ±17.1° | ±14.0° | ±9.9° | ±5.2° | ±0.8° |
| 1500 mm | ±44.5° | ±38.5° | ±31.5° | ±22.3° | ±11.6° | ±1.3° |

這張表以蒙地卡羅驗證過（3000 次、1px 高斯雜訊、620mm、耳肩距 170mm），解析式與模擬相差在 20% 以內：

```
方位角      0°      30°     45°     60°     75°     90°
解析式    ±7.60°  ±6.59°  ±5.39°  ±3.83°  ±2.03°  ±0.52°
模擬      ±7.14°  ±6.98°  ±5.96°  ±4.29°  ±2.16°  ±0.63°
```

判定門檻是 `θ_CA > 10°`。`measure_posture` 會用實測的深度與方位角算出這個數字並印在報告裡，超過門檻本身時發出警告。「實測深度」取的是耳朵與肩膀的深度平均，也就是 θ_CA 實際用到的那兩個點；早期版本取深度範圍的最大最小中點，只要有一個關節配對錯誤跑到遠處這個中點就會被拉走——實測一個手腕誤配到 2.8m，584mm 的受試者會被當成 1684mm，誤差從 ±3.9° 變成 ±32.2°。

三個結論：

- **方位角比距離有效得多。** 620mm 從正面移到 60° 側面，誤差從 ±7.6° 降到 ±3.8°；同樣的改善靠坐近要從 620mm 移到 440mm。移到 75° 是 ±2.0°，坐近做不到。**注意改善是非線性的**：30° 只換到 13%，要到 60° 以上才明顯。
- **時間序列平均是必要的，不是加分項。** 平均 N 幀把雜訊降到 1/√N。
- **正面是最差的位置。** 這跟前作用側面相機量 θ_CA 的理由一致，只是前作是因為 2D 影像上正面看不出前傾，本專案是因為深度軸的精度差 Z/B 倍。

真正的限制不在幾何而在**遮擋**：方位角愈大，遠側的耳朵與肩膀愈容易被身體擋住，而雙肩正是解剖平面的來源、θ_sym 也需要兩邊都看到。實測在 29.8° 時遠側耳朵已經抓不到、遠側肩膀還正常。θ_sym 本身對方位角不敏感（模擬中 0° 到 90° 只從 ±0.17° 變成 ±0.44°，遠低於 5° 的門檻），所以決定上限的是能不能看到，不是準不準。

**基線**也是深度誤差的分母（反比），60mm 對 1m 外的目標偏短，但那要換硬體。

**提高相機解析度沒有幫助。** fx 隨解析度成正比變大，但熱圖量化換算回原始像素也同比例變大，兩者相消。真正有效的是提高網路輸入高度 `--input-height`。

### 個人基準：判定門檻不能套在原始角度上

頸部解剖結構因人而異。兩個人同樣坐得端正，量到的 θ_CA 可以差十幾度，直接比 `θ_CA > 10°` 會讓一個人永遠報警、另一個人永遠不報。前作的自適應歸零處理的就是這件事，實測讓誤報率從 18.7% 降到 4.2%。

```
python posture.py baseline --subject chenyue --out data/baselines/chenyue.json
```

請受試者坐正、目視前方、雙肩放鬆，保持不動 10 秒（實機約 6.4fps，約 60 幀）。取平均而非單幀的理由與執行期相同：拿單幀當基準等於把一個 ±10° 的偏移永久寫進之後所有判定。

```
受試者 chenyue（2026-09-23T22:36:12）
  θ_CA  基準 +12.13° ± 0.42°（單幀標準差 ±3.2°）
  θ_sym 基準  +0.01° ± 0.01°（單幀標準差 ±0.1°）
  取樣 60 幀 / 10.0 秒，略過 0 幀，距離 588 mm，方位角 1°
```

基準本身的品質會被檢查兩項。標準誤差超過 2° 時提醒重取——這個偏移會固定留在之後每一次判定裡，相對 10° 的門檻已經可觀。另一項是把實測散佈跟該距離與方位下的理論量測雜訊對照，實測明顯較大代表受試者在取基準的過程中動了，多出來的那部分不是雜訊。

之後帶著基準量測，印出的就是相對這個人端正坐姿的偏移量：

```
python posture.py live --baseline data/baselines/chenyue.json --log data/sessions/0923.csv
```

### 逐幀記錄

`--log` 把每一幀寫成 CSV，**包含被略過的幀**。既有的檔案不會被蓋掉，要覆蓋得加 `--overwrite`；基準檔同理，而且 `--out` 的預設檔名跟著 `--subject` 走，連續替幾個人取基準時不會互相覆蓋。輸出路徑在開相機之前就檢查——基準要請人坐著不動十秒，量測一次要二十分鐘，等做完才發現檔名撞到，浪費的是受試者的時間。

`--log` 把每一幀寫成 CSV，**包含被略過的幀**。略過率本身就是結果的一部分（偵測在什麼條件下會失效），而且只記成功的幀會讓事後分析看不出資料有多少缺口。原始角度與扣除基準後的角度都記，基準日後重取時才能重算。

每寫一列就 flush，因為這個程式是用 Ctrl-C 結束的，量測期間也可能因為相機斷線而中斷。缺失值寫成空字串而不是 0——0 是合法的角度值。

### 真實資料上的品質指標

標定板上的 RMS 只反映標定當下的品質。實際量測時 `measure_posture` 會另外回報三項。這三項都不會讓程式出錯，只會讓結果變錯：

- **校正後的垂直視差**：`stereoRectify` 的目的就是讓對極線水平，所以配對正確時左右的 y 應該幾乎相同。差超過 3px 代表標定不夠準，或左右眼配對到不同的人體部位。三角測量遇到兩條不相交的視線只會取最近點，給出的座標數量級一樣正常，所以這是唯一的線索。
- **深度範圍**：桌前坐姿應落在 200~3000mm。整體偏掉通常是 `--square-size-mm` 填錯造成尺度不對。
- **左右共同關鍵點數**：太少的話角度的參考價值有限。

三角測量的流程是先執行 `cv2.undistortPoints`（帶入 `R=R1, P=P1`）去除畸變並套用校正轉換，再進入 `cv2.triangulatePoints`。第一步不能省略：`P1/P2` 定義在校正後的座標系，把原始像素座標直接輸入不會出現錯誤訊息，算出來的3D點是錯的。`tests/test_triangulation.py` 用已知3D點反推驗證這條路徑。

### 角度定義與前作的差異

角度公式來自前一屆科展的前作《整合多視角姿態估測與幾何特徵量化之即時坐姿監測系統研究》。前作使用兩顆獨立單眼相機（正面0°＋側面45°），靠信心度加權融合2D結果，沒有真正的立體視覺；θ_CA 需要用 `/sin(45°)` 補償單一45°相機缺乏深度所造成的透視壓縮，正確性完全取決於相機確實架設在45°這個假設。

改用真正的雙目對極幾何之後，這個補償係數整個消失了，而非替換成更好的數值——它存在的唯一理由已經不復存在。現在直接取肩膀→耳朵的3D向量（程式裡是 `ear - shoulder`），投影到矢狀面後與垂直軸計算帶正負號的夾角，即為 θ_CA；θ_sym 同理，將左右肩連線投影到冠狀面後與左右軸計算夾角。

消掉這個係數需要兩步，只做第一步的話它只是換了個位置——第二步是把解剖平面也從相機手上拿回來，見上面的「相機架在哪裡都可以」。

角度採用 `atan2` 而非 `arccos(內積)`，因為後者只給出0~180°的大小、無法區分方向，而右肩比左肩高5°與左肩比右肩高5°是兩件不同的事。

θ_KA 在前作中完全沒有提及，公式尚未定案，`theta_ka()` 目前拋出 `NotImplementedError`——不以推測的公式填入，因為猜錯不會有人發現，但整份研究結論會建立在錯誤的定義上。個人校正基準（θ_offset）與 10°/5°/20px 這類判定門檻屬於執行期監測邏輯，不在這個模組的範圍內。

### 資料退化時會拋出例外，不會回傳0度

兩個關鍵點被計算到同一個3D位置、或向量完全垂直於量測平面時，角度函式會拋出 `ValueError`。

原因在於 0° 代表完全沒有偏移，也就是最理想的姿勢。偵測失效時若回傳0°，系統會把故障判讀成完美坐姿，永遠不會觸發警示。同理，三角測量遇到左右對應點幾乎重合的情況（視線平行、交點在無窮遠處）會回傳 NaN 而非 inf。

## 目錄

```
src/calibration/
  chessboard.py          棋盤格角點偵測
  charuco.py             ChArUco板角點偵測（容許只看到板子一部分）
  mono_calibration.py    單眼張氏標定，兩種板子共用同一套計算核心
  stereo_calibration.py  stereoCalibrate + stereoRectify，同上
  capture.py             連接相機互動式拍照（6種模式）＋解析度查詢
  cli.py                 讀取已拍影像、執行標定計算
src/pose/
  topology.py            關鍵點命名與順序（COCO 18點，含neck）
  keypoints.py           PersonKeypoints + RMSE計算
  preprocess.py          等比例縮放補邊＋座標還原（純numpy，不依賴torch）
  engine.py              PoseEngine介面 + Lightweight OpenPose實作（torch延遲import）
  benchmark.py           延遲量測、FP32/FP16精度比對
  cli.py                 執行基準測試與精度比對
src/geometry/
  triangulation.py       雙目3D三角測量核心
  keypoints3d.py         PersonKeypoints3D + 單人關鍵點三角測量
  angles.py              通用角度數學（與研究主題無關的純幾何運算）
  posture_angles.py      θ_CA/θ_sym實作，θ_KA尚未實作
tests/
  synthetic.py           合成測試影像（棋盤格標定）
  charuco_synthetic.py   合成測試影像（ChArUco標定）
  geometry_synthetic.py  合成雙目標定＋3D點投影（三角測量）
  pose_fakes.py          假引擎，讓pose模組的測試不需要GPU
```

## 待確認的假設

撰寫這些程式時，手邊沒有真實硬體或安裝好的模型可供核對。

| 假設 | 假設有誤時要修改的位置 |
|---|---|
| `torch2trt` 與 Lightweight OpenPose 的 API 呼叫方式 | `pose/engine.py` 內部，對照實際clone下來的原始碼 |
| 相機大致水平架設、無明顯翻滾角 | `geometry/posture_angles.py`；架設明顯傾斜會讓角度產生系統性偏移 |
| `--min-shared-corners` 6、`--rmse-threshold-px` 3.0、`_MIN_VECTOR_NORM` 1e-3 | 都是依經驗設定的起始值，取得實機數據後回頭調整 |

### 已在實機確認的項目

- **相機模式**（2026-09-21）：`probe` 量到 1280×480 @30.0fps、2560×720 @32.4fps、3840×1080 @15.5fps 三個原生並排模式。**已選定 2560×720**（每眼 1280×720）。這是只接雙目模組時量的，正面相機接回去後要重測頻寬。
- **權重可下載**（2026-09-21）：`checkpoint_iter_370000.pth` 已實際下載，84MB。
- **關鍵點順序**（2026-09-21）：上游 `Pose.kpt_names` 與 `pose/topology.py` 逐一相符，`neck` 在索引 1。
- **權重載入**（2026-09-21）：`load_state()` 沒有任何 `Not found pre-trained parameters` 警告，0.4.1 年代的 state_dict 在 JetPack 6 的 PyTorch 上完全對得上。

之前記錄的 trt_pose 拓樸核對（2026-09-18）已經作廢——那是針對 trt_pose 的 `human_pose.json`，排序與現在的模型完全不同。

### 仍未驗證的項目

`pose/engine.py` 的 API 假設包含：`fp16_mode` 參數、`TRTModule` 的存取方式、`extract_keypoints`/`group_keypoints` 的回傳格式（假設每列是 `(x, y, score, id)`）、`stages_output[-2]` 是熱圖而 `[-1]` 是 PAF、`pose_entries` 用 `-1` 表示未偵測。這些全部依上游原始碼撰寫，但尚未實機執行過。

（載入 state_dict 這一項已在 Jetson 上確認沒問題。）
