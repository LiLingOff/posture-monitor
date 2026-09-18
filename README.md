# posture-monitor

高中科展「基於雙目對極幾何之即時三維坐姿監測系統」的程式部分。一顆正面相機加一組雙目模組拍坐姿，標定相機後把雙目看到的關鍵點還原成真實3D座標，再算出頸椎前傾角與肩膀水平角。

| 模組 | 狀態 | 內容 |
|---|---|---|
| `src/calibration` | 可用 | 棋盤格／ChArUco，單眼＋雙目，合成資料驗證過準確度 |
| `src/pose` | **推論未實機驗證** | trt_pose + TensorRT。安裝與關鍵點拓樸已在Jetson確認，推論本身還沒跑過 |
| `src/geometry` | 可用 | 三角測量、θ_CA／θ_sym；θ_KA 公式未定 |
| 判定與回饋 | 未開始 | 個人基準校正、閾值判定、LED／蜂鳴器 |

```
標定 ──► 拍攝 ──► 關鍵點偵測 ──► 三角測量 ──► 角度 ──► (未做) 判定回饋
 .npz      BGR       2D像素         3D座標(mm)    度
```

## 環境

| | 開發機 | 部署機 |
|---|---|---|
| 平台 | Windows，無NVIDIA GPU | Jetson Orin Nano |
| 系統 | — | JetPack 6、Ubuntu 22.04、Python 3.10、CUDA 12 |
| 能跑什麼 | 標定、三角測量、全部測試 | 上述全部＋關鍵點偵測推論 |

這個分裂直接影響程式怎麼寫：`src/pose/engine.py` 裡所有 `import torch` / `import trt_pose` 都寫在方法內部而不是檔案頂端。開發機裝不了這些套件，但又必須能 `import pose` 跑測試，所以把 import 延遲到真正要用時才執行——沒裝 torch 的機器上 `import pose.engine` 依然成功，只有真的建立 `TrtPoseEngine` 實例才會需要。`tests/test_pose_engine_import.py` 鎖住這個性質。看到程式裡有地方寫得繞，原因通常是這個。

## 安裝與測試

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

85個測試，全部用合成資料，不需要相機或GPU。

標定的準確度是這樣驗證的：已知一組相機內參與基線長度，用單應變換把正面棋盤格圖「合成」成該相機在特定姿態下拍到的樣子，餵進標定演算法，檢查還原出來的參數跟真值差多少。這在數學上是嚴格等價而非近似——標定板是平面，`Z=0` 讓投影方程式退化成單應變換。三角測量的測試則改用 `cv2.projectPoints` 直接投影已知3D點（單應變換的前提是共平面，而三角測量要驗證的正是非共平面的點），合成資料下還原誤差 0.000mm。

判準是「還原出來的焦距誤差 <5%、基線長度誤差 <10%」，不是「有沒有丟例外」。

## 資料型別

| 型別 | 欄位 | 形狀 | 單位 |
|---|---|---|---|
| `StereoCalibrationResult` | `camera_matrix_left/right`、`dist_coeffs_*` | (3,3)、(5,) | — |
| | `R`、`T` | (3,3)、(3,1) | T 為 mm |
| | `R1/R2`、`P1/P2`、`Q` | (3,3)、(3,4)、(4,4) | 三角測量吃 `P1/P2` |
| `PersonKeypoints` | `points`、`confidences` | (18,2)、(18,) | 像素 |
| `PersonKeypoints3D` | `points` | (18,3) | **mm** |

三個貫穿全系統的慣例：

- **未偵測一律用 NaN**，不是0也不是-1——0是合法座標，用它代表「沒有」會讓壞資料混進計算而不被發現
- **長度單位一路都是 mm**，源頭是標定時的 `--square-size-mm`。這個數字填錯，整個系統的長度尺度就跟著錯
- **精度沒達標只印警告，資料明顯壞掉才拋例外**。前者是「結果可用但品質待評估」，後者是「這個結果沒有意義」

## 標定：拍攝

| 情境 | 指令 |
|---|---|
| 正面相機 | `python -m src.calibration.capture mono --camera 0 --target-count 40` |
| 雙目＝兩顆獨立USB相機 | `python -m src.calibration.capture stereo --left-camera 1 --right-camera 2` |
| 雙目＝單一USB裝置合併輸出 | `python -m src.calibration.capture stereo --single-device --left-camera 1` |
| 產生可列印的ChArUco板 | `python -m src.calibration.capture board --out data/charuco_board.png --squares-x 10 --squares-y 8 --square-size-mm 25 --marker-size-mm 18` |

| 旗標 | 用途 |
|---|---|
| `--charuco` | 改用ChArUco板（可與其餘旗標並用） |
| `--single-device` | 雙目是單一裝置、左右眼合併在一張畫面；`--left-camera` 填該裝置index |
| `--vertical-split` | 合併畫面是上下拼接而非左右 |
| `--swap-lr` | 左右眼顛倒 |
| `--legacy-pattern` | 板子用OpenCV 4.6以前的ArUco排列（現成板子常見） |
| `--min-shared-corners` | 左右需有幾個共同角點才准存檔，預設6 |

操作：空白鍵存檔、ESC結束。視窗即時畫出偵測到的角點與左右共同角點數，拍完會停在最後一幀等按鍵。資料夾已有足夠張數時會直接跳過，要重拍先清空資料夾。

拍攝要領：板子要涵蓋畫面中心與四個角落，並變換距離與傾斜角度；都拍同一個位置的話畸變係數解不準。列印時選「實際尺寸」不要自動縮放，印完拿尺量一格實際幾mm，用量到的數字當 `--square-size-mm`。

現成板子照包裝規格填即可，例如 AndyMark 那張是 `--squares-x 15 --squares-y 11 --square-size-mm 15 --marker-size-mm 11 --dictionary DICT_4X4_250`。

| 症狀 | 處理 |
|---|---|
| 看得到板子但沒有標記疊上去 | 加 `--legacy-pattern` |
| **`stereo left`／`stereo right` 顯示不同場景** | 拿到的不是左右並排的雙目畫面，而是單眼畫面被切一半。用 `--width/--height` 指定並排模式的解析度（見下方） |
| 共同角點數一直不夠 | 先確認上一項；再把板子往兩鏡頭視野重疊區移，或調低 `--min-shared-corners` |
| `can't open camera by index` | Linux上先 `v4l2-ctl --list-devices` 查index（沒裝先 `sudo apt install v4l-utils`）；並確認 `groups` 有 `video`，沒有的話 `sudo usermod -aG video $USER` 後重新登入 |

### 雙目模組一定要指定解析度

OpenCV 在 V4L2 下不指定就用驅動的預設模式，通常是 640×480。雙目模組的「左右眼並排」輸出往往只存在於特定的寬解析度模式（2560×720 之類），預設模式給的是單眼或裁切畫面——切一半會得到兩塊**不重疊**的裁切，看起來像兩個不同場景，標定永遠湊不到共同角點。

先查相機支援哪些模式：

```
v4l2-ctl -d /dev/video1 --list-formats-ext
```

挑寬度是單眼兩倍的那個模式（例如單眼 1280×720 就找 2560×720），拍攝時帶上：

```
python -m src.calibration.capture stereo --single-device --left-camera 1 --width 2560 --height 720 --charuco ...
```

程式開相機後會印出實際拿到的解析度，要求不到時會明講。單一裝置模式下還會檢查第一幀的長寬比——不像並排輸出就出聲警告，不會讓它安靜地錯下去。

### 為什麼非用 ChArUco 不可

一般棋盤格標定要求整塊板子完整入鏡——`findChessboardCorners` 找的是完整的 N×M 網格，缺一角就整張作廢，因為棋盤格的角點長得都一樣，沒有完整網格就無法判斷哪個角點是哪個。雙目更嚴苛：同一時刻左右兩張都要看到完整板子。

這組雙目模組兩顆鏡頭的視野重疊區很小，不管怎麼調距離都達不到這個條件。ChArUco 板每個白格嵌一個獨一無二的 ArUco 標記，角點因此有獨立ID，左右只要有足夠多共同ID的角點就能建立對應，不需要整塊板子入鏡。

這不是理論上的權宜之計：`tests/test_charuco_calibration.py` 裡的測試，合成的兩台相機在整個過程中都沒有任何一幀看到完整的63個角點，標定依然把基線長度與相對旋轉精確還原出來。

## 標定：計算參數

| 目標 | 指令 |
|---|---|
| 單眼 | `python -m src.calibration.cli mono --images data/calibration_images/front` |
| 雙目 | `python -m src.calibration.cli stereo --left-images data/calibration_images/stereo_left --right-images data/calibration_images/stereo_right` |
| ChArUco | 上述加 `--charuco` 與拍攝時**完全相同**的板子參數 |

輸出 `.npz`：內參、畸變係數；雙目另含 `R/T/E/F/R1/R2/P1/P2/Q`。目標重投影誤差單眼 <0.3px、雙目 <0.5px。

板子參數跟拍攝時不一致的話，角點ID會對到錯的物理座標——標定結果是錯的，但通常不會報錯，這種錯誤很難發現。

超標只印警告、照樣存檔。不是因為無所謂，是因為機械式擋著不給用反而會讓人略過警告字面、改用更寬鬆的門檻重跑，不如把數字攤開讓人自己判斷。

### 重投影誤差踩過的坑

印出來的 RMS 跟 `cv2.calibrateCamera` 自己回傳的數字一致，`tests/test_reprojection_error.py` 鎖住這件事。

會特別測，是因為這裡錯過一次：原本沿用 OpenCV 官方教學的寫法 `cv2.norm(...) / len(projected)`，但 `cv2.norm` 給的是「所有點誤差平方和再開根號」，要得到 RMS 應該除以 `sqrt(N)` 而不是 `N`。除以 `N` 會把誤差低估 `sqrt(N)` 倍——35個角點就是5.9倍，真實 0.13px 印成 0.02px。

後果不是「數字有點小」，而是整個品質門檻形同虛設：真實誤差 1.5px 的爛標定會顯示成 0.25px，順利通過 <0.3px 的檢查。這種錯沒有任何外顯症狀，不刻意跟 OpenCV 自己的回傳值對照就永遠不會發現。

## 關鍵點偵測 + TensorRT

原研究文件寫的是 OpenPose Body_25，但 OpenPose（Caffe架構）約2021年後就沒再更新、官方最高支援到 CUDA 10，在 JetPack 6 的 CUDA 12 環境裝不太起來。改用 [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)——NVIDIA 自己為 Jetson 做的 TensorRT pose 函式庫。方法學不變（部署基準測試、FP16加速、精度比對、必要時退回FP32），換的只是實作用的模型。

**這個模組沒有在真實硬體上跑過。** 能寫能測的只有不碰GPU的部分：關鍵點資料結構、RMSE計算、延遲統計、CLI參數解析，用假引擎（`tests/pose_fakes.py`）驗證邏輯。真正的推論與 TensorRT 轉換要在 Jetson 上才能執行，程式寫好了但沒人驗證過會不會動。

Jetson 上手動安裝：

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt && python3 setup.py install --user && cd ..

git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose && python3 setup.py install --user && cd ..

cp trt_pose/tasks/human_pose/human_pose.json <repo>/data/pose_models/
```

trt_pose 的 `setup.py` 沒把相依套件宣告完整，還要補 `pip3 install --user tqdm pillow`（`pycocotools` 只有訓練/評估才用得到，在 ARM 上要編譯很久，不用裝）。

PyTorch 要裝 NVIDIA 官方的 Jetson 專用 wheel，不是 `pip install torch`（到 https://forums.developer.nvidia.com/t/pytorch-for-jetson 查 JetPack 6 對應版本）。模型權重 `.pth` 從 trt_pose 的 model zoo 手動下載，放進 `data/pose_models/`。

| 用途 | 指令 |
|---|---|
| 驗證安裝 | `python3 -c "import torch, torch2trt, trt_pose.models, trt_pose.coco; print('ok')"` |
| 核對關鍵點定義 | `python3 -c "import json; print(json.load(open('data/pose_models/human_pose.json'))['keypoints'])"` |
| 延遲基準（FP32） | `python -m src.pose.cli benchmark --precision fp32 --front-camera 1 --stereo-camera 3 --single-device` |
| 延遲基準（FP16） | 同上改 `--precision fp16`；第一次會花幾分鐘建TensorRT engine並存快取 |
| 精度比對 | `python -m src.pose.cli compare-precision --camera 1 --samples 30 --rmse-threshold-px 3.0` |

第一次在 Jetson 上跑起來時，最該先確認的是 `infer()` 回傳的關鍵點座標數量級——要是像素等級的數字（幾百），不是 0~1 的小數。trt_pose 內部吐出的 peaks 是正規化座標，程式會乘回原始畫面寬高換算成像素；漏掉這步的話三角測量會整個歪掉，而且完全不會報錯。

## 三角測量與坐姿角度

`src/geometry/` 把雙目標定輸出（`P1/P2` 等）跟 pose 模組輸出（`PersonKeypoints`）接起來。

```python
from geometry import triangulate_person_keypoints
from geometry.posture_angles import theta_ca, theta_sym

keypoints_3d = triangulate_person_keypoints(stereo_calib, left_keypoints, right_keypoints)
theta_ca(keypoints_3d)   # 頸椎前傾角（度，帶正負號），side="right"/"left"
theta_sym(keypoints_3d)  # 肩膀水平角（度，帶正負號）
```

三角測量的流程是先 `cv2.undistortPoints`（帶 `R=R1, P=P1`）去畸變並套用校正轉換，再進 `cv2.triangulatePoints`。第一步不能省：`P1/P2` 定義在校正後的座標系，把原始像素座標直接丟進去不會報錯，只會安靜地算出錯的3D點。`tests/test_triangulation.py` 用已知3D點反推驗證這條路徑。

### 角度定義與前作的差異

角度公式來自前一屆科展的前作《整合多視角姿態估測與幾何特徵量化之即時坐姿監測系統研究》。前作用兩顆獨立單眼相機（正面0°＋側面45°）靠信心度加權融合2D結果，沒有真正的立體視覺；θ_CA 得用 `/sin(45°)` 補償單一45°相機缺乏深度造成的透視壓縮，正確性完全綁在「相機真的架在45°」這個假設上。

改用真正的雙目對極幾何之後，這個補償係數不是換成更好的值，而是整個消失了——它存在的唯一理由已經不存在。現在直接取耳朵→肩膀的3D向量投影到矢狀面、與垂直軸算帶號夾角就是 θ_CA；θ_sym 同理，左右肩連線投影到冠狀面與水平軸算夾角。

角度用 `atan2` 而非 `arccos(內積)`，因為後者只給0~180°的大小、分不出方向，而「右肩比左肩高5°」跟「左肩比右肩高5°」是不同的事。

θ_KA 前作完全沒提到，公式未定案，`theta_ka()` 目前拋 `NotImplementedError`——不用猜的公式硬填，猜錯了不會有人發現，但整份研究結論會建立在錯的定義上。個人校正基準（θ_offset）與 10°/5°/20px 這類判定門檻屬於執行期監測邏輯，不在這個模組。

### 資料退化時會報錯，不會回傳0度

兩個關鍵點被算到同一個3D位置、或向量完全垂直於量測平面時，角度函式拋 `ValueError`。

因為 0° 的意思是「完全沒有偏移」，也就是最理想的姿勢。偵測壞掉時回傳0°，系統會把故障判讀成完美坐姿、永遠不觸發警示。同理，三角測量遇到左右對應點幾乎重合（視線平行、交點在無窮遠）會回傳 NaN 而不是 inf。

## 目錄

```
src/calibration/
  chessboard.py          棋盤格角點偵測
  charuco.py             ChArUco板角點偵測（容許只看到板子一部分）
  mono_calibration.py    單眼張氏標定，兩種板子共用同一套計算核心
  stereo_calibration.py  stereoCalibrate + stereoRectify，同上
  capture.py             接相機互動式拍照（6種模式）
  cli.py                 讀取已拍影像、跑標定計算
src/pose/
  topology.py            關鍵點命名（COCO 18點，含neck）
  keypoints.py           PersonKeypoints + RMSE計算
  preprocess.py          畫面前處理（純numpy，不碰torch）
  engine.py              PoseEngine介面 + trt_pose實作（torch延遲import）
  benchmark.py           延遲量測、FP32/FP16精度比對
  cli.py                 跑基準測試/精度比對
src/geometry/
  triangulation.py       雙目3D三角測量核心
  keypoints3d.py         PersonKeypoints3D + 單人關鍵點三角測量
  angles.py              通用角度數學（跟研究無關的純幾何）
  posture_angles.py      θ_CA/θ_sym實作，θ_KA是stub
tests/
  synthetic.py           合成測試影像（棋盤格標定）
  charuco_synthetic.py   合成測試影像（ChArUco標定）
  geometry_synthetic.py  合成雙目標定＋3D點投影（三角測量）
  pose_fakes.py          假引擎，測pose模組不需要GPU
```

## 待確認的假設

寫這些的時候手上沒有真實硬體或裝好的 trt_pose 可以核對。

| 假設 | 錯了要改哪裡 |
|---|---|
| `torch2trt`/`trt_pose` 的 API 呼叫方式 | `pose/engine.py` 內部，對照實際clone下來的原始碼 |
| 相機大致水平、無明顯翻滾角 | `geometry/posture_angles.py`；架設歪得明顯會讓角度系統性偏移 |
| `--min-shared-corners` 6、`--rmse-threshold-px` 3.0、`_MIN_VECTOR_NORM` 1e-3 | 都是憑經驗抓的起始值，跑過實機數據後回頭調 |

### 已在實機確認的

- **關鍵點拓樸**（2026-09-18，Jetson）：trt_pose 的 `human_pose.json` 確實是 COCO 17 點加第 18 個 `neck`，名稱與順序跟 `pose/topology.py` 完全一致，不需要修改。`neck` 是模型真的偵測出來的點，不是左右肩推算的中點——這是當初選 trt_pose 的理由之一。`tests/test_pose_topology.py` 已把這份清單寫死鎖住。

### 仍未驗證的

`pose/engine.py` 的 API 假設包含：`fp16_mode` 參數、`TRTModule` 存讀方式、`ParseObjects` 用法、`peaks` 是正規化座標要乘回畫面尺寸、信心度從 `cmap` 取值。這些全部照公開資料寫的，沒有實機驗證過。
