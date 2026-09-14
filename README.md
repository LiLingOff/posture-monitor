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

## 目錄

```
src/calibration/
  chessboard.py         角點偵測
  mono_calibration.py   正面相機張氏標定
  stereo_calibration.py stereoCalibrate + stereoRectify
  capture.py            接相機拍照
  cli.py                跑標定計算
tests/
  synthetic.py           合成測試影像用
```
