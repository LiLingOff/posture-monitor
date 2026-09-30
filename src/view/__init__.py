"""即時畫面：把一幀量測畫成人看得懂的東西。

`geometry` 那一層的模組開頭都寫著沒有相機、沒有終端機、沒有 argparse。視窗有
按鍵狀態、有無頭環境的退路，正是那一層排除的東西，所以另外開一包。

刻意不 re-export 任何東西。`overlay` 與 `window` 會拉進 cv2，`text` 會拉進
Pillow，而 `live_view` 只用 numpy；在這裡 re-export 的話，光是 import view
就把兩個重量級的套件都載進來。用的人直接 import 子模組。
"""
