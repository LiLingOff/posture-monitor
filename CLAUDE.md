# posture-monitor

第 67 屆科展作品《基於雙目對極幾何之即時三維坐姿監測系統》的程式。
前作是第 66 屆的《整合多視角姿態估測與幾何特徵量化之即時坐姿監測系統研究》。

## 文件在另一個 repo

`../posture-monitor-docs`（**private**，程式這邊是 public）。十一份 Markdown：

- **09-現況與待辦.md 是交接文件**，先讀它再動手
- 06-驗證方式與踩過的坑.md 記著每一個踩過的坑與當初的實測數字
- 07-與前作的差異.md 有前作原始碼的逐條佐證

程式與文件一起改的時候**兩邊都要 commit、都要 push**，它們是獨立的 git repo。
文件是 private，所以 public 的 README 不要放指向文件的連結。

## commit

- 訊息用繁體中文，寫清楚為什麼改，不只寫改了什麼
- **不要加 `Co-Authored-By`**，這是使用者明確要求的
- 只在使用者要求時才 commit 或 push

## 動手之前

```bash
python -m pytest -q          # 2026-10-01 是 538 個
python -m pyflakes posture.py src/**/*.py tests/*.py
```

兩個都要過。測試全部用合成資料，不需要相機或 GPU。

## 幾個容易被「修正」回去的刻意設計

- **面板用白話**（頭前傾、肩膀高低、肩膀下沉），文件與報告用 θ_CA、θ_sym、
  肩部垂直位移。刻意不一致，理由在 `src/view/overlay.py` 的註解裡。
- **`PostureBaseline` 後來加的欄位一定要有預設值**，`load()` 只檢查沒有預設值的
  欄位。現有的基準檔缺那幾欄，讀不進來的話沒有任何地方補得回來。
- **判定的第三軸（肩高）要 `watch_shoulder_drop()` 打開才生效**，這樣既有 CSV
  重播出來的結果與加這一項之前一樣。
- **`geometry/` 不碰相機、終端機與 argparse**，`view/` 才碰 cv2 與視窗。
  唯一的例外是 `recording.write_frame` 延遲匯入 `view.overlay`，為的是讓快照
  與即時畫面用同一套繪圖。
- **平均值的誤差走批次平均法**（`geometry/uncertainty.py`），不是 `std/√N`。
  相鄰幀的自相關是 0.73，用 `std/√N` 實測低估 2.6 倍。
- **跨受試者的誤差分母是人數不是幀數**（`geometry/cohort.py`），與上一條相反，
  理由寫在該檔案的註解裡。

## 現在卡在哪

1. 姿勢清單還沒定，擋著 doc 10 的受試者指導語與整場實驗設計
2. `--display`、`monitor`、肩部垂直位移都還沒在 Jetson 上跑過
3. LED／蜂鳴器**先不做**（使用者說那不是重點）。技術上沒有障礙，是排序的
   決定，不是永久排除。細節見 doc 09 第 8 項
