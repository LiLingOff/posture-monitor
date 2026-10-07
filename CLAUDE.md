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
python -m pytest -q          # 2026-10-08 是 611 個
python -m pyflakes posture.py src/**/*.py tests/*.py
```

兩個都要過。測試全部用合成資料，不需要相機或 GPU。

## 幾個容易被「修正」回去的刻意設計

- **三個指標一律叫 θ_CA、θ_sym、肩部垂直位移**，面板、終端機、文件、報告
  全部一致。2026-10-08 之前面板刻意用白話（頭前傾、肩膀高低、肩膀下沉），
  那條規則已經取消，因為報告書用的是符號，而畫面截圖會進報告。
- **面板預設英文**（`--display-lang`，2026-10-08），而且**英文一律走 Hershey**，
  有沒有 TTF 字型都一樣。Hershey 的筆畫比 TTF 的 Latin 粗，站在旁邊瞄比較清楚。
  Pillow 只給中文用。選英文時不要印「找不到字型，改用英文」，那不是退路。
- **下標寫成 `θ_{CA}`**，`TextPainter` 會把它畫成小一級、往下挪的真下標。
  不要改用 Unicode 的小型大寫（ꜱʏᴍ）或下標字母（ₛ）,微軟正黑體整排缺字，
  實測會變成一排豆腐方塊。
- **非 ASCII 能不能畫是問出來的，不是猜的**（`hershey_can_draw`）。OpenCV 5 的
  `putText` 自己會畫 Unicode，4.x 的 Hershey 只有 ASCII。畫不出來時
  `TextPainter` 自己換成 ASCII 寫法（θ→theta、±→+-，表在 `_ASCII_INSTEAD`），
  排版那邊兩側一律寫符號本身。
- **`PostureBaseline` 後來加的欄位一定要有預設值**，`load()` 只檢查沒有預設值的
  欄位。現有的基準檔缺那幾欄，讀不進來的話沒有任何地方補得回來。
- **判定的第三軸（肩高）要 `watch_shoulder_drop()` 打開才生效**，這樣既有 CSV
  重播出來的結果與加這一項之前一樣。
- **`geometry/` 不碰相機、終端機與 argparse**，`view/` 才碰 cv2 與視窗。
  唯一的例外是 `recording.write_frame` 延遲匯入 `view.overlay`，為的是讓快照
  與即時畫面用同一套繪圖。
  「印給人看的文字」不算碰終端機：`live_report.py`、`session_report.py`、
  `cohort_report.py` 都在 geometry 裡，它們回傳字串，由呼叫端決定印在哪。
- **相機這個裝置在 `calibration/device.py`**，不在 `capture.py`。capture 是拍
  標定影像的互動工具，量測只需要 device 那一層，依賴方向是 capture → device。
- **平均值的誤差走批次平均法**（`geometry/uncertainty.py`），不是 `std/√N`。
  相鄰幀的自相關是 0.73，用 `std/√N` 實測低估 2.6 倍。
- **跨受試者的誤差分母是人數不是幀數**（`geometry/cohort.py`），與上一條相反，
  理由寫在該檔案的註解裡。

## 現在卡在哪

1. 姿勢清單已定（2026-10-06）：upright、head-forward、head-back、
   left-shoulder-up、right-shoulder-up，分兩趟跑（頭一趟、肩膀一趟）。
   θ_CA 改成前傾後仰都判。**不補**動到肩部垂直位移的條件，那一軸在報告
   裡只引用 2026-10-01 的 forward 資料，這是使用者的決定
2. 相機在量測中途掉線過四次，原因沒定。**不是 PhotonVision**：2026-10-06
   確認服務沒在跑還是掉了。一直沒拿到 dmesg，再掉的話第一件事是
   `dmesg | tail -40`。study 現在掉線後會重新找節點並重量那一段
3. LED／蜂鳴器**先不做**（使用者說那不是重點）。技術上沒有障礙，是排序的
   決定，不是永久排除。細節見 doc 09 第 8 項
