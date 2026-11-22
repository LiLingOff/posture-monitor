# 測試用字型

`NotoSansTC-Regular-subset.ttf`、`NotoSansTC-Bold-subset.ttf` 只給
`tests/test_text_with_bundled_font.py` 用，程式本身不會載入它們。

- 來源：Noto Sans TC（可變字型 2.04，`NotoSansTC-VF.ttf`），
  版權 © 2014-2021 Adobe (http://www.adobe.com/)，with Reserved Font Name 'Source'。
- 授權：SIL Open Font License, Version 1.1，<https://openfontlicense.org>
  （字型檔內的授權欄位也寫著同樣的聲明與網址）。
- 授權全文見同資料夾的 `OFL.txt`，取自
  <https://github.com/google/fonts/blob/main/ofl/notosanstc/OFL.txt>
  （2026-11-22 下載，4388 bytes，首行版權與字型檔內一致）。
- 修改：用 fontTools 把可變字型各取字重 400（Regular）與 700（Bold）的實例，
  再瘦身成只含 ASCII、θ ± ° 等符號，以及 `src/view`、判定理由與視窗測試裡出現過的
  中文字（約 800 字），去掉 hinting，其餘未改。字型名稱維持 "Noto Sans TC"，
  沒有使用保留字型名稱 "Source"。

要重做的話：字重實例用 `fontTools.varLib.instancer`，瘦身用 `fontTools.subset`，
字元清單由上述原始檔掃出來。
