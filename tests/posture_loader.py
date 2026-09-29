"""把 posture.py 當成模組載進來測。

`posture.py` 放在 repo 根目錄而不是做成套件（理由見它自己的開頭說明），所以
測試要用檔案路徑載入。

**載進來的模組必須登記到 `sys.modules`。** `@dataclass` 在處理一個類別時會去
`sys.modules[cls.__module__]` 找模組來解析字串形式的註解，沒登記的話那裡是
None，裝飾器直接丟 `AttributeError: 'NoneType' object has no attribute
'__dict__'`，而錯誤訊息完全看不出真正的原因。

載入時 `sys.argv` 要先換掉。`posture.py` 的頂層沒有跑 argparse，但它會在
`if __name__ == "__main__"` 之外先改 `sys.path`，而測試程序自己的參數不該
被它看到。
"""
from __future__ import annotations

import importlib.util
import sys


def load_posture_module(name: str = "posture_cli"):
    """載入 posture.py。name 是登記到 sys.modules 的名字。"""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(name, "posture.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    saved = sys.argv
    sys.argv = ["posture.py"]
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # 載入失敗的半成品留在 sys.modules 裡的話，下一次呼叫會拿到它，
        # 而那個模組缺了一半的東西，錯誤會出現在離原因很遠的地方。
        sys.modules.pop(name, None)
        raise
    finally:
        sys.argv = saved
    return module
