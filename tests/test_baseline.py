"""個人基準（θ_offset）。

判定門檻不能直接套在原始角度上：頸部解剖結構因人而異，同樣坐得端正，
兩個人量到的 θ_CA 可以差十幾度。基準取錯的後果比執行期單幀算錯嚴重：
執行期的雜訊會被平均掉，基準的偏差會固定留在之後每一次判定裡。
"""
from dataclasses import asdict

import numpy as np
import pytest

from geometry.baseline import BaselineCollector, PostureBaseline
from geometry.pipeline import measure_posture
from geometry_synthetic import make_synthetic_stereo_calibration, project_point
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE_MM = 60.0


def _pose(theta_ca_deg: float, jitter_px: float = 0.0, seed: int = 0):
    ear_len, half = 120.0, 180.0
    dy = -ear_len * np.cos(np.radians(theta_ca_deg))
    dz = -ear_len * np.sin(np.radians(theta_ca_deg))
    right_sho = np.array([-half, 0.0, 600.0])
    left_sho = np.array([half, 0.0, 600.0])
    pose = {
        "right_shoulder": right_sho,
        "left_shoulder": left_sho,
        "right_ear": right_sho + np.array([40.0, dy, dz]),
        "left_ear": left_sho + np.array([-40.0, dy, dz]),
    }
    rng = np.random.default_rng(seed)
    left = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    right = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    conf = np.zeros(NUM_KEYPOINTS, np.float32)
    for name, p in pose.items():
        i = COCO18_KEYPOINT_NAMES.index(name)
        left[i] = project_point(K, np.eye(3), np.zeros((3, 1)), p) + rng.normal(0, jitter_px, 2)
        right[i] = project_point(
            K, np.eye(3), np.array([[-BASELINE_MM], [0.0], [0.0]]), p
        ) + rng.normal(0, jitter_px, 2)
        conf[i] = 0.9
    return PersonKeypoints(left, conf), PersonKeypoints(right, conf.copy())


def _calib():
    return make_synthetic_stereo_calibration(K, BASELINE_MM)


def _collect(theta_ca_deg: float, frames: int = 60, jitter_px: float = 0.5):
    collector = BaselineCollector()
    calib = _calib()
    for i in range(frames):
        left, right = _pose(theta_ca_deg, jitter_px, seed=i)
        collector.add(measure_posture(calib, left, right))
    return collector


def test_baseline_recovers_the_posture_it_was_shown():
    collector = _collect(12.0)
    baseline = collector.finish("A", 10.0)
    assert baseline.theta_ca_deg == pytest.approx(12.0, abs=2.0)
    assert baseline.frames >= 20


def test_two_people_sitting_equally_upright_get_different_baselines():
    """這就是為什麼門檻不能套在原始角度上。

    兩個人都坐得端正，但頸部結構不同，量到的 θ_CA 差 12 度。
    直接比 10° 的門檻，一個永遠報警、另一個永遠不報。
    """
    a = _collect(4.0).finish("A", 10.0)
    b = _collect(16.0).finish("B", 10.0)
    assert abs(a.theta_ca_deg - b.theta_ca_deg) > 10.0

    # 兩人都往前傾 8 度，扣掉各自的基準之後偏移量應該一致。
    # 容許值取基準本身的標準誤差，那是這個方法固有的殘差。
    leaned_a, _ = a.correct(4.0 + 8.0, 0.0)
    leaned_b, _ = b.correct(16.0 + 8.0, 0.0)
    tolerance = 3 * max(a.theta_ca_standard_error_deg, b.theta_ca_standard_error_deg)
    assert leaned_a == pytest.approx(8.0, abs=tolerance)
    assert leaned_b == pytest.approx(8.0, abs=tolerance)
    assert abs(leaned_a - leaned_b) < tolerance


def test_correct_passes_none_through():
    """角度算不出來時不能變成 -offset，那會被當成一個真實的偏移量。"""
    baseline = _collect(10.0).finish("A", 10.0)
    assert baseline.correct(None, None) == (None, None)


def test_refuses_to_produce_a_baseline_from_too_few_frames():
    """幀數不足時要明講，不能給出一個看起來正常的數字。"""
    collector = _collect(10.0, frames=5)
    with pytest.raises(ValueError, match="至少要"):
        collector.finish("A", 1.0)


def test_warns_when_the_subject_moved_during_the_baseline():
    """實測散佈遠大於理論量測雜訊，多的那部分只可能來自受試者本人。"""
    collector = BaselineCollector()
    calib = _calib()
    rng = np.random.default_rng(7)
    for i in range(60):
        # 每一幀的姿勢都在變，等於受試者一直在動
        left, right = _pose(10.0 + rng.normal(0, 12.0), jitter_px=0.5, seed=i)
        collector.add(measure_posture(calib, left, right))
    baseline = collector.finish("A", 10.0)
    assert any("動了" in w for w in collector.quality_warnings(baseline))


def test_a_still_subject_produces_no_movement_warning():
    collector = _collect(10.0)
    baseline = collector.finish("A", 10.0)
    assert not any("動了" in w for w in collector.quality_warnings(baseline))


def test_round_trips_through_a_file(tmp_path):
    baseline = _collect(10.0).finish("chenyue", 10.0)
    path = tmp_path / "b.json"
    baseline.save(path)
    assert PostureBaseline.load(path) == baseline


def test_rejects_a_baseline_file_missing_fields(tmp_path):
    """欄位少了就是舊版檔案，沿用的話，判定基準會錯而且看不出來。"""
    path = tmp_path / "old.json"
    path.write_text('{"subject": "A", "theta_ca_deg": 5.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少欄位"):
        PostureBaseline.load(path)


def test_describe_names_the_conditions_it_was_taken_under():
    text = _collect(10.0).finish("chenyue", 10.0).describe()
    for expected in ("chenyue", "θ_CA", "距離", "方位角"):
        assert expected in text


def test_refuses_to_overwrite_an_existing_baseline(tmp_path):
    """連續替幾位受試者取基準時很容易忘記換檔名。

    蓋掉的話前一位的判定基準就沒了，而且不會有任何跡象：
    檔案還在，內容卻換成了另一個人的。
    """
    path = tmp_path / "b.json"
    first = _collect(5.0).finish("A", 10.0)
    first.save(path)

    second = _collect(18.0).finish("B", 10.0)
    with pytest.raises(FileExistsError, match="overwrite"):
        second.save(path)
    assert PostureBaseline.load(path).subject == "A"

    second.save(path, overwrite=True)
    assert PostureBaseline.load(path).subject == "B"


def _reject_all(reason_depth_mm: float, frames: int = 12):
    """生出一批全部因為同一個原因被擋掉的量測。"""
    collector = BaselineCollector()
    calib = _calib()
    for i in range(frames):
        left, right = _pose(12.0, jitter_px=0.5, seed=i)
        moved = right.points.copy()
        moved[:, 0] -= 600.0   # 右眼整組往左移，視差變到不可能的大小
        collector.add(measure_posture(calib, left, PersonKeypoints(moved, right.confidences)))
    return collector


def test_failure_names_the_reason_the_frames_were_actually_dropped():
    """2026-09-24 實機：12 幀全部因為深度 54mm 被擋掉。

    當時的訊息說「確認受試者在畫面內、光線足夠」，但人明明偵測到了，
    是深度不合理被擋的。通用的提醒會把人指向錯的方向。
    """
    collector = _reject_all(54.0)
    assert collector.count == 0 and collector.rejected == 12

    with pytest.raises(ValueError) as exc:
        collector.finish("test", 5.0)
    message = str(exc.value)
    assert "12/12" in message, "要說出幾乎全部是同一個原因"
    assert "深度" in message
    assert "54" in message or "mm" in message, "顯示的是原句，數字要留著"
    assert "Nmm" not in message, "佔位符只該當分組的鍵，不該出現在訊息裡"
    assert "--all-keypoints" in message, "要給出下一步能查的指令"
    assert "光線" not in message, "人有偵測到，不該再叫人去查光線"


def test_groups_rejections_that_differ_only_in_their_numbers():
    """「深度 54mm」和「深度 61mm」是同一個問題，不該被算成兩種。"""
    from geometry.baseline import group_rejection_reason

    assert group_rejection_reason("深度 54mm 落在合理範圍外（200~3000mm）") == \
        group_rejection_reason("深度 -61mm 落在合理範圍外（200~3000mm）")


def test_says_so_when_nothing_was_captured_at_all():
    """一幀都沒進來是另一回事，那時才該查畫面與光線。"""
    collector = BaselineCollector()
    with pytest.raises(ValueError, match="光線"):
        collector.finish("test", 5.0)


def test_advice_points_at_distance_when_distance_is_the_problem():
    """2026-09-24：在 1796mm、方位角 50° 取到的基準，誤差 ±3.8°。

    當時的建議是「把雙目模組往側面移（正面是精度最差的位置）」，
    但模組已經在 50° 了，而且同一個距離推到 75° 只從 ±41° 降到 ±17°，
    坐到 700mm 則是 ±6.3°。建議指錯了方向。
    """
    from geometry.pipeline import precision_advice

    advice = precision_advice(1796.0, 50.0)
    assert "坐到 700mm" in advice
    assert "往側面移" not in advice
    assert "1/6" in advice, "要給出具體的改善倍數"


def test_advice_points_at_azimuth_once_the_distance_is_fine():
    from geometry.pipeline import precision_advice

    advice = precision_advice(650.0, 15.0)
    assert "方位角" in advice and "往側面移" in advice
    assert "坐到" not in advice


def test_advice_admits_when_there_is_nothing_left_to_move():
    from geometry.pipeline import precision_advice

    advice = precision_advice(650.0, 80.0)
    assert "極限" in advice and "平均視窗" in advice


def test_advice_says_so_when_the_azimuth_was_not_measured():
    from geometry.pipeline import precision_advice

    assert "方位角量不到" in precision_advice(650.0, None)
    assert "距離量不到" in precision_advice(None, 50.0)


def test_baseline_warning_carries_the_same_advice():
    """取樣夠長之後，剩下的誤差才歸因於幾何條件，這時該與逐幀的警告說同一套話。"""
    collector = _collect(12.0, frames=200)
    baseline = collector.finish("A", 31.0)
    far = PostureBaseline(**{**asdict(baseline),
                             "distance_mm": 1796.0, "azimuth_deg": 50.0,
                             "theta_ca_standard_error_deg": 3.8,
                             "theta_ca_std_deg": 23.5})
    warning = next(w for w in collector.quality_warnings(far) if "基準誤差" in w)
    assert "坐到 700mm" in warning
    assert "往側面移" not in warning


def _poor_baseline(**overrides):
    base = _collect(12.0, frames=200).finish("A", 31.0)
    return PostureBaseline(**{**asdict(base), **overrides})


def _with(**overrides):
    base = _collect(12.0, frames=200).finish("A", 31.0)
    return PostureBaseline(**{**asdict(base), **overrides})


def test_a_short_baseline_is_told_how_long_to_sample_next_time():
    """取樣不足時，調距離與方位角沒有用，唯一有效的是拉長取樣。

    秒數要算出來。誤差隨獨立樣本數的平方根下降，所以 10 秒拿到 ±3.1°
    要壓到 ±2° 需要 10×(3.1/2)² 約 24 秒。
    """
    from geometry.baseline import baseline_quality_warnings

    short = _with(frames=60, duration_s=10.0, distance_mm=650.0, azimuth_deg=32.0,
                  theta_ca_standard_error_deg=3.1)
    warning = next(w for w in baseline_quality_warnings(short) if "基準誤差" in w)
    assert "24 秒" in warning


def test_the_advice_never_tells_you_to_lengthen_to_what_you_already_did():
    """2026-09-29 實機：142 幀 / 30 秒的基準收到「拉長到 30 秒以上」。

    當時的判斷用幀數門檻、建議卻寫死成秒數，於是在 5fps 的機器上永遠自相矛盾。
    """
    from geometry.baseline import baseline_quality_warnings

    real = _with(frames=142, duration_s=30.1, distance_mm=650.0, azimuth_deg=32.0,
                 theta_ca_standard_error_deg=2.97, theta_ca_std_deg=11.1)
    warning = next(w for w in baseline_quality_warnings(real) if "基準誤差" in w)
    assert "66 秒" in warning
    assert "30 秒以上" not in warning
    # 速率要一起講，不然使用者無從判斷那個秒數合不合理
    assert "4.7 幀" in warning


def test_when_lengthening_cannot_help_it_says_so_and_points_elsewhere():
    """誤差大到要取樣好幾分鐘時，拉長取樣補不回來，因為姿勢本身會漂。"""
    from geometry.baseline import baseline_quality_warnings

    hopeless = _with(frames=142, duration_s=30.0, distance_mm=650.0,
                     azimuth_deg=32.0, theta_ca_standard_error_deg=5.0)
    warning = next(w for w in baseline_quality_warnings(hopeless) if "基準誤差" in w)
    assert "補不回來" in warning
    assert "往側面移" in warning


def test_sitting_too_far_outranks_sampling_longer():
    """坐太遠的時候建議取樣兩分鐘是在浪費受試者的時間，椅子往前拉一次就解決。"""
    from geometry.baseline import baseline_quality_warnings

    far = _with(frames=200, duration_s=31.0, distance_mm=1796.0, azimuth_deg=50.0,
                theta_ca_standard_error_deg=3.8)
    warning = next(w for w in baseline_quality_warnings(far) if "基準誤差" in w)
    assert "坐到 700mm" in warning
    assert "秒" not in warning.split("這個偏移會留在之後每一次判定裡。")[1]


def test_a_poor_baseline_still_warns_when_it_is_loaded_back(tmp_path):
    """取基準當下看過一次就過去了，但這個偏移會進到之後每一次判定。

    存檔之後不再提醒的話，一份明知不好的基準會默默污染整場量測。
    """
    from geometry.baseline import baseline_quality_warnings

    poor = _poor_baseline(distance_mm=1384.0, azimuth_deg=28.0,
                          theta_ca_standard_error_deg=5.7, theta_ca_std_deg=36.2)
    path = tmp_path / "poor.json"
    poor.save(path)

    reloaded = PostureBaseline.load(path)
    warnings = baseline_quality_warnings(reloaded)
    assert any("基準誤差" in w for w in warnings)
    assert any("坐到 700mm" in w for w in warnings)


def test_a_good_baseline_loads_without_complaint():
    from geometry.baseline import baseline_quality_warnings

    good = _poor_baseline(distance_mm=680.0, azimuth_deg=40.0,
                          theta_ca_standard_error_deg=0.9, theta_ca_std_deg=5.7)
    assert baseline_quality_warnings(good, expected_single_frame_error_deg=6.0) == []


def test_movement_check_needs_the_expected_noise_to_compare_against():
    """沒有理論值時不能判斷散佈大是雜訊還是受試者在動，那就不要猜。"""
    from geometry.baseline import baseline_quality_warnings

    wobbly = _poor_baseline(distance_mm=680.0, azimuth_deg=40.0,
                            theta_ca_standard_error_deg=0.9, theta_ca_std_deg=30.0)
    assert baseline_quality_warnings(wobbly, None) == []
    assert any("動了" in w for w in baseline_quality_warnings(wobbly, 6.0))


def test_an_elevated_spread_is_addressed_before_sampling_longer():
    """2026-09-29 實機：單幀散佈 ±11.1° 對上該有的 ±6.9°。

    誤差與散佈成正比、與時間的平方根成反比，所以把散佈壓一半等於取樣時間
    砍到四分之一。叫人坐定再開始比叫人多坐 36 秒有效得多。
    """
    from geometry.baseline import baseline_quality_warnings

    jittery = _with(frames=142, duration_s=30.1, distance_mm=650.0, azimuth_deg=32.0,
                    theta_ca_standard_error_deg=2.97, theta_ca_std_deg=11.1)
    warning = next(w for w in baseline_quality_warnings(jittery, 6.9)
                   if "基準誤差" in w)
    assert "先讓受試者坐定" in warning
    assert "66 秒" not in warning


def test_a_normal_spread_still_gets_the_sampling_length_advice():
    """散佈正常時誤差就是取樣長度造成的，這時候拉長取樣才是對的建議。"""
    from geometry.baseline import baseline_quality_warnings

    steady = _with(frames=142, duration_s=30.1, distance_mm=650.0, azimuth_deg=32.0,
                   theta_ca_standard_error_deg=2.97, theta_ca_std_deg=7.0)
    warning = next(w for w in baseline_quality_warnings(steady, 6.9) if "基準誤差" in w)
    assert "66 秒" in warning
    assert "先讓受試者坐定" not in warning


def test_a_high_rejection_rate_is_warned_about_even_when_the_error_is_small():
    """2026-09-29 實機：109 幀收下、41 幀略過（27%），誤差 ±1.34° 所以零警告。

    誤差小不代表這份基準可信。留下來的幀可能全都偏向同一邊，而被擋掉的
    那些正好是另一種姿勢。
    """
    collector = _collect(12.0, frames=109)
    for _ in range(41):
        collector._reasons["right_shoulder 的垂直視差 N px，左右配對錯了"] += 1
        collector._reason_examples.setdefault(
            "right_shoulder 的垂直視差 N px，左右配對錯了",
            "right_shoulder 的垂直視差 8.1px，左右配對錯了")
        collector.rejected += 1
    baseline = _with(frames=109, theta_ca_standard_error_deg=1.34,
                     theta_ca_std_deg=5.5, distance_mm=597.0, azimuth_deg=25.0)
    warnings = collector.quality_warnings(baseline)
    stated = next(w for w in warnings if "略過" in w)
    assert "27%" in stated
    # 最常見的原因與它的建議要一起講，否則使用者只知道有問題不知道動什麼
    assert "right_shoulder" in stated
    assert "遠側肩膀" in stated


def test_a_normal_rejection_rate_says_nothing():
    """乾淨的量測是 2~5%，那個範圍不該被打擾。"""
    collector = _collect(12.0, frames=146)
    for _ in range(7):
        collector.rejected += 1
    baseline = _with(frames=146, theta_ca_standard_error_deg=0.63,
                     theta_ca_std_deg=4.2, distance_mm=579.0, azimuth_deg=20.0)
    assert not any("略過" in w for w in collector.quality_warnings(baseline))
