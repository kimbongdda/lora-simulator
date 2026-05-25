"""기준선 및 보상 변형 비교에서 쓰는 공통 시각화 스타일."""

from __future__ import annotations

import matplotlib


# ── 폰트 설정 (한글 깨짐 방지) ──────────────────────────────────────────────
def configure_fonts() -> None:
    """Windows/Linux/Mac 에서 한글이 깨지지 않도록 폰트를 설정한다."""
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── 색상 — 12개 모두 고유 색 ───────────────────────────────────────────────
# matplotlib tab10 10색 + dark-green + goldenrod 추가
# 색상표: https://matplotlib.org/stable/gallery/color/named_colors.html
BASELINE_COLORS: dict[str, str] = {
    "pure_aloha":               "#555555",   # 진회색
    "adr_like":                 "#1f77b4",   # tab10 파랑
    "retry_aware":              "#2ca02c",   # tab10 초록
    "lora_mab":                 "#ff7f0e",   # tab10 주황
    "lora_mab_neg":             "#d62728",   # tab10 빨강
    "lora_mab_idle":            "#9467bd",   # tab10 보라
    "lora_mab_idle_neg":        "#8c564b",   # tab10 갈색
    "thompson_mab":             "#e377c2",   # tab10 분홍
    "thompson_mab_idle":        "#17becf",   # tab10 청록
    "dual_mab":                 "#006400",   # 진초록 (tab10 초록과 명도 다름)
    "dual_mab_no_acb":          "#daa520",   # 황금색 (골드)
    "decentralized_q_learning": "#4b0082",   # 인디고
    "q_learning_abs":           "#00fa9a",   # 미디엄 스프링 그린
    # ER Q-learning variant 전용 (에너지 패널티만)
    "q_learning_er_baseline":   "#9467bd",   # 연보라
    "q_learning_er_etd":        "#d62728",   # 빨강
    "q_learning_er_em":         "#ff7f0e",   # 주황
    # ER Q-learning variant 전용 (에너지 상태 포함)
    "q_learning_er_etd_s18":    "#6a0dad",   # 보라 (SF_delta+Eb, 9states)
    "q_learning_er_etd_s17":    "#8b0000",   # 다크레드 (SF+Eb, 18states)
    "q_learning_er_etd_s15":    "#b85c00",   # 다크오렌지 (SF+Gmax+Eb, 54states)
    "q_learning_er_em_s18":     "#4b0070",   # 다크바이올렛 (EM SF_delta+Eb)
    "q_learning_er_em_s17":     "#c77100",   # 어두운황색 (EM SF+Eb)
    "q_learning_er_em_s15":     "#7b5200",   # 브라운 (EM SF+Gmax+Eb)
    # ER Q-learning variant — SF+CH / SF-delta+CH 상태공간 (채널 인식, 에너지 상태 없음)
    "q_learning_er_etd_s13":    "#1f77b4",   # 파랑 (ETD, SF+CH, 18states)
    "q_learning_er_etd_s14":    "#17becf",   # 시안 (ETD, SF-delta+CH, 9states)
}

REWARD_COLORS: dict[str, str] = {
    "v0_current":    "#1f77b4",
    "v1_split":      "#ff7f0e",
    "v2_aggressive": "#2ca02c",
    "v3_sparse":     "#d62728",
    "v4_balanced":   "#9467bd",
    "base":       "#17becf",
    "fair":       "#2ca02c",
    "explore":    "#ff7f0e",
    "congestion": "#8c564b",
}

# q_learning_abs 전용 연색 팔레트 (tab20 light — decentralized_q_learning의 진색과 쌍을 이룸)
Q_LEARNING_ABS_REWARD_COLORS: dict[str, str] = {
    "v0_current":    "#aec7e8",   # 연파랑
    "v1_split":      "#ffbb78",   # 연주황
    "v2_aggressive": "#98df8a",   # 연초록
    "v3_sparse":     "#ff9896",   # 연빨강
    "v4_balanced":   "#c5b0d5",   # 연보라
    "base":       "#9edae5",
    "fair":       "#98df8a",
    "explore":    "#ffbb78",
    "congestion": "#c49c94",
}

# ── 선 스타일 ─────────────────────────────────────────────────────────────────
BASELINE_LINESTYLES: dict[str, str | tuple] = {
    "pure_aloha":               "-",
    "adr_like":                 "-",
    "retry_aware":              "-",
    "lora_mab":                 "-",
    "lora_mab_neg":             "--",
    "lora_mab_idle":            "-",
    "lora_mab_idle_neg":        "--",
    "thompson_mab":             "-",
    "thompson_mab_idle":        "--",
    "dual_mab":                 "-",
    "dual_mab_no_acb":          "--",
    "decentralized_q_learning": (0, (3, 1, 1, 1)),   # 대시-점
    "q_learning_abs":           "--",
    "q_learning_er_baseline":   (0, (3, 1, 1, 1)),
    "q_learning_er_etd":        "-",
    "q_learning_er_em":         "--",
    "q_learning_er_etd_s18":    (0, (3, 1)),           # 짧은 대시 (SF_delta+Eb)
    "q_learning_er_etd_s17":    (0, (5, 2)),           # 긴 대시
    "q_learning_er_etd_s15":    (0, (5, 2, 1, 2)),     # 대시-점
    "q_learning_er_em_s18":     (0, (3, 1)),
    "q_learning_er_em_s17":     (0, (5, 2)),
    "q_learning_er_em_s15":     (0, (5, 2, 1, 2)),
    "q_learning_er_etd_s13":    "-",
    "q_learning_er_etd_s14":    "--",
}

# ── 마커 — 12개 모두 고유 도형 ────────────────────────────────────────────────
BASELINE_MARKERS: dict[str, str] = {
    "pure_aloha":               "o",    # ● 원
    "adr_like":                 "s",    # ■ 사각
    "retry_aware":              "^",    # ▲ 삼각
    "lora_mab":                 "D",    # ◆ 다이아몬드
    "lora_mab_neg":             "d",    # ◇ 슬림 다이아몬드
    "lora_mab_idle":            "v",    # ▼ 역삼각
    "lora_mab_idle_neg":        "<",    # ◀ 좌삼각
    "thompson_mab":             "P",    # ✚ 플러스(채움)
    "thompson_mab_idle":        "p",    # ⬠ 오각형
    "dual_mab":                 "*",    # ★ 별
    "dual_mab_no_acb":          "h",    # ⬡ 육각형
    "decentralized_q_learning": "X",    # ✖ X(채움)
    "q_learning_abs":           "8",    # ⬟ 팔각형
    "q_learning_er_baseline":   "X",
    "q_learning_er_etd":        "^",    # ▲ 삼각
    "q_learning_er_em":         "D",    # ◆ 다이아몬드
    "q_learning_er_etd_s18":    "H",    # ⬡ 회전육각형 (SF_delta+Eb)
    "q_learning_er_etd_s17":    "v",    # ▼ 역삼각
    "q_learning_er_etd_s15":    "<",    # ◀ 좌삼각
    "q_learning_er_em_s18":     "h",    # ⬡ 육각형
    "q_learning_er_em_s17":     "P",    # ✚ 플러스
    "q_learning_er_em_s15":     "p",    # ⬠ 오각형
    "q_learning_er_etd_s13":    ">",    # ▶ 우삼각 (ETD, SF+CH)
    "q_learning_er_etd_s14":    "1",    # ↓ 트라이다운 (ETD, SF-delta+CH)
}

REWARD_MARKERS: dict[str, str] = {
    "":              "o",
    "v0_current":    "o",
    "v1_split":      "s",
    "v2_aggressive": "^",
    "v3_sparse":     "D",
    "v4_balanced":   "v",
    "base":       "X",
    "fair":       "*",
    "explore":    "D",
    "congestion": "v",
}

# 그래프에서 참조할 기본 마커 크기 (plotting.py / app.py 에서 사용)
DEFAULT_MARKERSIZE = 9


# ── 공개 API ──────────────────────────────────────────────────────────────────
def series_color(base_key: str, reward_variant: str = "") -> str:
    if reward_variant:
        if base_key == "q_learning_abs":
            return Q_LEARNING_ABS_REWARD_COLORS.get(reward_variant, BASELINE_COLORS.get(base_key, "#888888"))
        return REWARD_COLORS.get(reward_variant, BASELINE_COLORS.get(base_key, "#888888"))
    return BASELINE_COLORS.get(base_key, "#888888")


def series_linestyle(base_key: str, reward_variant: str = "") -> str | tuple:
    return BASELINE_LINESTYLES.get(base_key, "-")


def series_marker(base_key: str, reward_variant: str = "") -> str:
    if reward_variant:
        return REWARD_MARKERS.get(reward_variant, BASELINE_MARKERS.get(base_key, "o"))
    return BASELINE_MARKERS.get(base_key, "o")
