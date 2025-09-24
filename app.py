# app.py  — Streamlit single-file demo (no backend)
# Run:
#   pip install streamlit pandas numpy pydeck
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pydeck as pdk

# ---------- Page setup ----------
st.set_page_config(page_title="NCR Dengue — Single-file Demo", layout="wide")
st.title("NCR Dengue — Single-file Demo")
st.caption("粒度トグル / 動的日付 / 目標スライダー / マップ + 時系列（デモ用スタブ推論）")

# ---------- Synthetic metadata ----------
CENTER_LAT, CENTER_LON = 14.5995, 120.9842
rng = np.random.default_rng(7)

ADM_AREAS = [f"ADM3-{i:03d}" for i in range(1, 25)]
GRID_AREAS = [f"GRID-{i:03d}" for i in range(1, 41)]

# 安定したダミー座標（地図表示用）
adm_positions = {}
for i, a in enumerate(ADM_AREAS):
    lat = CENTER_LAT + ((i % 6) - 2.5) * 0.045
    lon = CENTER_LON + ((i // 6) - 2.0) * 0.055
    adm_positions[a] = (lat, lon)

grid_positions = {}
for i, g in enumerate(GRID_AREAS):
    lat = CENTER_LAT + ((i % 8) - 3.5) * 0.028
    lon = CENTER_LON + ((i // 8) - 2.5) * 0.035
    grid_positions[g] = (lat, lon)

# 直近78週（月曜）
weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    agg = st.radio(
        "粒度 (Granularity)",
        ["adm", "grid1km"],
        format_func=lambda x: "行政区" if x == "adm" else "1kmグリッド",
    )
    base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
    horizon = st.select_slider("予測ホライズン (weeks)", [0, 1, 2], value=2)

    hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
    fa = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
    mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)

    area_ids = ADM_AREAS if agg == "adm" else GRID_AREAS
    sel = st.multiselect("対象エリア", options=area_ids, default=area_ids[:6])

# ---------- Prediction stub ----------
@st.cache_data(show_spinner=False)
def predict_stub(areas, base_day: date, horizon_wk: int, agg_level: str) -> pd.DataFrame:
    """季節性 + ノイズのデモ用スコア生成"""
    week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
    season = (np.sin(2 * np.pi * week_num / 52.0) + 1) / 2  # 0..1

    rows = []
    for a in areas:
        local_rng = np.random.default_rng(abs(hash(a)) % (2**32))
        base = 40 + 40 * season
        noise = local_rng.normal(0, 10)
        score = float(np.clip(base + noise + 10 * horizon_wk, 0, 100))
        level = "low" if score < 33 else ("med" if score < 66 else "high")
        drivers = "rain↑, NDVI↓" if season > 0.5 else "LST↑, NDVI↓"

        if agg_level == "adm":
            lat, lon = adm_positions[a]
        else:
            lat, lon = grid_positions[a]

        rows.append(
            {
                "area": a,
                "risk_score": round(score, 1),
                "risk_level": level,
                "drivers": drivers,
                "lat": lat,
                "lon": lon,
                "horizon_wk": horizon_wk,
                "base_week": base_day.isoformat(),
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def make_timeseries(area: str, end_day: date, weeks_back: int = 12) -> pd.DataFrame:
    """選択エリアの簡易リスク時系列（デモ用）"""
    days = [end_day - timedelta(weeks=w) for w in range(weeks_back, -1, -1)]
    rows = []
    for i, d in enumerate(days):
        week_num = int(datetime.combine(d, datetime.min.time()).strftime("%U"))
        season = (np.sin(2 * np.pi * week_num / 52.0) + 1) / 2
        local_rng = np.random.default_rng(abs(hash(area)) % (2**32) + i)
        base = 40 + 40 * season
        noise = local_rng.normal(0, 9)
        score = float(np.clip(base + noise, 0, 100))
        rows.append({"date": d, "risk_score": score})
    return pd.DataFrame(rows)

# ---------- Run ----------
if not sel:
    st.warning("対象エリアを1つ以上選んでください。")
    st.stop()

pred_df = predict_stub(sel, base_date, horizon, agg)

# ---------- Map (pydeck) ----------
st.subheader("Risk Map")
map_df = pred_df[["lat", "lon", "risk_score", "risk_level", "area"]].copy()

# 色は risk_level に応じて H(色相) を変える（pydeckの式）
color_expr = [
    "risk_level == 'low' ? 60 : risk_level == 'med' ? 180 : 350",
    "80",
    "80",
]
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[lon, lat]",
    get_radius=1000 if agg == "adm" else 600,
    radius_min_pixels=6,
    radius_max_pixels=30,
    get_fill_color=color_expr,
    pickable=True,
    auto_highlight=True,
)
view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{area}\nRisk: {risk_score}"},
    )
)

# ---------- Table + timeseries ----------
st.subheader("Predictions")
st.dataframe(
    pred_df[["area", "risk_score", "risk_level", "drivers", "horizon_wk", "base_week"]],
    use_container_width=True,
)

focus_area = sel[0]
with st.expander(f"Timeseries — {focus_area}"):
    ts = make_timeseries(focus_area, base_date)
    st.line_chart(ts.set_index("date"))

# ---------- Targets (visual only) ----------
with st.sidebar:
    st.markdown("---")
    st.caption("※ 目標は可視化のみ（デモ）。本番では警報閾値・誤警報評価に反映。")
    st.metric("Hit率 目標", f"{int(hit*100)}%")
    st.metric("過警報 許容", f"{int(fa*100)}%")
    st.metric("MAE改善 目標", f"{int(mae*100)}%")

st.success("稼働中：粒度・日付・ホライズン・スライダーを動かして挙動を確認してください。")
