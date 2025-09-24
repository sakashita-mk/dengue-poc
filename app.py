# streamlit_app_single.py — Single-file demo (no backend needed)
# How to run locally:
#   1) pip install streamlit pandas numpy
#   2) streamlit run streamlit_app_single.py
#
# To deploy on Streamlit Community Cloud:
#   - Push this file to a public GitHub repo
#   - Create a new app in streamlit.io, select this file as the entry point

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pydeck as pdk

st.set_page_config(page_title="NCR Dengue — Single-file Demo", layout="wide")
st.title("NCR Dengue — Single-file Demo")
st.caption("Granularity toggle • dynamic date • target sliders • map + timeseries (all stubbed)")

# -----------------------------------
# Synthetic metadata (areas + grid)
# -----------------------------------
# NCR approximate center
CENTER_LAT, CENTER_LON = 14.5995, 120.9842
rng = np.random.default_rng(7)

ADM_AREAS = [f"ADM3-{i:03d}" for i in range(1, 25)]
GRID_AREAS = [f"GRID-{i:03d}" for i in range(1, 41)]

# Stable pseudo-random positions around center
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

# Weeks list (last 78 Mondays)
weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")
weeks_str = [w.strftime("%Y-%m-%d") for w in weeks]

# -----------------------------------
# Controls (sidebar)
# -----------------------------------
with st.sidebar:
    st.header("Controls")
    agg = st.radio(
        "粒度 (Granularity)",
        ["adm", "grid1km"],
        format_func=lambda x: "行政区" if x == "adm" else "1kmグリッド",
        help="地図の集計粒度を切り替えます。行政区=市区等の境界、1kmグリッド=等間隔メッシュ"
    )
    base_date = st.date_input(
        "基準日 (Week base)",
        value=weeks[-1].date(),
        help="予測の基準となる週の月曜日を選びます"
    )
    horizon = st.select_slider(
        "予測ホライズン (weeks)", [0, 1, 2], value=2,
        help="基準日から何週間先を表示するか（0=当週, 1=来週, 2=再来週）"
    )

    hit = st.slider(
        "Hit率目標", 0.0, 0.9, 0.6, 0.05,
        help="Highリスク週を事前に当てられる割合の目標値（参考：60%）"
    )
    fa = st.slider(
        "過警報許容", 0.0, 0.5, 0.3, 0.05,
        help="不要なHigh警報を許容する上限（参考：30%）"
    )
    mae = st.slider(
        "MAE改善目標", 0.0, 0.4, 0.2, 0.05,
        help="季節平均などのベースラインに対して、誤差をどの程度縮めたいか"
    )

    area_ids = ADM_AREAS if agg == "adm" else GRID_AREAS
    sel = st.multiselect(
        "対象エリア",
        options=area_ids,
        default=area_ids[:6],
        help="地図と表に表示する地域を選びます（複数選択可）"
    )

# -----------------------------------
# Prediction stub
# -----------------------------------
@st.cache_data(show_spinner=False)
def predict_stub(areas, base_day: date, horizon_wk: int, agg_level: str):
    # derive week number for seasonality
    week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
    season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2  # 0..1

    out = []
    for idx, a in enumerate(areas):
        # seed each area so results are stable per area
        local_rng = np.random.default_rng(abs(hash(a)) % (2**32))
        base = 40 + 40*season
        noise = local_rng.normal(0, 10)
        score = float(np.clip(base + noise + 10*horizon_wk, 0, 100))
        level = "low" if score < 33 else ("med" if score < 66 else "high")
        drivers = ["rain↑" if season>0.5 else "LST↑", "NDVI↓"]

        if agg_level == "adm":
            lat, lon = adm_positions[a]
        else:
            lat, lon = grid_positions[a]

        out.append({
            "area": a,
            "risk_score": round(score, 1),
            "risk_level": level,
            "drivers": ", ".join(drivers),
            "lat": lat,
            "lon": lon,
            "horizon_wk": horizon_wk,
            "base_week": base_day.isoformat(),
        })
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def make_timeseries(area: str, end_day: date, agg_level: str, weeks_back: int = 12):
    days = [end_day - timedelta(weeks=w) for w in range(weeks_back, -1, -1)]
    rows = []
    for i, d in enumerate(days):
        week_num = int(datetime.combine(d, datetime.min.time()).strftime("%U"))
        season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2
        local_rng = np.random.default_rng(abs(hash(area)) % (2**32) + i)
        base = 40 + 40*season
        noise = local_rng.normal(0, 9)
        score = float(np.clip(base + noise, 0, 100))
        rows.append({"date": d, "risk_score": score})
    return pd.DataFrame(rows)

# Run prediction
if not sel:
    st.warning("対象エリアを1つ以上選んでください。")
    st.stop()

pred_df = predict_stub(sel, base_date, horizon, agg)

# -----------------------------------
# Map visualization (pydeck)
# -----------------------------------
st.subheader("Risk Map")
map_df = pred_df[["lat","lon","risk_score","risk_level","area"]].copy()
color_expr = [
    "risk_level == 'low' ? 60 : risk_level == 'med' ? 180 : 350",
    "80",
    "80"
]
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[lon, lat]',
    get_radius=1000 if agg=="adm" else 600,
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
    ),
    height=720,   # ★ ここを追加（好みで 650〜900 くらい）
)


# -----------------------------------
# Table + details
# -----------------------------------
st.subheader("Predictions")
st.dataframe(pred_df[["area","risk_score","risk_level","drivers","horizon_wk","base_week"]], use_container_width=True)

# Timeseries for the first selected area
focus_area = sel[0]
with st.expander(f"Timeseries — {focus_area}"):
    ts = make_timeseries(focus_area, base_date, agg)
    st.line_chart(ts.set_index("date"))

# -----------------------------------
# Targets panel (visual only)
# -----------------------------------

st.success("デモ稼働中：粒度トグル、日付、ホライズン、目標スライダーを動かして挙動を確認してください。")

