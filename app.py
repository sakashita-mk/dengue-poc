# streamlit_app_single.py — Single-file demo (no backend needed)
# How to run locally:
# 1) pip install streamlit pandas numpy
# 2) streamlit run streamlit_app_single.py
#
# To deploy on Streamlit Community Cloud:
# - Push this file to a public GitHub repo
# - Create a new app in streamlit.io, select this file as the entry point


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
agg = st.radio("粒度 (Granularity)", ["adm", "grid1km"], format_func=lambda x: "行政区" if x=="adm" else "1kmグリッド")
base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
horizon = st.select_slider("予測ホライズン (weeks)", [0,1,2], value=2)


hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
fa = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)


if agg == "adm":
area_ids = ADM_AREAS
else:
area_ids = GRID_AREAS


default_sel = area_ids[:6]
sel = st.multiselect("対象エリア", options=area_ids, default=default_sel)


# -----------------------------------
# Prediction stub
# -----------------------------------
@st.cache_data(show_spinner=False)
def predict_stub(areas, base_day: date, horizon_wk: int, agg_level: str):
# derive week number for seasonality
week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2 # 0..1


out = []
for idx, a in enumerate(areas):
# seed each area so results are stable per area
local_rng = np.random.default_rng(abs(hash(a)) % (2**32))
base = 40 + 40*season
noise = local_rng.normal(0, 10)
