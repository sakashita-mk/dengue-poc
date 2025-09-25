# app.py — NCR Dengue demo (single file)
# Run:
#   pip install streamlit pandas numpy pydeck geopandas shapely pyproj requests
#   streamlit run app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import box
import requests

def _hash_gdf(gdf: gpd.GeoDataFrame):
    # 形状のバウンディングボックスを丸めてハッシュのキーに
    # （厳密にしたければ unary_union.wkb でもOK）
    import numpy as np
    return tuple(np.round(gdf.total_bounds, 6))

# 追加：幾何簡略化ユーティリティ
def simplify_gdf(gdf: gpd.GeoDataFrame, tol_m: float = 50) -> gpd.GeoDataFrame:
    g = gdf.to_crs(32651).copy()           # NCRはUTM51NでOK
    g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
    g = g.to_crs(4326)
    g["geometry"] = g.buffer(0)            # 念のため修復
    return g

# ---------- Page setup ----------
st.set_page_config(page_title="NCR Dengue — PoC Demo", layout="wide")
st.title("NCR Dengue — PoC Demo")
st.caption("行政区ポリゴン / 1kmグリッド / 粒度トグル / 動的日付 / 目標スライダー（予測はデモ用スタブ）")

CENTER_LAT, CENTER_LON = 14.5995, 120.9842

# ---------- Helper: geoBoundaries ----------
@st.cache_data(show_spinner=False)
def load_geoboundaries_ncr():
    # ■ GDAL/pyogrio の GeoJSON サイズ上限を引き上げ（MB単位）
    #   例: 2000MB（十分大きめ）
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "2000"

    # ADM1/ADM2 メタデータを取得（配列で返る）
    adm1_meta = requests.get(
        "https://www.geoboundaries.org/api/current/gbOpen/PHL/ADM1/", timeout=30
    ).json()
    adm2_meta = requests.get(
        "https://www.geoboundaries.org/api/current/gbOpen/PHL/ADM2/", timeout=30
    ).json()

    # ■ 念のため配列対応：最初の要素から gjDownloadURL を取得
    def pick_url(meta):
        if isinstance(meta, list) and len(meta) > 0:
            m = meta[0]
        elif isinstance(meta, dict):
            m = meta
        else:
            raise ValueError("geoBoundaries API から想定外のレスポンス")
        return m["gjDownloadURL"]

    adm1_url = pick_url(adm1_meta)
    adm2_url = pick_url(adm2_meta)

    # ■ pyogrio にも直接 config_options を渡して上限を拡張
    read_kwargs = dict(engine="pyogrio",
                       config_options={"OGR_GEOJSON_MAX_OBJ_SIZE": "2000"})

    adm1 = gpd.read_file(adm1_url, **read_kwargs).to_crs(4326)
    adm2 = gpd.read_file(adm2_url, **read_kwargs).to_crs(4326)

    # 名称カラム推定（geoBoundaries は shapeName が基本）
    name_cols = [c for c in adm1.columns if "name" in c.lower()] or list(adm1.columns)
    name_col_1 = name_cols[0]

    # NCR 名称ゆらぎに対応
    ncr_keywords = ["national capital region", "metropolitan manila", "ncr"]
    mask = adm1[name_col_1].astype(str).str.lower().apply(
        lambda s: any(k in s for k in ncr_keywords)
    )
    ncr = adm1[mask].copy()
    if ncr.empty:
        raise ValueError("NCR を ADM1 から特定できませんでした。")

    # 幾何修復＋ explode
    ncr["geometry"] = ncr.buffer(0)
    ncr = ncr.explode(index_parts=False).reset_index(drop=True)

    # ADM2 側も修復
    adm2 = adm2.copy()
    adm2["geometry"] = adm2.buffer(0)

    # clip の方が overlay より壊れにくい
    adm2_in_ncr = gpd.clip(adm2, ncr[["geometry"]])
    adm2_in_ncr = adm2_in_ncr.explode(index_parts=False).reset_index(drop=True)

    ncr = simplify_gdf(ncr, 50)
    adm2_in_ncr = simplify_gdf(adm2_in_ncr, 50)
    
    return ncr, adm2_in_ncr

def to_polygon_coords(g):
    if g.geom_type == "Polygon":
        return [list(map(list, g.exterior.coords))]
    return []


@st.cache_data(
    show_spinner=False,
    hash_funcs={gpd.GeoDataFrame: _hash_gdf}  # ← これがポイント
)
def make_grid_over_ncr(ncr_gdf):
    """NCR(4326)→UTM(32651) に投影して 1km 格子を生成。戻す際に座標 & 幾何を整形。"""
    ncr_utm = ncr_gdf.to_crs(32651).copy()
    ncr_utm["geometry"] = ncr_utm.buffer(0)
    minx, miny, maxx, maxy = ncr_utm.total_bounds

    cell = 1000  # 1km
    union = ncr_utm.unary_union
    cells = []
    for x in np.arange(minx, maxx, cell):
        for y in np.arange(miny, maxy, cell):
            geom = box(x, y, x + cell, y + cell)
            if union.intersects(geom):
                cells.append(geom)

    grid = gpd.GeoDataFrame(geometry=cells, crs=32651).to_crs(4326)
    grid = simplify_gdf(grid, 50)
    grid["geometry"] = grid.buffer(0)
    grid = grid.explode(index_parts=False).reset_index(drop=True)
    grid["coordinates"] = grid.geometry.apply(
        lambda g: [list(map(list, g.exterior.coords))] if g.geom_type == "Polygon" else []
    )
    return grid

# ---------- Synthetic fallback positions ----------
ADM_AREAS = [f"ADM3-{i:03d}" for i in range(1, 25)]
GRID_AREAS = [f"GRID-{i:03d}" for i in range(1, 41)]
adm_positions, grid_positions = {}, {}
for i, a in enumerate(ADM_AREAS):
    adm_positions[a] = (CENTER_LAT + ((i % 6) - 2.5) * 0.045,
                        CENTER_LON + ((i // 6) - 2.0) * 0.055)
for i, g in enumerate(GRID_AREAS):
    grid_positions[g] = (CENTER_LAT + ((i % 8) - 3.5) * 0.028,
                         CENTER_LON + ((i // 8) - 2.5) * 0.035)

weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
    horizon = st.select_slider("予測ホライズン (weeks)", [0,1,2], value=2)
    hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
    fa  = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
    mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)

    # 表示モード切替
    poly_mode = st.radio(
        "表示ポリゴン",
        ["行政区(ADM)", "1kmグリッド", "表示しない"],
        index=0
    )

    agg = st.radio("粒度 (Granularity)", ["adm", "grid1km"],
                   format_func=lambda x: "行政区" if x=="adm" else "1kmグリッド")
    area_ids = ADM_AREAS if agg=="adm" else GRID_AREAS
    sel = st.multiselect("対象エリア", options=area_ids, default=area_ids[:6])

# ---------- Prediction stub ----------
@st.cache_data(show_spinner=False)
def predict_stub(areas, base_day: date, horizon_wk: int, agg_level: str):
    week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
    season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2
    out = []
    for a in areas:
        rng = np.random.default_rng(abs(hash(a)) % (2**32))
        base = 40 + 40*season
        noise = rng.normal(0, 10)
        score = float(np.clip(base + noise + 10*horizon_wk, 0, 100))
        level = "low" if score < 33 else ("med" if score < 66 else "high")
        drivers = "rain↑, NDVI↓" if season>0.5 else "LST↑, NDVI↓"
        if agg_level=="adm": lat, lon = adm_positions[a]
        else: lat, lon = grid_positions[a]
        out.append(dict(area=a, risk_score=round(score,1), risk_level=level,
                        drivers=drivers, lat=lat, lon=lon,
                        horizon_wk=horizon_wk, base_week=base_date.isoformat()))
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def make_timeseries(area: str, end_day: date, weeks_back: int = 12):
    days = [end_day - timedelta(weeks=w) for w in range(weeks_back, -1, -1)]
    rows = []
    for i, d in enumerate(days):
        week_num = int(datetime.combine(d, datetime.min.time()).strftime("%U"))
        season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2
        rng = np.random.default_rng(abs(hash(area)) % (2**32) + i)
        score = float(np.clip(40+40*season + rng.normal(0,9), 0, 100))
        rows.append({"date": d, "risk_score": score})
    return pd.DataFrame(rows)

if not sel:
    st.warning("対象エリアを1つ以上選んでください。")
    st.stop()

pred_df = predict_stub(sel, base_date, horizon, agg)

# ---------- Map ----------
st.subheader("Risk Map")
view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10)
layers = []
try:
    ncr_gdf, adm2_in_ncr = load_geoboundaries_ncr()

    if poly_mode == "行政区(ADM)":
        ncr_gdf["coordinates"] = ncr_gdf.geometry.apply(to_polygon_coords)
        layers.append(pdk.Layer(
            "PolygonLayer", data=ncr_gdf, get_polygon="coordinates",
            get_fill_color=[200,30,0,25], stroked=True, get_line_color=[200,30,0],
            line_width_min_pixels=1, pickable=True
        ))
        if not adm2_in_ncr.empty:
            adm2_in_ncr["coordinates"] = adm2_in_ncr.geometry.apply(to_polygon_coords)
            layers.append(pdk.Layer(
                "PolygonLayer", data=adm2_in_ncr, get_polygon="coordinates",
                get_fill_color=[0,120,200,12], stroked=True, get_line_color=[0,120,200],
                line_width_min_pixels=1, pickable=True
            ))

    elif poly_mode == "1kmグリッド":
        grid = make_grid_over_ncr(ncr_gdf)
        grid["coordinates"] = grid.geometry.apply(to_polygon_coords)
        layers.append(pdk.Layer(
            "PolygonLayer", data=grid, get_polygon="coordinates",
            get_fill_color=[255,255,0,8], stroked=True, get_line_color=[255,255,0],
            line_width_min_pixels=1
        ))

    # 点レイヤ（粒度に関係なく表示）
    map_df = pred_df[["lat","lon","risk_score","risk_level","area"]].copy()
    # 選択行ハイライト対応（次節）
    color_expr = [
        f"(properties.area == '{st.session_state.get('highlight_area','')}')"
        " ? 0 : (properties.risk_level == 'low' ? 60 : "
        "(properties.risk_level == 'med' ? 180 : 350))",
        "(properties.area == '" + st.session_state.get('highlight_area','') + "') ? 255 : 80",
        "(properties.area == '" + st.session_state.get('highlight_area','') + "') ? 0 : 80"
    ]
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=map_df,
        get_position='[lon, lat]',
        get_radius=1200 if agg=="adm" else 700,
        radius_min_pixels=6, radius_max_pixels=30,
        get_fill_color=color_expr, pickable=True, auto_highlight=True
    ))

except Exception as e:
    st.error("ポリゴン描画に失敗しました。詳細ログを下に表示します。")
    st.exception(e)
else:
    map_df = pred_df[["lat","lon","risk_score","risk_level","area"]].copy()
    color_expr = ["risk_level == 'low' ? 60 : risk_level == 'med' ? 180 : 350","80","80"]
    scatter = pdk.Layer("ScatterplotLayer", data=map_df,
                        get_position='[lon, lat]',
                        get_radius=1000 if agg=="adm" else 600,
                        radius_min_pixels=6, radius_max_pixels=30,
                        get_fill_color=color_expr, pickable=True, auto_highlight=True)
    layers.append(scatter)

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state), height=720)

# ---------- Table ----------
st.subheader("Predictions")
table = pred_df[["area","risk_score","risk_level","drivers","horizon_wk","base_week"]]

left, right = st.columns([4,1])
with left:
    st.dataframe(table, use_container_width=True, hide_index=True)
with right:
    highlight = st.selectbox("ハイライト", options=["(なし)"] + table["area"].tolist())
    if highlight != "(なし)":
        st.session_state["highlight_area"] = highlight
    else:
        st.session_state["highlight_area"] = ""
        
# ---------- Targets ----------
with st.sidebar:
    st.markdown("---")
    st.metric("Hit率 目標", f"{int(hit*100)}%")
    st.metric("過警報 許容", f"{int(fa*100)}%")
    st.metric("MAE改善 目標", f"{int(mae*100)}%")

st.success("稼働中：粒度・日付・ホライズン・スライダーを動かして挙動を確認してください。")
