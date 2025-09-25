# app.py — NCR Dengue demo (single file)
# Run:
#   pip install streamlit pandas numpy pydeck geopandas shapely pyproj requests
#   streamlit run app.py

import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.makedirs("/tmp/.streamlit", exist_ok=True)
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
    
# 0–100 の score を RGB に変換（青→黄→赤）
def risk_to_rgb(score: float):
    s = max(0.0, min(100.0, float(score))) / 100.0
    # カラーストップ: 0=青(49,130,189) → 0.5=黄(255,237,160) → 1=赤(220,20,60)
    stops = [(0.0, (49,130,189)), (0.5, (255,237,160)), (1.0, (220,20,60))]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if s <= p1:
            t = 0.0 if p1 == p0 else (s - p0) / (p1 - p0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)
    return stops[-1][1]

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
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [list(map(list, g.exterior.coords))]
    elif g.geom_type == "MultiPolygon":
        return [list(map(list, poly.exterior.coords)) for poly in g.geoms]
    else:
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

# ---------- Real areas & positions built from actual geoms ----------

# 1) 一度だけ境界を取得（cacheされる）
ncr_gdf_cache, adm2_in_ncr_cache = load_geoboundaries_ncr()

# 2) ADM（市区）ID＝名称（実データ）
ADM_NAME_COL = [c for c in adm2_in_ncr_cache.columns if "name" in c.lower()][0]

# 同名を統合（マルチをまとめる）→ 代表点を安全に取得
adm_diss = adm2_in_ncr_cache[[ADM_NAME_COL, "geometry"]].dissolve(by=ADM_NAME_COL)
adm_diss = adm_diss[~adm_diss.geometry.is_empty].to_crs(4326).copy()
adm_pts  = adm_diss.geometry.representative_point()  # 内部に必ず落ちる点

ADM_AREAS = adm_diss.index.astype(str).tolist()
adm_positions = {name: (pt.y, pt.x) for name, pt in zip(ADM_AREAS, adm_pts)}

# 3) 1kmグリッド（実件数ぶんIDを振る）
grid_cache = make_grid_over_ncr(ncr_gdf_cache).reset_index(drop=True)
grid_cache = grid_cache[~grid_cache.geometry.is_empty].copy()
grid_cache["grid_id"] = [f"GRID-{i+1:03d}" for i in range(len(grid_cache))]
GRID_AREAS = grid_cache["grid_id"].tolist()

grid_pts = grid_cache.geometry.representative_point()
grid_positions = {gid: (pt.y, pt.x) for gid, pt in zip(GRID_AREAS, grid_pts)}


# --- 空間相関ノイズを作るユーティリティ（RBFガウスカーネル） ---
def make_spatial_field(pos_dict, length_km=8.0, seed=123):
    ids  = list(pos_dict.keys())
    lats = np.array([pos_dict[i][0] for i in ids])
    lons = np.array([pos_dict[i][1] for i in ids])
    lat0 = float(lats.mean())
    # 緯度経度→km近似（NCR程度のスケールならOK）
    x = (lons - lons.mean()) * np.cos(np.deg2rad(lat0)) * 111.0
    y = (lats - lats.mean()) * 111.0
    X = np.vstack([x, y]).T
    # 距離の二乗行列
    d2 = ((X[:, None, :] - X[None, :, :])**2).sum(axis=2)
    K = np.exp(-d2 / (2 * (length_km**2)))         # RBFカーネル
    rng = np.random.default_rng(seed)
    z = rng.normal(size=len(ids))
    field = K @ z                                   # 相関ノイズ
    field = (field - field.mean()) / (field.std() + 1e-6)  # 標準化
    field = field * 10.0                            # だいたい ±10点の振幅
    return {i: float(v) for i, v in zip(ids, field)}

# ADM/GRID 用に一度だけ前計算（スケールは適宜調整）
USE_SPATIAL_NOISE = True  # ← まずは安全にOFF（後でTrueにすれば復活）

if USE_SPATIAL_NOISE:
    SPATIAL_NOISE_ADM  = make_spatial_field(adm_positions,  length_km=10.0, seed=42)
    SPATIAL_NOISE_GRID = make_spatial_field(grid_positions, length_km=6.0,  seed=42)
else:
    SPATIAL_NOISE_ADM, SPATIAL_NOISE_GRID = {}, {}

weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
    horizon = st.select_slider("予測ホライズン (weeks)", [0,1,2], value=2)
    hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
    fa  = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
    mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)

    granularity = st.radio("粒度 / 表示レイヤー", ["行政区", "1kmグリッド"], index=0)

    # 地図スタイルの選択だけ（レイヤは後段で作る）
    basemap = st.radio("地図スタイル（トークン不要）",
                       ["OSM標準", "CARTOライト", "CARTOダーク", "ESRI衛星"], index=2)

    show_choropleth = st.checkbox("ヒートマップ表示（面を色塗り）", value=True)
    hide_polygons = st.checkbox("ポリゴンを非表示にする", value=False)

    if granularity == "行政区":
        agg = "adm"
        area_ids = ADM_AREAS
    else:
        agg = "grid1km"
        area_ids = GRID_AREAS

    sel = st.multiselect("対象エリア", options=area_ids, default=area_ids[:10])

# ---------- Prediction stub ----------
@st.cache_data(show_spinner=False)
def predict_stub(areas, base_day: date, horizon_wk: int, agg_level: str):
    week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
    season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2
    out = []
    for a in areas:
        rng = np.random.default_rng(abs(hash(a)) % (2**32))
        base = 40 + 40*season

        # ★ 空間相関ノイズを加える（地域で“まとまり”が出る）
        #spatial = 0.0
        spatial = (SPATIAL_NOISE_ADM.get(a, 0.0) if agg_level == "adm"
                   else SPATIAL_NOISE_GRID.get(a, 0.0))
        # 個別の微小ノイズ（粒度合わせ）
        noise_iid = rng.normal(0, 3)

        score = float(np.clip(base + spatial + noise_iid + 10*horizon_wk, 0, 100))
        level = "low" if score < 33 else ("med" if score < 66 else "high")
        drivers = "rain↑, NDVI↓" if season > 0.5 else "LST↑, NDVI↓"
        if agg_level == "adm":
            lat, lon = adm_positions[a]
        else:
            lat, lon = grid_positions[a]
        out.append(dict(area=a, risk_score=round(score, 1), risk_level=level,
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

# 面塗り用：全エリアのスコア（テーブルは sel のままでOK）
areas_for_map = ADM_AREAS if agg=="adm" else GRID_AREAS
pred_map = predict_stub(areas_for_map, base_date, horizon, agg)[["area","risk_score"]]

# ---------- Table & Highlight control (before Map) ----------
st.subheader("Predictions")
table = pred_df[["area","risk_score","risk_level","drivers","horizon_wk","base_week"]]
left, right = st.columns([4,1])

# 10行ぶんの高さを確保（行が10未満でも見やすい高さ）
ROWS_TO_SHOW = 10
ROW_H = 34   # 1行あたりの目安高さ（環境で±数px差あり）
HEADER_H = 38
height = HEADER_H + ROW_H * max(ROWS_TO_SHOW, 1)

with left:
    st.dataframe(table, use_container_width=True, hide_index=True)
with right:
    highlight = st.selectbox("ハイライト", options=["(なし)"] + table["area"].tolist(), key="hl")
highlight_id = "" if highlight == "(なし)" else highlight

# ---------- Map ----------
st.subheader("Risk Map")
view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10)
layers = []

try:
    ncr_gdf, adm2_in_ncr = load_geoboundaries_ncr()

    if agg == "adm":
        # ベース形状（市区を1面に統合）
        name_col = [c for c in adm2_in_ncr.columns if "name" in c.lower()][0]
        adm_base = adm2_in_ncr[[name_col, "geometry"]].dissolve(by=name_col).reset_index()
        adm_base["area"] = adm_base[name_col].astype(str)
        # 予測と結合 → 色列を作成
        adm_base = adm_base.merge(pred_map, on="area", how="left")
        rgb = adm_base["risk_score"].fillna(0).apply(risk_to_rgb)
        adm_base[["r","g","b"]] = pd.DataFrame(rgb.tolist(), index=adm_base.index)
        adm_base["a"] = 180  # 透過
        adm_base["coordinates"] = adm_base.geometry.apply(to_polygon_coords)

        if not hide_polygons:
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=adm_base,
                get_polygon="coordinates",
                get_fill_color='[r,g,b,a]' if show_choropleth else [0,0,0,0],
                stroked=True,
                get_line_color=[200,30,0] if not show_choropleth else [255,255,255,60],
                line_width_min_pixels=1,
                pickable=True,
            ))

    elif agg == "grid1km":
        # グリッド全面表示（常に）＋コロプレス塗り
        grid = grid_cache.copy()
        grid = grid.merge(pred_map.rename(columns={"area":"grid_id"}), on="grid_id", how="left")
        rgb = grid["risk_score"].fillna(0).apply(risk_to_rgb)
        grid[["r","g","b"]] = pd.DataFrame(rgb.tolist(), index=grid.index)
        grid["a"] = 160
        grid["coordinates"] = grid.geometry.apply(to_polygon_coords)

        layers.append(pdk.Layer(
            "PolygonLayer",
            data=grid,
            get_polygon="coordinates",
            get_fill_color='[r,g,b,a]' if show_choropleth else [0,0,0,0],
            stroked=True,
            get_line_color=[255,255,0] if not show_choropleth else [255,255,255,60],
            line_width_min_pixels=1,
            pickable=True,
        ))

    # --- 面ハイライト（選択ID） ---
    if highlight_id:
        if agg == "adm":
            h = adm2_in_ncr[adm2_in_ncr[name_col].astype(str) == highlight_id]
            if not h.empty:
                h = h.dissolve(by=name_col).explode(index_parts=False).reset_index(drop=True)
                h["coordinates"] = h.geometry.apply(to_polygon_coords)
                layers.append(pdk.Layer(
                    "PolygonLayer",
                    data=h, get_polygon="coordinates",
                    get_fill_color=[255,0,255,50],
                    stroked=True, get_line_color=[255,255,255],
                    line_width_min_pixels=3, pickable=False
                ))
        else:
            h = grid_cache[grid_cache["grid_id"] == highlight_id].copy()
            if not h.empty:
                h["coordinates"] = h.geometry.apply(to_polygon_coords)
                layers.append(pdk.Layer(
                    "PolygonLayer",
                    data=h, get_polygon="coordinates",
                    get_fill_color=[255,0,255,50],
                    stroked=True, get_line_color=[255,255,255],
                    line_width_min_pixels=3, pickable=False
                ))

except Exception as e:
    st.error("ポリゴン描画に失敗しました。詳細ログを下に表示します。")
    st.exception(e)

TILES = {
    "OSM標準": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "minzoom": 0, "maxzoom": 19,
    },
    "CARTOライト": {
        "url": "https://a.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png",
        "attribution": "© CARTO © OpenStreetMap contributors",
        "minzoom": 0, "maxzoom": 19,
    },
    "CARTOダーク": {
        "url": "https://a.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
        "attribution": "© CARTO © OpenStreetMap contributors",
        "minzoom": 0, "maxzoom": 19,
    },
    "ESRI衛星": {
        "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Source: Esri, Maxar, Earthstar Geographics",
        "minzoom": 0, "maxzoom": 19,
    },
}
tile = TILES[basemap]
basemap_layer = pdk.Layer(
    "TileLayer",
    data={
        "tilejson": "2.2.0",
        "tiles": [tile["url"]],
        "minzoom": tile["minzoom"],
        "maxzoom": tile["maxzoom"],
        "attribution": tile["attribution"],
        "name": basemap,
    },
    opacity=1.0,
)
layers = [basemap_layer] + layers  # ← ここなら layers は既に定義済み

# （任意）凡例の簡易表示
if show_choropleth:
    st.caption("Heatmap scale: 低← 青 — 黄 — 赤 →高")

deck_key = f"deck-{basemap}-{agg}-{int(show_choropleth)}-{int(hide_polygons)}-{highlight_id}"

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None, 
    ),
    height=720,
    key=deck_key,
)
     
# ---------- Targets ----------
with st.sidebar:
    st.markdown("---")
    st.metric("Hit率 目標", f"{int(hit*100)}%")
    st.metric("過警報 許容", f"{int(fa*100)}%")
    st.metric("MAE改善 目標", f"{int(mae*100)}%")

st.success("稼働中：粒度・日付・ホライズン・スライダーを動かして挙動を確認してください。")


