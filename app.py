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

weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
    horizon = st.select_slider("予測ホライズン (weeks)", [0,1,2], value=2)
    hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
    fa  = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
    mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)

    # ✅ 表示ポリゴンと粒度を統合
    granularity = st.radio(
        "粒度 / 表示レイヤー",
        ["行政区", "1kmグリッド"],
        index=0
    )

    # 「表示しない」チェックボックス（任意）
    hide_polygons = st.checkbox("ポリゴンを非表示にする", value=False)

    # エリアIDは選択した粒度で切り替え
    if granularity == "行政区":
        agg = "adm"
        area_ids = ADM_AREAS
    else:
        agg = "grid1km"
        area_ids = GRID_AREAS

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

# ---------- Table & Highlight control (before Map) ----------
st.subheader("Predictions")
table = pred_df[["area","risk_score","risk_level","drivers","horizon_wk","base_week"]]
left, right = st.columns([4,1])
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

    if not hide_polygons:   # ← サイドバーでチェックされていなければ描画
    # --- ここから追加：選択エリアのポリゴンをハイライト ---
    # テーブル側 selectbox の値（既にどこかで highlight_id を作っている前提）
    # 例: highlight_id = st.session_state.get("highlight_area", "")
        if highlight_id:
            if agg == "adm":
                # 市区名で抽出→統合→描画
                name_col = [c for c in adm2_in_ncr.columns if "name" in c.lower()][0]
                h = adm2_in_ncr[adm2_in_ncr[name_col].astype(str) == highlight_id]
                if not h.empty:
                    h = h.dissolve(by=name_col)
                    h = h.explode(index_parts=False).reset_index(drop=True)
                    h["coordinates"] = h.geometry.apply(to_polygon_coords)
                    layers.append(pdk.Layer(
                        "PolygonLayer",
                        data=h,
                        get_polygon="coordinates",
                        get_fill_color=[255, 0, 255, 50],   # うっすらマゼンタ塗り
                        stroked=True,
                        get_line_color=[255, 255, 255],    # 太めの白枠
                        line_width_min_pixels=3,
                        pickable=False,
                    ))
            elif agg == "grid1km":
                h = grid_cache[grid_cache["grid_id"] == highlight_id].copy()
                if not h.empty:
                    h["coordinates"] = h.geometry.apply(to_polygon_coords)
                    layers.append(pdk.Layer(
                        "PolygonLayer",
                        data=h,
                        get_polygon="coordinates",
                        get_fill_color=[255, 0, 255, 50],
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=3,
                        pickable=False,
                    ))
    
    # 点レイヤ（粒度に関係なく表示）
    map_df = pred_df[["lat","lon","risk_score","risk_level","area"]].copy()
    # ベース色（見やすい固定RGB）
    BASE_COLORS = {
        "low":  (64, 160, 255),   # 青
        "med":  (255, 180, 64),   # オレンジ
        "high": (255, 64, 64),    # 赤
    }

    # ベクトル化で色・半径・アルファを作成
    is_hl = (map_df["area"] == highlight_id) & (highlight_id != "")
    map_df["fill_r"] = np.where(is_hl, 255, map_df["risk_level"].map(lambda lv: BASE_COLORS[lv][0]))
    map_df["fill_g"] = np.where(is_hl,   0, map_df["risk_level"].map(lambda lv: BASE_COLORS[lv][1]))
    map_df["fill_b"] = np.where(is_hl, 255, map_df["risk_level"].map(lambda lv: BASE_COLORS[lv][2]))
    map_df["fill_a"] = np.where(is_hl, 255, 180)  # 透過も強めに差をつける

    map_df["radius"] = np.where(
        is_hl,
        1800 if agg == "adm" else 1100,     # ← ハイライトはデカく
        1200 if agg == "adm" else 700,
    )

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=30,
        get_fill_color='[fill_r, fill_g, fill_b, fill_a]',  # ← RGBAで渡す
        pickable=True,
        auto_highlight=True,
    ))

except Exception as e:
    st.error("ポリゴン描画に失敗しました。詳細ログを下に表示します。")
    st.exception(e)

tooltip = {"text": "{area}\nscore: {risk_score}\nlevel: {risk_level}"}

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip          # ← ここを追加
    ),
    height=720
)

st.caption(f"ADM={len(ADM_AREAS)} areas, GRID={len(GRID_AREAS)} cells")

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
