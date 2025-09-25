# app_safe.py ーー まず確実に起動する版
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import box

st.set_page_config(page_title="NCR Dengue — Safe Mode", layout="wide")
st.title("NCR Dengue — PoC Demo (Safe Mode)")
st.caption("外部DLなし・軽量ジオメトリのみで動作確認")

CENTER_LAT, CENTER_LON = 14.5995, 120.9842

def simplify_gdf(gdf: gpd.GeoDataFrame, tol_m: float = 50) -> gpd.GeoDataFrame:
    g = gdf.to_crs(32651).copy()
    g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
    g = g.to_crs(4326)
    g["geometry"] = g.buffer(0)
    return g

def to_polygon_coords(g):
    if g.is_empty: return []
    if g.geom_type == "Polygon":
        return [list(map(list, g.exterior.coords))]
    elif g.geom_type == "MultiPolygon":
        return [list(map(list, poly.exterior.coords)) for poly in g.geoms]
    return []

def risk_to_rgb(score: float):
    s = max(0.0, min(100.0, float(score))) / 100.0
    stops = [(0.0,(49,130,189)), (0.5,(255,237,160)), (1.0,(220,20,60))]
    for (p0,c0),(p1,c1) in zip(stops, stops[1:]):
        if s <= p1:
            t = 0.0 if p1==p0 else (s-p0)/(p1-p0)
            return (
                int(c0[0]+t*(c1[0]-c0[0])),
                int(c0[1]+t*(c1[1]-c0[1])),
                int(c0[2]+t*(c1[2]-c0[2])),
            )
    return stops[-1][1]

@st.cache_data(show_spinner=False)
def make_ncr_bbox():
    # NCR中心から約40km四方のBBox（簡易）
    lat, lon = CENTER_LAT, CENTER_LON
    dy, dx = 0.36, 0.45
    g = gpd.GeoDataFrame(geometry=[box(lon-dx, lat-dy, lon+dx, lat+dy)], crs=4326)
    return simplify_gdf(g, 50)

@st.cache_data(show_spinner=False)
def make_grid_over_poly(poly_gdf):
    g = poly_gdf.to_crs(32651).copy()
    minx, miny, maxx, maxy = g.total_bounds
    cell = 1000
    union = g.unary_union
    cells = []
    import numpy as np
    for x in np.arange(minx, maxx, cell):
        for y in np.arange(miny, maxy, cell):
            geom = box(x, y, x+cell, y+cell)
            if union.intersects(geom):
                cells.append(geom)
    grid = gpd.GeoDataFrame(geometry=cells, crs=32651).to_crs(4326)
    grid = simplify_gdf(grid, 50)
    grid = grid.buffer(0)
    grid = gpd.GeoDataFrame(geometry=grid, crs=4326).explode(index_parts=False).reset_index(drop=True)
    grid["grid_id"] = [f"GRID-{i+1:03d}" for i in range(len(grid))]
    grid["coordinates"] = grid.geometry.apply(to_polygon_coords)
    # 代表点
    pts = grid.geometry.representative_point()
    grid["lat"] = [p.y for p in pts]
    grid["lon"] = [p.x for p in pts]
    return grid

ncr_bbox = make_ncr_bbox()
grid = make_grid_over_poly(ncr_bbox)

weeks = pd.date_range(date.today() - timedelta(weeks=77), periods=78, freq="W-MON")

with st.sidebar:
    st.header("Controls")
    base_date = st.date_input("基準日 (Week base)", value=weeks[-1].date())
    horizon = st.select_slider("予測ホライズン (weeks)", [0,1,2], value=2)
    granularity = st.radio("粒度 / 表示レイヤー", ["1kmグリッド"], index=0)
    show_choropleth = st.checkbox("ヒートマップ表示（面を色塗り）", value=True)
    hide_polygons = st.checkbox("ポリゴンを非表示にする", value=False)
    sel = st.multiselect("対象エリア", options=grid["grid_id"].tolist(), default=grid["grid_id"].tolist()[:6])
    hit = st.slider("Hit率目標", 0.0, 0.9, 0.6, 0.05)
    fa  = st.slider("過警報許容", 0.0, 0.5, 0.3, 0.05)
    mae = st.slider("MAE改善目標", 0.0, 0.4, 0.2, 0.05)

def predict_stub(areas, base_day: date, horizon_wk: int):
    week_num = int(datetime.combine(base_day, datetime.min.time()).strftime("%U"))
    season = (np.sin(2*np.pi*week_num/52.0) + 1) / 2
    out = []
    for a in areas:
        rng = np.random.default_rng(abs(hash(a)) % (2**32))
        base = 40 + 40*season
        score = float(np.clip(base + rng.normal(0,3) + 10*horizon_wk, 0, 100))
        level = "low" if score < 33 else ("med" if score < 66 else "high")
        r = {"area": a, "risk_score": round(score,1), "risk_level": level,
             "drivers": "demo", "horizon_wk": horizon_wk, "base_week": base_day.isoformat()}
        out.append(r)
    return pd.DataFrame(out)

if not sel:
    st.warning("対象エリアを1つ以上選んでください。")
    st.stop()

pred_df = predict_stub(sel, base_date, horizon)
pred_map = predict_stub(grid["grid_id"].tolist(), base_date, horizon)[["area","risk_score"]]
grid_vis = grid.merge(pred_map.rename(columns={"area":"grid_id"}), on="grid_id", how="left")
rgb = grid_vis["risk_score"].fillna(0).apply(risk_to_rgb)
grid_vis[["r","g","b"]] = pd.DataFrame(rgb.tolist(), index=grid_vis.index)
grid_vis["a"] = 160

st.subheader("Predictions")
left, right = st.columns([4,1])
with left:
    st.dataframe(pred_df[["area","risk_score","risk_level","drivers","horizon_wk","base_week"]],
                 use_container_width=True, hide_index=True)
with right:
    highlight = st.selectbox("ハイライト", options=["(なし)"] + pred_df["area"].tolist())
    highlight_id = "" if highlight == "(なし)" else highlight

st.subheader("Risk Map")
layers = []
if not hide_polygons:
    layers.append(pdk.Layer(
        "PolygonLayer",
        data=grid_vis,
        get_polygon="coordinates",
        get_fill_color='[r,g,b,a]' if show_choropleth else [0,0,0,0],
        stroked=True,
        get_line_color=[255,255,255,60],
        line_width_min_pixels=1,
        pickable=True,
    ))
if highlight_id:
    h = grid[grid["grid_id"] == highlight_id].copy()
    if not h.empty:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=h,
            get_polygon="coordinates",
            get_fill_color=[255,0,255,50],
            stroked=True,
            get_line_color=[255,255,255],
            line_width_min_pixels=3,
            pickable=False
        ))

view_state = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LON, zoom=10)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state), height=720)

with st.sidebar:
    st.markdown("---")
    st.metric("Hit率 目標", f"{int(hit*100)}%")
    st.metric("過警報 許容", f"{int(fa*100)}%")
    st.metric("MAE改善 目標", f"{int(mae*100)}%")

st.success("Safe Modeで稼働中。外部DLを戻す前に、まずはこれでデプロイ確認を。")
