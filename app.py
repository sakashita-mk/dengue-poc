import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="Infectious Disease × Satellite PoC", layout="wide")
st.title("衛星データ × 感染症対策 PoC（デモUI）")

left, right = st.columns([1, 2])
with left:
    st.subheader("条件設定")
    cities = ["Manila", "Jakarta", "Bangkok", "Ho Chi Minh City"]
    target_city = st.selectbox("対象都市", cities, index=cities.index("Manila"))
    metrics = st.multiselect(
        "分析指標（複数選択可）",
        ["降雨量", "気温", "湿度", "土地被覆（緑地/水域）", "ヒートリスク", "人口密度（参考）"],
        default=["降雨量", "気温"]
    )
    rain_thr = st.slider("降雨量の閾値（mm/day）", 0, 200, 50, step=5)
    temp_thr = st.slider("気温の閾値（°C）", 0, 45, 30, step=1)
    today = date.today()
    start = st.date_input("開始日", today - timedelta(days=14))
    end = st.date_input("終了日", today)
    if start > end:
        st.error("開始日は終了日よりも前にしてください。")
    run = st.button("実行")

with right:
    st.subheader(f"結果ビュー：{target_city}")
    if run:
        days = pd.date_range(start=start, end=end, freq="D")
        df = pd.DataFrame({
            "date": days,
            "rain_mm": np.random.gamma(shape=2.0, scale=20.0, size=len(days)),
            "temp_c": np.random.normal(loc=30, scale=3, size=len(days)),
        })
        df["risk_score"] = (df["rain_mm"] > rain_thr).astype(int) + (df["temp_c"] > temp_thr).astype(int)
        st.caption("ダミーデータ（本実装ではAPI/衛星データへ差し替え）")
        st.line_chart(df.set_index("date")[["rain_mm", "temp_c"]])
        st.bar_chart(df.set_index("date")[["risk_score"]])
        st.dataframe(df)
        st.info(
            "実装ポイント例：\n"
            "- GEE/EO APIから取得\n- 都市境界で集計\n- 閾値はスライダー連動\n- 介入（例：薬剤散布）推奨ロジックに接続"
        )
    else:
        st.write("左の条件を設定して **実行** を押してください。")
