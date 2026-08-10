
import matplotlib
matplotlib.use("Agg")  # 화면 없이 이미지로만 렌더링 (Streamlit 의존 제거)
import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from pymongo import MongoClient

matplotlib.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="Today", layout="wide")

MONGO_URL = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True)
col = client["Target"]["target"]
dbs = pd.DataFrame(col.find({}, {"_id": 0}))
MM = client["stock"]["stock"]
memo_docs = list(MM.find({}, {"_id": 0, "코드": 1, "Memo": 1}))
memo_map = {str(d["코드"]): d.get("Memo", "") for d in memo_docs}

result = []
for _, row in dbs.iterrows():
    item = row["종목"]          
    code = str(row["코드"])     
    ref = row["기준"]

    df = fdr.DataReader(code).tail(60).reset_index()

    CC = df['Close'].iloc[-1]
    CH = round(df['Change'].iloc[-1] * 100, 1)

    # 승률(RC) 계산 및 삼각형 표시
    RC = round((CC - ref) / ref * 100, 1)
    if RC > 0:
        rate = f"▲ {RC:.1f}"
    elif RC < 0:
        rate = f"▼ {abs(RC):.1f}"
    else:
        rate = f"{RC:.1f}"

    if CH > 0:
        ch_display = f"▲ {CH:.2f}"
    elif CH < 0:
        ch_display = f"▼ {abs(CH):.2f}"
    else:
        ch_display = f"{CH:.2f}"

    memo = memo_map.get(code, "")
    chart = f'<a href="https://m.stock.naver.com/fchart/domestic/stock/{code}" target="_blank">차트</a>'

    result.append({
        "종목": item,
        "코드": code,
        "현재": CC,
        "등락": ch_display,   # 문자열로 표시 (▲/▼ 포함)
        "CH_raw": CH,       # 색상 판정용 원본 숫자
        "기준": ref,
        "승률": rate,
        "RC": RC,
        "메모": memo,
        "Chart": chart
    })
result_df = pd.DataFrame(result)

def color_ch(row):
    color = "red" if row["CH_raw"] > 0 else ("blue" if row["CH_raw"] < 0 else "black")
    return [f"color: {color}" if col == "등락" else "" for col in row.index]

def color_rate(row):
    color = "red" if row["RC"] > 0 else ("blue" if row["RC"] < 0 else "black")
    return [f"color: {color}" if col == "승률" else "" for col in row.index]

display_cols = ["종목", "코드", "현재", "등락", "기준", "승률", "메모", "Chart"]
right_cols = ["기준", "현재"]

def highlight_hc(val):
    if val >= 50:
        return "background-color: #f8c8ec"
    return ""

def highlight_lc(val):
    if val >= 10:
        return "background-color: #f8c8ec"
    return ""

styled = (
    result_df.style
    .apply(color_rate, axis=1)
    .apply(color_ch, axis=1)     
    .format({
        "기준": "{:,.0f}",
        "현재": "{:,.0f}",
    })
    .set_properties(subset=right_cols, **{"text-align": "right"})
    .set_table_styles(
        [
            {"selector": "th",
             "props": [("background-color", "lightgray"),
                       ("text-align", "center")]},
        ]
        + [
            {"selector": f"th.col_heading.col{result_df.columns.get_loc(c)}",
             "props": [("text-align", "center")]}
            for c in right_cols
        ],
        overwrite=False
    )
    .hide(axis="index")
    .hide(axis="columns", subset=["RC", "CH_raw"])   
)

st.markdown(styled.to_html(escape=False), unsafe_allow_html=True)
