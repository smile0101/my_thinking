
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


col_widths = [1.8, 1.2, 2.5, 1.2, 1.0, 0.8]

def format_change_span(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:.1f}%</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):.1f}%</span>'
    else:
        return f'<span>0.0%</span>'

for _, row in dbs.iterrows():
    item = row["종목"]
    code = str(row["코드"]) if pd.notna(row["코드"]) else "000nan"
    ref = str(row["기준"]) if pd.notna(row["기준"]) else "0"

    # 코드가 "000nan"이거나 기준이 "0"인 경우 큰 글씨 타이틀로 출력
    if code == "000nan" or ref == "0":
        st.markdown(
            f'<div style="font-size: 24px; font-weight: bold; margin-top:'
            f' 20px; margin-bottom: 10px; color: #2c3e50;">📌 {item}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<hr style='margin: 4px 0px; border-top: 2px solid #2c3e50;'>",
            unsafe_allow_html=True,
        )
        continue

    # 조건에 해당하지 않을 때 숫자형 변환 및 차트 URL 구성
    ref_val = float(ref)
    link = f"https://m.stock.naver.com/fchart/domestic/stock/{code}"

    try:
        df = fdr.DataReader(code).tail(5).reset_index()

        if len(df) >= 4:
            CC = df["Close"].iloc[-1]
            CH1 = df["Change"].iloc[-1] * 100
            CH2 = df["Change"].iloc[-2] * 100
            CH3 = df["Change"].iloc[-3] * 100
            RC = round((CC - ref_val) / ref_val * 100, 1)

            row_cols = st.columns(col_widths)

            row_cols[0].markdown( f'<div style="font-size: 18px;' f' font-weight: bold;">{item}</div>',unsafe_allow_html=True,)
            row_cols[1].markdown( f'<div style="font-size: 18px; text-align:'f' right;">{CC:,.0f}원</div>', unsafe_allow_html=True, )

            ch_combined = f"{format_change_span(CH3)} / {format_change_span(CH2)} / {format_change_span(CH1)}"
            row_cols[2].markdown( f'<div style="font-size: 18px;">{ch_combined}</div>', unsafe_allow_html=True, )

            row_cols[3].markdown( f'<div style="font-size: 18px; text-align:'f' right;">{ref_val:,.0f}원</div>', unsafe_allow_html=True,)
 
            row_cols[4].markdown( f'<div style="font-size: 18px;">{format_change_span(RC)}</div>',unsafe_allow_html=True, )
            row_cols[5].markdown( f'<div style="font-size: 18px;"><a href="{link}"' ' target="_blank">Chart</a></div>',unsafe_allow_html=True,)

            st.markdown( "<hr style='margin: 6px 0px; border-color: #eee;'>", unsafe_allow_html=True,)

        else:
            st.error(f"{item}({code}): 거래 데이터가 부족합니다.")

    except Exception as e:
        st.error(f"{item}({code}) 데이터를 불러오는 중 오류 발생: {e}")

memo_rows = result_df[result_df["메모"].notna() & (result_df["메모"].astype(str).str.strip() != "")]
if not memo_rows.empty:
    st.markdown("---")
    for _, r in memo_rows.iterrows():
        st.markdown(f"**{r['종목']}** : {r['메모']}")
