import pandas as pd
import matplotlib
import FinanceDataReader as fdr
from urllib.parse import quote
import streamlit as st
from pymongo import MongoClient

matplotlib.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="Today", layout="wide")

MONGO_URL = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True)
col = client["Target"]["target"]
dfv = pd.DataFrame(col.find({}, {"_id": 0}))

MM = client["stock"]["stock"]
memo_docs = list(MM.find({}, {"_id": 0, "코드": 1, "Memo": 1}))
memo_map = {str(d["코드"]): d.get("Memo", "") for d in memo_docs if "코드" in d}

###############################################################################################
def format1(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:.1f}%</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):.1f}%</span>'
    else:
        return '<span>0.0%</span>'


def color_format(val):
    try:
        v = float(val)
        color = "red" if v > 0 else "blue"
        text = f"{v:,.1f}"
    except (TypeError, ValueError):
        color = "black"
        text = val
    return f'<span style="color:{color}">{text}</span>'


@st.cache_data(ttl=600)
def get_recent(code):
    df = fdr.DataReader(code).tail(5).reset_index()
    df['Change'] = round(df['Change'] * 100, 1)
    return df


st.subheader("📈 관심 종목 현황")
memo_list = []
for idx, row in dfv.iterrows():
    item, code, 구분, buy = row.종목, row.코드, row.구분, row.buy
    df = get_recent(code)
    sch = df['Change'].sum()
    SCHD = format1(sch)
    CC = df['Close'].iloc[-1]
    CD = f"{buy:,.0f}"
    ch = " / ".join(df["Change"].iloc[-5:].apply(color_format))
    RR = round((CC - buy) / buy * 100, 1)
    RD = format1(RR)
    memo_val = memo_map.get(code, "")
    if memo_val:
        memo_list.append({"종목": item, "메모": memo_val})

    st.markdown(f"##### 📌{item}_{구분}") 
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"https://ssl.pstatic.net/imgfinance/chart/item/area/week/{code}.png")  
        st.markdown( f' [{ch}]_&nbsp;&nbsp;{SCHD}', unsafe_allow_html=True)
        st.markdown(f'{CD}원, ({RD}) '
            f'<span style="margin-left:10px;"></span>'
            f'<a href="https://stock.naver.com/domestic/stock/{code}/price" target="_blank">네이버 </a>',  unsafe_allow_html=True )

    with col2:
        st.image(f"https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png")
        st.markdown(
            f'<a href="https://t1.daumcdn.net/media/finance/chart/kr/daumstock/d/A{code}.png" target="_blank" style="margin-right:18px;">일일</a>'
            f' / <a href="https://www.thinkpool.com/item/{code}" target="_blank" style="margin-right:18px;">Think</a>'
            f' / <a href="https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}" target="_blank" style="margin-right:18px;">Tr</a>'
            f' / <a href="https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko" target="_blank">{item} 뉴스</a>',
            unsafe_allow_html=True  )

if memo_list:
    st.markdown("---")
    for r in memo_list:
        st.markdown(f"**{r['종목']}** : {r['메모']}")

