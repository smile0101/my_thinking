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
df_portfolio  = pd.DataFrame(col.find({}, {"_id": 0}))

@st.cache_data(ttl=3600)
def get_recent(code):
    try:
        df = fdr.DataReader(str(code)).tail(5).reset_index()
        if 'Change' in df.columns:
            df['Change'] = round(df['Change'] * 100, 1)
        else:
            df['Change'] = 0.0
        return df
    except Exception as e:
        return None

# 3. 포맷팅 함수들
def format1(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:.1f}%</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):.1f}%</span>'
    else:
        return '<span>0.0%</span>'

def format(val):
    try:
        v = float(val)
        color = "red" if v > 0 else "blue"
        text = f"{v:,.0f}"
    except (TypeError, ValueError):
        color = "black"
        text = str(val)
    return f'<span style="color:{color}">{text}</span>'

# 스트림릿 화면 구성
st.subheader("📈 Today")

# 구분(구분별 그룹화)
groups = df_portfolio['구분'].unique()

for group in groups:
    st.markdown(f"#### 📌 {group}")
    group_df = df_portfolio[df_portfolio['구분'] == group]
    
    for _, row in group_df.iterrows():
        item = row.get('종목', '')
        code = row.get('코드', '')
        
        # 값이 비어있거나 NaN일 경우 체크
        buy = row.get('buy', 0)
        if pd.isna(buy):
            buy = 0
            
        vol = row.get('수량', 0)
        if pd.isna(vol):
            vol = 0
            
        MM = row.get('Memo', '')
        if pd.isna(MM):
            MM = ''
        
        df_recent = get_recent(code)
        
        if df_recent is not None and not df_recent.empty:
            CH = df_recent['Change'].iloc[-1]
            CC = df_recent['Close'].iloc[-1]
            CH_html = format1(CH) if not pd.isna(CH) else ""
            
            # buy나 vol이 없거나 0일 때 nan 대신 공백("") 또는 빈 값으로 처리
            if buy > 0 and vol > 0:
                Cha = CC - buy
                BC = Cha * vol / 10000
                INV = buy * vol / 10000
                RR = round((CC - buy) / buy * 100, 1)
                RD = format1(RR)
                BC_html = format(BC)
                buy_str = f"매수 : {buy:,.0f}원 &nbsp;&nbsp;|&nbsp;&nbsp;"
                inv_vol_str = f"[ {BC_html} : {INV:,.0f} / {vol:,.0f}주  ]"
                rd_str = f"수익률: {RD}"
            else:
                RD = ""
                BC_html = ""
                buy_str = ""
                inv_vol_str = ""
                rd_str = ""
            
            row_html = f"""
            <div class="header-line" style="line-height: 1.6;">
                <span style="font-size:24px;font-weight:bold;">ㅇ {item} : </span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                <span style="font-size:20px;font-weight:bold;">{CC:,.0f}원 ({CH_html})&nbsp;&nbsp;
                {buy_str}{rd_str} {inv_vol_str}</span>
            </div>
            <div style="margin: 8px 0;">
                &nbsp;&nbsp;&nbsp;<a href="https://webchart.thinkpool.com/2021ReNew/CS5MinBong/A{code}.png" target="_blank" style="padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:12px;margin:2px 10px 2px 0;display:inline-block;">Day</a>
                <a href="https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}" target="_blank" style="padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:12px;margin:2px 10px 2px 0;display:inline-block;">Tr</a>
                <a href="https://news.google.com/search?q={quote(str(item))}&hl=ko&gl=KR&ceid=KR:ko" target="_blank" style="padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:12px;margin:2px 10px 2px 0;display:inline-block;">Google</a>
                <span style="font-size:18px;font-weight:bold;">{MM}</span>
            </div>
            """

            st.markdown(row_html, unsafe_allow_html=True)

        else:
            st.markdown(f"**{item}** ({code}) - 데이터를 불러오지 못했습니다.", unsafe_allow_html=True)
            
    st.markdown("---")


