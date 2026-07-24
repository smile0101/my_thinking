import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
import os

# 페이지 설정
st.set_page_config(page_icon="♥", page_title="지수", layout="wide")
st.subheader("📊 지수") 

def set_korean_font():
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux(Streamlit Cloud)
        "C:/Windows/Fonts/malgun.ttf",                       # Windows 로컬
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            return
    # 못 찾으면 기본값 유지 (한글 깨짐 방지용 최소 조치)
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font() 


keys = {

    '코스피': 'https://t1.daumcdn.net/media/finance/chart/kr/stock/d/KGG01P.png?',
    '코스닥':'https://t1.daumcdn.net/media/finance/chart/kr/stock/d/QGG01P.png?timestamp=202603021557',
    '다우': 'https://ssl.pstatic.net/imgfinance/chart/world/continent/DJI@DJI.png', 
    '나스닥': 'https://ssl.pstatic.net/imgfinance/chart/world/continent/NAS@IXIC.png',
    '투자자(코스피)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSPI.png?sid=1697448197552',
    '투자자(코스닥)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSDAQ.png?sid=1697448286377',
    '증시자금' : 'https://ssl.pstatic.net/imgfinance/chart/sise/deposit_customer_deposit.png',
    'BTC(1일)' : 'https://imagechart.upbit.com/d/mini/BTC.png',
}

items = list(keys.items()) # (이름, URL) 튜플 리스트로 변환
cols_per_row = 4
for i in range(0, len(items), cols_per_row):
    row_items = items[i : i + cols_per_row]
    cols = st.columns(cols_per_row)
    
    for idx, (name, url) in enumerate(row_items):
        with cols[idx]: 
            st.caption(f"**{name}**") # 이미지 위에 제목 표시
            st.image(url, width='stretch') #`width='content'


##################################################################################################################

keys1 = {

    '환율(1개월)': 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/FX_USDKRW.png',
    '엔화(1개월)' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/FX_JPYKRW.png',
    'WTI(1개월)' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/OIL_CL.png',    
    '국내금' :'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/CMDT_GC.png',
    '구리' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/CMDT_CDY.png',
    '일본중시': 'https://ssl.pstatic.net/imgfinance/chart/world/month3/NII@NI225.png',
    '상해증시' : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/SHS@000001.png',
    '인도증시'  : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/INI@BSE30.png'}

items = list(keys1.items()) # (이름, URL) 튜플 리스트로 변환
cols_per_row = 4
for i in range(0, len(items), cols_per_row):
    row_items = items[i : i + cols_per_row]
    cols = st.columns(cols_per_row)
    
    for idx, (name, url) in enumerate(row_items):
        with cols[idx]: 
            st.caption(f"**{name}**") # 이미지 위에 제목 표시
            st.image(url, width='stretch') #`width='content'
