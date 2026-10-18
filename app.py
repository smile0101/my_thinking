import os
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_icon="♥", page_title="지수", layout="wide")
st.subheader("📊 지수") 

def load_data(code, T=60, N =1):
    try :
        day = (datetime.now() - timedelta(days=150)).strftime("%Y%m%d") #300
        dd = fdr.DataReader(code, day).reset_index()
        # dd = fdr.DataReader(code, '20260301', '20261013').reset_index()
        if 'index' in dd.columns:
            dd = dd.rename(columns={'index': 'Date'})
        if 'Change' in dd.columns:
            dd['Change'] = round(dd['Change'] * 100, 1)
        else:
            dd['Change'] = round(dd['Close'].pct_change() * 100, 1)

        dd = dd.ffill()

        for n in [5, 10, 20, 60, 120]:
            dd[f'MA{n}'] = dd['Close'].rolling(window=n).mean()
        dd['MA5_d'] = dd['MA5'].diff()
        dd['MA10_d'] = dd['MA10'].diff()
        dd['S5'] = np.degrees(np.arctan(np.gradient(dd['MA5'].values)))
        dd['S10'] = np.degrees(np.arctan(np.gradient(dd['MA10'].values)))
        dd['pos'] = np.where( dd['High'] == dd['Low'], 0,  round((dd['Close'] - dd['Low']) / (dd['High'] - dd['Low']) * 100, 1))
        dd['pos_ma5'] = dd['pos'].rolling(window=3).mean()

        end_idx = -(N - 1) if N > 1 else None
        start_idx = -(T + N - 1)
        dd['Date'] = pd.to_datetime(dd['Date']).dt.strftime('%m.%d')
        return dd.iloc[start_idx:end_idx].copy()
    except Exception:
        print("실패")
        return None

##############################################################################

def plot_trend_subplot(ax, df, close_prices, order_val, item):

    count_map = {1: 5, 2: 4, 3: 3}
    limit_count = count_map.get(order_val, 3)  # 기본값은 3개

    # 1. 저점 계산
    minima_indices = argrelextrema(close_prices, np.less, order=order_val)[0]
    if len(minima_indices) >= 2:
        low_indices = sorted(minima_indices, reverse=True)[:limit_count]
        low_indices.sort()
    else:
        fallback = [len(close_prices) - 20, len(close_prices) - 10, len(close_prices) - 1]
        low_indices = [idx for idx in fallback if 0 <= idx < len(close_prices)]

    x_low = np.array(low_indices)
    y_low = close_prices[x_low] if len(x_low) > 0 else np.array([])

    if len(x_low) >= 2:
        slope_low, intercept_low = np.polyfit(x_low, y_low, 1)
        trend_line_low = slope_low * np.arange(len(df)) + intercept_low
    else:
        trend_line_low = np.full(len(df), close_prices[-1])

    # 2. 고점(Maxima) 계산
    maxima_indices = argrelextrema(close_prices, np.greater, order=order_val)[0]
    if len(maxima_indices) >= 2:
        high_indices = sorted(maxima_indices, reverse=True)[:limit_count]
        high_indices.sort()
    else:
        fallback = [len(close_prices) - 20, len(close_prices) - 10, len(close_prices) - 1]
        high_indices = [idx for idx in fallback if 0 <= idx < len(close_prices)]

    x_high = np.array(high_indices)
    y_high = close_prices[x_high] if len(x_high) > 0 else np.array([])

    if len(x_high) >= 2:
        slope_high, intercept_high = np.polyfit(x_high, y_high, 1)
        trend_line_high = slope_high * np.arange(len(df)) + intercept_high
    else:
        trend_line_high = np.full(len(df), close_prices[-1])

    # 3. 변동률(Cha) 계산
    latest_high = y_high[-1] if len(y_high) > 0 else close_prices[-1]
    latest_low = y_low[-1] if len(y_low) > 0 else close_prices[0]
    cha_value = round((latest_high - latest_low) / latest_low * 100) if latest_low != 0 else 0

    # 4. 그래프 그리기
    x = df["Date_Str"]
    y = df["Close"]

    # 종가 선
    ax.plot(x, y, color="tab:blue", linewidth=1.5, marker="o", markersize=3)

    # 저점 추세선 및 마커
    ax.plot(x, trend_line_low, color="tab:red", linewidth=2, linestyle="--")
    if len(x_low) > 0:
        ax.scatter(x.iloc[x_low], y_low, color="tab:red", s=50, zorder=5)
        for idx in x_low:
            val = close_prices[idx]
            ax.text(
                x.iloc[idx], val, f" {val:,.0f}",
                color="tab:red", fontsize=8, weight="bold",
                verticalalignment="top", horizontalalignment="center"
            )

    # 고점 추세선 및 마커
    ax.plot(x, trend_line_high, color="tab:green", linewidth=2, linestyle="--")
    if len(x_high) > 0:
        ax.scatter(x.iloc[x_high], y_high, color="tab:green", s=50, zorder=5)
        for idx in x_high:
            val = close_prices[idx]
            ax.text(
                x.iloc[idx], val, f" {val:,.0f}",
                color="tab:green", fontsize=8, weight="bold",
                verticalalignment="bottom", horizontalalignment="center"
            )

    # 축 및 타이틀 설정
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_yticklabels([])

    sub_title = f"{order_val}.L: {latest_low:,.0f} | H : {latest_high:,.0f} | Cha: {cha_value}%"
    ax.set_title(sub_title, fontsize=10, pad=10)

def trend(item, code):
    df = fdr.DataReader(code).tail(50).reset_index()
    # df = fdr.DataReader(code, '20260601','20260815').reset_index()
    date_col = df.columns[0]
    df.rename(columns={date_col: "Date", "Close": "Close"}, inplace=True)

    formatted_dates = [f"{d.month}.{d.day}" for d in pd.to_datetime(df["Date"])]
    df["Date_Str"] = formatted_dates

    close_prices = df["Close"].values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    plot_trend_subplot(ax1, df, close_prices, order_val=1, item=item)
    plot_trend_subplot(ax2, df, close_prices, order_val=2, item=item)
    plot_trend_subplot(ax3, df, close_prices, order_val=3, item=item)

    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)

############################        등락률         ###########################################
def calc_period(df, rows):
  sub = df.tail(rows)
  return { 'Change': round(sub['Change'].sum(), 1), }

def fmt_cell(val, row):

  if val is None or (isinstance(val, float) and pd.isna(val)):
    return ''

  if row == 'Change':
    if val > 0:
      return (
          f'<span style="color:#d63031; font-weight:bold;">{val:+.1f}%</span>'
      )
    elif val < 0:
      return (
          f'<span style="color:#0984e3; font-weight:bold;">{val:.1f}%</span>'
      )
    else:
      return '<span>0.0%</span>'
  return str(val)
def build(df):
    periods = []
    n = len(df)
    for rows in [5, 10, 15, 25, 50]:
        if n >= rows:
            periods.append(calc_period(df, rows))
        else:
            periods.append(None) # 또는 빈 딕셔너리 {}
    return tuple(periods)

def format1(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:.1f}%</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):.1f}%</span>'
    else:
        return f'<span>0.0%</span>'

def format2(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:,}</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):,}</span>'
    else:
        return f'<span>0</span>'

def format3(val):
    if val > 100:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val}</span>'
    elif val < 100:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val)}</span>'
    else:
        return f'<span>100</span>'

####################################### 그래프 ##################################################################
def showV( item_code, d ):

    ## 이동평균선 교차점 계산
    def find_cross_points(df, col1, col2):
        cross_points = []
        for i in range(1, len(df)):
            if (df[col1].iloc[i] > df[col2].iloc[i] and df[col1].iloc[i-1] <= df[col2].iloc[i-1]) or \
            (df[col1].iloc[i] < df[col2].iloc[i] and df[col1].iloc[i-1] >= df[col2].iloc[i-1]):
                cross_points.append(i-1)
        return cross_points

    def extract_last_cross_data(df, cross_points, col1, col2):
        if cross_points:
            last_cross_index = cross_points[-1]
            last_cross_date = df['Date'].iloc[last_cross_index]
            last_cross_value = df[[col1, col2]].iloc[last_cross_index].mean()
            return last_cross_date, last_cross_value
        return None, None

    def find_extrema(values):
        peaks, _ = find_peaks(values)
        valleys, _ = find_peaks(-values)
        return peaks, valleys

    def extract_extrema_data(df, values, peaks, valleys):
        maxi = values.iloc[peaks]
        mini = values.iloc[valleys]
        max_dates = df['Date'].iloc[peaks]
        min_dates = df['Date'].iloc[valleys]
        return maxi, mini, max_dates, min_dates

    dates = d['Date'].values
    ## 3달(100일) 
    max_15 = d['High'].max()
    min_15 = d['Close'].min()
    min_15_idx = d['Close'].values.argmin()          # 최저값 위치(위치 기반 index)
    min_15_date = dates[min_15_idx]                 # 최저값 날짜(x좌표)

    values_day = d['Close']
    values_5day = d['MA5'].dropna()
    valid_ma5 = d['pos_ma5'].dropna()

    max_val = valid_ma5.max()
    max_idx = valid_ma5.idxmax()
    max_date = d.loc[max_idx, 'Date']

    min_val = valid_ma5.min()
    min_idx = valid_ma5.idxmin()
    min_date = d.loc[min_idx, 'Date']

    peaks_day, valleys_day = find_extrema(values_day)
    peaks_5day, valleys_5day = find_extrema(values_5day)

    maxi_day, mini_day, max_dates_day, min_dates_day = extract_extrema_data(d, values_day, peaks_day, valleys_day)
    maxi_5day, mini_5day, max_dates_5day, min_dates_5day = extract_extrema_data(d, values_5day, peaks_5day, valleys_5day)

    # 마지막 교차점
    cross_close_20_points = find_cross_points(d, 'Close', 'MA20')
    last_cross_close_20_date, last_cross_close_20_value = extract_last_cross_data(d, cross_close_20_points, 'Close', 'MA20')
    cross_close_60_points = find_cross_points(d, 'Close', 'MA60')
    last_cross_close_60_date, last_cross_close_60_value = extract_last_cross_data(d, cross_close_60_points, 'Close', 'MA60')
    cross_close_120_points = find_cross_points(d, 'Close', 'MA120')
    last_cross_close_120_date, last_cross_close_120_value = extract_last_cross_data(d, cross_close_120_points, 'Close', 'MA120')

    if d['Close'].iloc[-1] > d['MA5'].iloc[-1] :
        R1 = 'M5'
    else : 
        R1 = ""
    if d['Close'].iloc[-1] > d['MA10'].iloc[-1] :
        R2 = 'M10'
    else :
        R2 = ""
    plt.rc('font', family='Malgun Gothic')
    fig = plt.figure(figsize=(14,7)) #14, 7.5
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 0.21, 0.21, 0.21], hspace=0.01)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # x축 공유
    ax3 = fig.add_subplot(gs[2], sharex=ax1) # x축 공유
    ax4 = fig.add_subplot(gs[3], sharex=ax1) # x축 공유


    ax1.set_title(f"{item_code}",fontsize=8)
    ##################################################################
    ax1.plot(d['Date'], d['Close'], linewidth=1.4, marker="o",label='Close', color ="red")
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)
    ax1.axhline(round(d['Close'].mean(),1), color='blue', linestyle='--')
    ax1.plot(min_15_date, min_15, marker='d', color='magenta', markersize=20, zorder=6)
    ax1.axhline(y = max_15, color="#A9ED17", linestyle='--')  

    # 거래 변동률
    ax1_t = ax1.twinx()
    ax1_t.bar(d['Date'], d['Change'], alpha=0.5, width=0.4,)
    ax1_t.axhline(y=0,  color="gray", linestyle="-", linewidth=2, alpha=0.7)

    for i in range(-10, 0):
        ax1_t.text( d['Date'].iloc[i], d['Change'].iloc[i] + 0.1,str(d['Change'].iloc[i]), ha='center',
            va='bottom', fontsize=12, color='black')
    ax1_t.tick_params(axis='y', labelsize=6)
    for j in range(len(d)):
        ax1.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax1.tick_params(axis='x', rotation=45, labelsize=1)

    ax1.tick_params(axis='y', labelsize=10) # 6
    pos = ax1.get_position()
    ax1.set_position([0.06, pos.y0, 0.9, pos.height])

    ##################################################################
    ax2.plot(d['Date'], d['Close'], linestyle='--', color='pink')
    ax2.plot(d['Date'], d['MA5'], linestyle='-', color='green', label='MA5')
    # ax2.plot(d['Date'], d['MA10'], linestyle='-.', color='black', label='MA10')
    ax2.plot(d['Date'], d['MA20'], linestyle='-', color='magenta', label='MA20')
    ax2.plot(d['Date'], d['MA60'], linestyle='-', color='blue', label='MA60')
    ax2.plot(d['Date'], d['MA120'], linestyle='-', color='black', label='MA120')
    
    ax2.plot(min_dates_day, mini_day, "o", color='purple', markersize=5)
    ax2.plot(max_dates_day, maxi_day, "o", color='orange', markersize=5)
    ax2.plot(max_dates_5day, maxi_5day, "o", color='red', markersize=11)
    ax2.plot(min_dates_5day, mini_5day, "o", color='purple', markersize=12)
    if last_cross_close_20_date: ax2.plot(last_cross_close_20_date,last_cross_close_20_value,"d",color='magenta',markersize=20) ## Close외 20교차점
    if last_cross_close_60_date: ax2.plot(last_cross_close_60_date,last_cross_close_60_value,"d",color='blue',markersize=15)
    if last_cross_close_120_date: ax2.plot(last_cross_close_120_date,last_cross_close_120_value,"d",color='black',markersize=11)
    for j in range(len(d)):
        ax2.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    # ax2.legend( loc='upper left', fontsize=10, frameon=False )
    ax2.tick_params(axis='y',labelsize=6)
    pos = ax2.get_position()
    ax2.set_position([0.06, pos.y0, 0.9, pos.height])

    ##################################################################

    ax3.plot(d['Date'], d['pos'],  marker='o', color='blue', linestyle='-', linewidth=1.8, markersize=5, label='Position (%)')
    ax3.plot(d['Date'], d['pos_ma5'], color='lightblue', linestyle='-.', linewidth=1.5, label='Position MA5') 
    ax3.axhline(y=50, color="#ED1762", linestyle='-',linewidth=2, alpha=0.7, label='50% Midpoint')

    ax3.scatter(max_date, max_val, color='red', s=100, zorder=5)
    ax3.annotate(f'{max_val:.1f}', (max_date, max_val), textcoords="offset points",  xytext=(0, 10), 
                ha='center', fontsize=10, fontweight='bold', color='red')

    # 최저값 표시 (푸른색 포인트 및 텍스트)
    ax3.scatter(min_date, min_val, color='dodgerblue', s=100, zorder=5)
    ax3.annotate(f'{min_val:.1f}',  (min_date, min_val), textcoords="offset points", xytext=(0, -15), 
                ha='center', fontsize=10, fontweight='bold', color='darkblue')
    ax32 = ax3.twinx()
    ax32.bar(d['Date'], d['MA5_d'], color=np.where(d['MA5_d']>=0,'royalblue','salmon'), alpha=0.5, width=0.4,)
    ax32.axhline(y=0,  color="gray", linestyle="-", linewidth=2, alpha=0.7)

    for j in range(len(d)):
        ax3.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax3.tick_params(axis='y', labelsize=6)
    ax32.tick_params(axis='y', labelsize=6)

    ##################################################################
    d['S5_detail'] = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    
    ax4.plot(d['Date'], d['MA5_d'], label='MA5', color='green', linestyle='-', linewidth = 1.5)
    ax4.legend( loc='upper left', fontsize=12, frameon=False )
    ax4.axhline(y=0 , color="gray", linestyle=":", linewidth=2, alpha=0.7)
    for j in range(len(d)):
        ax4.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax4.tick_params(axis='y', labelsize=6)
    ax4.tick_params(axis='x', rotation=45)
    for label in ax4.get_xticklabels():
        label.set_fontsize(8)  ## X좌표 크기

    # 보조축 설정
    ax5 = ax4.twinx()
    ax5.plot(d['Date'], d['S5_detail'], label='S5', color='magenta', linestyle='-.', linewidth=2)
    ax5.set_ylim(89.68, 90.03)
    ax5.set_yticks(np.arange(89.68, 90.03, 0.05))
    ax5.tick_params(axis='y', labelsize=6)

    # 또 다른 보조축 (종가 표시용)
    ax6 = ax4.twinx()
    ax6.plot(d['Date'], d['Close'], label='종가', linestyle='-.', color='black', linewidth=1, alpha=0.5)
    ax6.tick_params(axis='y', labelsize=6)

    # --- 전체 레이아웃 정렬 ---
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, left=0.05, right=0.95, top=0.95)

    st.pyplot(fig)
    plt.close(fig)

################################################################################################
stock_list = [
    {   "item": "코스피",
        "code": "^KS11",
        "url":  'https://t1.daumcdn.net/media/finance/chart/kr/stock/d/KGG01P.png?',
        "Tr" : 'KOSPI'},

    {
        "item": "코스닥",
        "code": "^KQ11",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/stock/d/QGG01P.png?timestamp=202603021557",
        "Tr" : 'KOSDAQ'},
    {
        "item": "다우지수",
        "code": "DJI",
        "url": "https://ssl.pstatic.net/imgfinance/chart/world/continent/DJI@DJI.png",
        "Tr" : 'DJI'},

    {
        "item": "나스닥",
        "code": "IXIC",
        "url": "https://ssl.pstatic.net/imgfinance/chart/world/continent/NAS@IXIC.png",
        "Tr" : 'IXIC'},

    {
        "item": "S&P",
        "code": "US500",
        "url": "https://t1.daumcdn.net/media/finance/chart/us/daumstock-mini/d/SP500.png",
        "Tr" : 'SPX'},  

    {
        "item": "달러",
        "code": "USD/KRW",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/daumforex/d/KRWUSD.png",
        "Tr" : 'USDKRW'},  
   {
        "item": "엔화",
        "code": "JPY/KRW",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/daumforex/d/KRWJPY.png", 
        "Tr" : 'JPYKRW'},
    {
        "item": "WTI",
        "code": "CL=F",
        "url": "https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/OIL_CL.png", 
        "Tr" : 'WTI'},

   {
        "item": "G0ld",
        "code": "GC=F",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/commodity-mini/m/GOLD.png", 
        "Tr" : 'GOLD '},

    {
        "item": "Silver",
        "code": "SI=F",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/commodity-mini/m/SI.png",
          "Tr" : 'SILVER '},
   {
        "item": "일본증시",
        "code": "N225",
        "url": "https://t1.daumcdn.net/media/finance/chart/jp/daumstock-mini/d/NI225.png", 
        "Tr" : 'NI225 '},]



for info in stock_list:
    item_name = info["item"]
    item_code = info["code"]
    item_url = info["url"]
    Tr = info["Tr"]

    # btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:15px;margin-left:20px;"
    btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:20px;margin:2px 20px 2p?x 0;"
    url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol={Tr}' 
    st.markdown(
    f'### {item_name} &emsp;&emsp; <a href="{url_tr}" target="_blank" style="{btn}">Tr</a>', 
    unsafe_allow_html=True)

    dfv = load_data(item_code, T=60, N=1)
    dfc = dfv.tail(15).copy()

    CC = dfv['Close'].iloc[-1]
    YY = dfv["Close"].iloc[-2]
    D5 = dfv["MA5"].iloc[-1]
    D20 = dfv["MA20"].iloc[-1]
    D60 = dfv["MA60"].iloc[-1]
    CD5 = int(D5/CC*100)
    CD20 = int(D20/CC*100)
    CD60 = int(D60/CC*100)
    CY = int(CC-YY)
    TRR = (CC-YY)/YY*100

    w1, w2, w3, m1, m2 = build(dfv)
    keys = ["1W", "2W", "3W", "1M", "2M"]
    values = [w1, w2, w3, m1, m2]

    all_ch = ", ".join( f"{key}: {fmt_cell(v['Change'], 'Change')}" for key, v in zip(keys, values) if v is not None )


    if dfv is not None and not dfv.empty:

        col_left, col_right = st.columns([1, 1.5], gap="medium")
        
        # 좌측: 이미지 (높이를 350px로 고정하고 박스 안에 맞춤)
        with col_left:
 
            st.markdown( f""" <div style="width: 100%;">  <img src="{item_url}" style="width: 600px; height: 150px; object-fit: fill;" /> </div> """,
                unsafe_allow_html=True )
            st.markdown( f'<div style="font-size: 18px;">&emsp;&emsp; {all_ch} &emsp;&emsp;현재:{CC:,.0f}</div>', unsafe_allow_html=True, )
            ch_combined = f"{format3(CD5)} / {format3(CD20)} / {format3(CD60)}  &emsp;&emsp;{format2(CY)} / {format1(TRR)} "
            st.markdown( f'<div style="font-size: 18px;"> &emsp;&emsp; {ch_combined}</div>', unsafe_allow_html=True, )
  
        with col_right:
            trend(item_name, item_code)   
        
        # 2. Trend 그래프 렌더링
        
        showV(item_code, dfc)

        st.markdown("---")
    else:
        st.warning(f"[{item_name}] ({item_code}) 데이터 실패")



######################################################################################################

keys = {

    '투자자(코스피)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSPI.png?sid=1697448197552',
    '투자자(코스닥)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSDAQ.png?sid=1697448286377',
    '증시자금' : 'https://ssl.pstatic.net/imgfinance/chart/sise/deposit_customer_deposit.png',
    'BTC(1일)' : 'https://imagechart.upbit.com/d/mini/BTC.png',
    '구리' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/CMDT_CDY.png',  
    '천연가스' : 'https://t1.daumcdn.net/media/finance/chart/kr/commodity-mini/m/NG.png',
    '미국채30년' : 'https://t1.daumcdn.net/media/finance/chart/kr/bond-mini/m/US30YT.png',
    '미국채10년' : 'https://t1.daumcdn.net/media/finance/chart/kr/bond-mini/m/US10YT.png',
    '국채 3년' : 'https://t1.daumcdn.net/media/finance/chart/kr/bond-mini/m/KRTSY3Y.png',
    '콜금리' : 'https://t1.daumcdn.net/media/finance/chart/kr/bond-mini/m3/KRCALL.png',
    '상해증시' : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/SHS@000001.png',
    '인도증시'  : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/INI@BSE30.png'}

items = list(keys.items()) # (이름, URL) 튜플 리스트로 변환
cols_per_row = 4
for i in range(0, len(items), cols_per_row):
    row_items = items[i : i + cols_per_row]
    cols = st.columns(cols_per_row)
    
    for idx, (name, url) in enumerate(row_items):
        with cols[idx]: 
            st.caption(f"**{name}**") # 이미지 위에 제목 표시
            st.image(url, width='stretch') #`width='content'



