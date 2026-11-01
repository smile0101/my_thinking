from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

st.set_page_config(page_title="메인", layout="wide")

def load_data(code, T=60, N=1):
    try:
        day = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
        dd = fdr.DataReader(code, day).reset_index()
        
        if 'index' in dd.columns:
            dd = dd.rename(columns={'index': 'Date'})
        if 'Date' not in dd.columns and 'date' in dd.columns:
            dd = dd.rename(columns={'date': 'Date'})
            
        if 'Change' in dd.columns:
            dd['Change'] = round(dd['Change'] * 100, 2)
        else:
            dd['Change'] = round(dd['Close'].pct_change() * 100, 2)
            
        for n in [5, 10, 20, 60, 120]:
            dd[f'MA{n}'] = dd['Close'].rolling(window=n).mean()
            
        dd['MA5_d'] = dd['MA5'].diff()
        dd['MA10_d'] = dd['MA10'].diff()
        dd['S5'] = np.degrees(np.arctan(np.gradient(dd['MA5'].values)))
        dd['S10'] = np.degrees(np.arctan(np.gradient(dd['MA10'].values)))
        
        end_idx = -(N - 1) if N > 1 else None
        start_idx = -(T + N - 1)
        dd['Date'] = pd.to_datetime(dd['Date']).dt.strftime('%m.%d')
        
        return dd.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    except Exception as e:
        st.error(f"데이터 로드 실패 ({code}): {e}")
        return None

def showV(item, d, T=60):
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
    max_100 = d['Close'].max()
    min_100 = d['Close'].min()
    min_100_idx = d['Close'].values.argmin()
    min_100_date = dates[min_100_idx]

    d5 = d.tail(5)
    CC = d5['Close'].iloc[-1]
    max_5, min_5 = d5['Close'].max(), d5['Close'].min()
    gap_up_5 = (max_5 - CC) / CC * 100
    gap_dn_5 = (CC - min_5) / CC * 100

    values_day = d['Close']
    values_5day = d['MA5'].dropna()

    peaks_day, valleys_day = find_extrema(values_day)
    peaks_5day, valleys_5day = find_extrema(values_5day.values)

    maxi_day, mini_day, max_dates_day, min_dates_day = extract_extrema_data(d, values_day, peaks_day, valleys_day)
    maxi_5day, mini_5day, max_dates_5day, min_dates_5day = extract_extrema_data(d, values_5day, peaks_5day, valleys_5day)

    cross_close_20_points = find_cross_points(d, 'Close', 'MA20')
    last_cross_close_20_date, last_cross_close_20_value = extract_last_cross_data(d, cross_close_20_points, 'Close', 'MA20')
    cross_close_60_points = find_cross_points(d, 'Close', 'MA60')
    last_cross_close_60_date, last_cross_close_60_value = extract_last_cross_data(d, cross_close_60_points, 'Close', 'MA60')
    cross_close_120_points = find_cross_points(d, 'Close', 'MA120')
    last_cross_close_120_date, last_cross_close_120_value = extract_last_cross_data(d, cross_close_120_points, 'Close', 'MA120')

    plt.rc('font', family='Malgun Gothic')
    fig = plt.figure(figsize=(18.5, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 0.21, 0.21, 0.21], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # 그래프 1
    ax1.set_title(f"{item}", fontsize=14, fontweight='bold')
    ax1.plot(d['Date'], d['Close'], linewidth=1.4, label='Close')
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)

    ax1.axhline(max_100, linestyle=':', color='black', linewidth=1.0)
    ax1.axhline(min_100, linestyle=':', color='black', linewidth=1.0)
    ax1.plot(min_100_date, min_100, marker='d', color='magenta', markersize=15, zorder=6)

    d_len = len(d)
    periods_config = [
        {'days': 20, 'offset': 19, 'color': 'black'},
        {'days': 40, 'offset': 39, 'color': 'blue'},
        {'days': 60, 'offset': 59, 'color': 'green'},
    ]
    
    for config in periods_config:
        if d_len >= config['days']:
            d_sub = d.tail(config['days'])
            p_max = d_sub['Close'].max()
            p_min = d_sub['Close'].min()
            gap_pct = (p_max - p_min) / p_min * 100
            x_start = d_sub['Date'].iloc[0]
            x_end = d_sub['Date'].iloc[-1]
            
            ax1.hlines(y=p_max, xmin=x_start, xmax=x_end, colors=config['color'], linestyles=':', linewidth=1.0)
            ax1.hlines(y=p_min, xmin=x_start, xmax=x_end, colors=config['color'], linestyles=':', linewidth=1.0)
            
            idx_pos = max(0, d_len - config['offset'])
            if idx_pos < d_len:
                x_pos = dates[idx_pos]
                ax1.annotate('', xy=(x_pos, p_max), xytext=(x_pos, p_min),
                             arrowprops=dict(arrowstyle='<->', linewidth=1.2, edgecolor=config['color']))
                ax1.text(x_pos, (p_max + p_min) / 2, f"{gap_pct:.0f}%",
                         ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec=config['color']))

    if d_len > 20:
        periods = {'1M': 20, '2M': 40, '3M': 60, '4M': 80}
        colors = ['#FF5733', '#33FF57', '#3357FF', "#EDF51A"]
        for i, (label, offset) in enumerate(periods.items()):
            if d_len > offset:
                idx = d_len - 1 - offset
                if idx >= 0:
                    target_date = d['Date'].iloc[idx]
                    target_price = d['Close'].iloc[idx]
                    ax1.plot(target_date, target_price, 'o', markersize=10,
                             markeredgecolor='black', markerfacecolor=colors[i], zorder=5)

    ax1.text(dates[0], max_100, f' Max {int(max_100):,}', fontsize=10, va='bottom')
    ax1.text(dates[0], min_100, f' Min {int(min_100):,}', fontsize=10, va='top')

    x5_start, x5_end = d5['Date'].iloc[0], d5['Date'].iloc[-1]
    ax1.hlines(y=max_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    ax1.hlines(y=min_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    
    x5_idx = max(0, d_len - 7)
    x5_text_pos = dates[x5_idx]
    ax1.annotate('', xy=(x5_text_pos, max_5), xytext=(x5_text_pos, CC), arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(x5_text_pos, (max_5 + CC)/2, f'+{gap_up_5:.1f}%', ha='left', fontsize=10, bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.8))
    ax1.annotate('', xy=(x5_text_pos, CC), xytext=(x5_text_pos, min_5), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax1.text(x5_text_pos, (CC + min_5)/2, f'-{gap_dn_5:.1f}%', ha='right', fontsize=10, bbox=dict(boxstyle='round', fc='lightcyan', alpha=0.8))
    ax1.text(d5['Date'].iloc[-1], max_5, f' {int(max_5):,}', color='red', fontsize=10, ha='left', va='center')
    ax1.text(d5['Date'].iloc[0], min_5, f' {int(min_5):,}', color='red', fontsize=10, ha='right', va='top')

    ax1_t = ax1.twinx()
    ax1_t.bar(d['Date'], d['Change'], alpha=0.25)
    for i in [-3, -2, -1]:
        if abs(i) <= len(d):
            ax1_t.text(d['Date'].iloc[i], d['Change'].iloc[i] + 0.1, str(d['Change'].iloc[i]), ha='center', va='bottom', fontsize=9, color='black')
    ax1_t.tick_params(axis='y', labelsize=8)
    
    for j in range(len(d)):
        ax1.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=0.8)
    ax1.tick_params(axis='y', labelsize=10)

    # 그래프 2
    ax2.plot(d['Date'], d['Close'], linestyle='--', color='pink')
    ax2.plot(d['Date'], d['MA5'], linestyle='-.', color='green', label='MA5')
    ax2.plot(d['Date'], d['MA10'], linestyle='-.', color='black', label='MA10')
    ax2.plot(d['Date'], d['MA20'], linestyle='-', color='magenta', label='MA20')
    ax2.plot(d['Date'], d['MA60'], linestyle='-', color='blue', label='MA60')
    ax2.plot(d['Date'], d['MA120'], linestyle='-', color='black', label='MA120')
    ax2.axhline(round(d['Close'].mean(), 1), color='orange', linestyle='--')
    
    ax2.plot(min_dates_day, mini_day, "o", color='purple', markersize=5)
    ax2.plot(max_dates_day, maxi_day, "o", color='orange', markersize=5)
    ax2.plot(max_dates_5day, maxi_5day, "o", color='red', markersize=8)
    ax2.plot(min_dates_5day, mini_5day, "o", color='purple', markersize=8)
    
    if last_cross_close_20_date: ax2.plot(last_cross_close_20_date, last_cross_close_20_value, "d", color='magenta', markersize=12)
    if last_cross_close_60_date: ax2.plot(last_cross_close_60_date, last_cross_close_60_value, "d", color='blue', markersize=10)
    if last_cross_close_120_date: ax2.plot(last_cross_close_120_date, last_cross_close_120_value, "d", color='black', markersize=8)
    
    for j in range(len(d)):
        ax2.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=0.8)
    ax2.tick_params(axis='y', labelsize=8)

    # 그래프 3
    ax3.plot(d['Date'], d['MA5'], label='MA5', color='red', linewidth=1.5)
    ax3.plot(d['Date'], d['MA10'], label='MA10', color='blue', linewidth=1.3)
    ax32 = ax3.twinx()
    ax32.bar(d['Date'], d['MA5_d'], color=np.where(d['MA5_d'] >= 0, 'royalblue', 'salmon'), alpha=0.5)
    ax32.axhline(y=0, color='green', linestyle='--', linewidth=1.5)
    
    for j in range(len(d)):
        ax3.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=0.8)
    ax3.tick_params(axis='y', labelsize=8)
    ax32.tick_params(axis='y', labelsize=8)

    # 그래프 4
    d['S5_detail'] = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    ax4.plot(d['Date'], d['MA5_d'], label='MA5변화', color='green', linestyle='-', alpha=0.5)
    ax4.legend(loc='upper left', fontsize=10, frameon=False)
    ax4.axhline(y=0, color='orange', linestyle='--', linewidth=1)
    
    for j in range(len(d)):
        ax4.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=0.8)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)

    ax5 = ax4.twinx()
    ax5.plot(d['Date'], d['S5_detail'], label='S5', color='magenta', linestyle='-.', linewidth=1.5)
    ax5.plot(d['Date'], d['S10_detail'], label='S10', linestyle='--', color='blue', linewidth=1)
    ax5.set_ylim(89.68, 90.03)
    ax5.set_yticks(np.arange(89.68, 90.03, 0.05))
    ax5.tick_params(axis='y', labelsize=8)

    ax6 = ax4.twinx()
    ax6.plot(d['Date'], d['Close'], label='종가', linestyle='-', color='black', linewidth=1.5, alpha=0.6)
    ax6.tick_params(axis='y', labelsize=8)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    return fig

T_list = [
    {"item": "코스피 선물", "url": "https://kred.dev/ko/kospi-200-night-futures"},
    {"item": "코스피", "url": "https://m.stock.naver.com/domestic/index/KOSPI/total", "code": "^KS11", "GG" : "KOSPI" },
    {"item": "코스닥", "url": "https://m.stock.naver.com/domestic/index/KOSDAQ/total", "code": "^KQ11", "GG" : "KOSDAQ" },
    {"item": "매매동향", "url": "https://stock.naver.com/market/stock/kr/trend/trader"},
    {"item": "자금동향", "url": "https://stock.naver.com/market/stock/kr/deposit"},
    {"item": "Think", "url": "https://www.thinkpool.com/"},
    {"item": "코스피맵", "url": "https://markets.hankyung.com/marketmap/kospi"},
    {"item": "코스닥맵", "url": "https://markets.hankyung.com/marketmap/kosdaq"},

    {"item": "S&P맵", "url": "https://finviz.com/map.ashx"},
    {"item": "S&P", "url": "https://m.stock.naver.com/worldstock/index/.INX/total", "code": "US500", "GG" : "SP500"},
    {"item": "다우지수", "url": "https://m.stock.naver.com/worldstock/index/.DJI/total", "code": "DJI", "GG" : "DJI"},
    {"item": "나스닥지수", "url": "https://m.stock.naver.com/worldstock/index/.IXIC/total", "code": "IXIC", "GG" : "IXIC"},

    {"item": "WTI", "url": "https://m.stock.naver.com/marketindex/energy/CLcv1?chartPeriod=day", "code": "CL=F", "GG" : "WTI"},
    {"item": "달러", "url": "https://stock.naver.com/marketindex/exchange/FX_USDKRW/price", "code": "USD/KRW", "GG" : "USDKRW"},
    {"item": "엔화", "url": "https://m.stock.naver.com/marketindex/exchange/FX_JPYKRW", "code": "JPY/KRW", "GG" : "JPYKRW"},
    {"item": "실시간", "url": "https://stock.naver.com/market/stock/kr/stocklist/priceTop"},
]

st.subheader("한국")
group1 = T_list[0:8]
html1 = ' '.join(f'<a href="{d["url"]}" target="_blank" style="text-decoration:none;"><b>🔗 {d["item"]}</b></a>' for d in group1)
st.markdown(f'<div style="display:flex; gap:25px; font-size:20px; flex-wrap:wrap; margin-bottom: 10px;">{html1}</div>', unsafe_allow_html=True)

st.subheader("미국")
group2 = T_list[8:12]
html2 = ' '.join(f'<a href="{d["url"]}" target="_blank" style="text-decoration:none;"><b>🔗 {d["item"]}</b></a>' for d in group2)
st.markdown(f'<div style="display:flex; gap:25px; font-size:20px; flex-wrap:wrap; margin-bottom: 10px;">{html2}</div>', unsafe_allow_html=True)

st.subheader("자산")
group3 = T_list[12:]
html3 = ' '.join(f'<a href="{d["url"]}" target="_blank" style="text-decoration:none;"><b>🔗 {d["item"]}</b></a>' for d in group3)
st.markdown(f'<div style="display:flex; gap:25px; font-size:20px; flex-wrap:wrap;">{html3}</div>', unsafe_allow_html=True)

##########################################################################################
if "selected_analysis" not in st.session_state:
    st.session_state.selected_analysis = None

st.subheader("그래프")
code_list = [d for d in T_list if "code" in d]
cols = st.columns(len(code_list))

for i, d in enumerate(code_list):
    with cols[i]:
        if st.button(f"📊 {d['item']}", key=f"btn_{d['code']}", use_container_width=True):
            st.session_state.selected_analysis = {"item": d["item"], "code": d["code"],"GG" : d['GG']}


if st.session_state.selected_analysis:
    selected = st.session_state.selected_analysis
    st.markdown("---")
    st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 25px; font-size: 20px; margin-bottom: 15px;">
        <a href="https://kr.tradingview.com/chart/Y3Tq45pg/?symbol={selected['GG']}" target="_blank"><b>📌 {selected['item']} </b></a>

    </div>
    """,  unsafe_allow_html=True)
    
    with st.spinner(f"{selected['item']} 데이터를 불러오는 중입니다..."):
        df = load_data(selected["code"])
        if df is not None and not df.empty:
            fig = showV(selected["item"], df)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error(f"{selected['item']} 데이터를 가져오지 못했습니다.")

#####################################################################################################

keys = {

    'BTC(1일)' : 'https://imagechart.upbit.com/d/mini/BTC.png',
    '구리' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month/CMDT_CDY.png',  
    '천연가스' : 'https://t1.daumcdn.net/media/finance/chart/kr/commodity-mini/m/NG.png',
    '미국채30년' : 'https://t1.daumcdn.net/media/finance/chart/kr/bond-mini/m/US30YT.png',
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
