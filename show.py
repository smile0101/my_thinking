import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from urllib.parse import quote

# 1. 페이지 설정
st.set_page_config(page_title="오늘", layout="wide")

t = """
[삼성01]				
S1	RISE코리아밸류업위클리고정커버드콜	0094M0	1132	 15,187 
S1	SK텔레콤	017670	130	 87,950 
S1	한화오션	042660	100	 88,739 
S1	SK이노베이션	096770	100	 159,226 
S1	SK스퀘어	402340	46	 782,829 
S1	삼성블룸버그인버스2XWTI원유선물ETN	530134	24000	 1,676 
				
				
[키움W]				
KW	KODEX코스닥150레버리지	233740	1300	 8,243 
KW	KODEXWTI원유선물인버스(H)	271050	4000	 1,742 
KW	SK스퀘어	402340	20	 1,081,500 
KW	KODEX반도체레버리지	494310	100	 80,645 
KW	삼성블룸버그인버스2XWTI원유선물ETN	530134	53000	 1,673 
				
				
[ISA_14]				
ISA	TIGER미국S&P500레버리지(합성H)	225040	300	 61,033 
ISA	KODEX2차전지산업레버리지	462330	4000	 1,066 
ISA	KODEX200타겟위클리커버드콜	498400	4663	 19,006 
				
				
[DC_28_30%]				
DC	SOL팔란티어커버드콜OTM채권혼합	0040Y0	3682	 8,153 
DC	RISE코리아밸류업위클리고정커버드콜	0094M0	133	 17,510 
DC	TIGER미국S&P500	360750	789	 25,135 
DC	KODEX테슬라인컴프리미엄채권혼합액티브	475080	4192	 9,583 
DC	RISE200위클리커버드콜	475720	4105	 9,845 
DC	KODEX미국30년국채+12%프리미엄(합성H)	481060	1860	 8,952 
DC	KODEX미국AI테크TOP10+15%프리미엄	483280	2226	 11,167 
DC	KODEX미국나스닥100데일리커버드콜OTM	494300	2000	 9,957 
DC	PLUS미국배당증가성장주데일리커버드콜	494420	1250	 9,689 
				
				
				
[연금_15]				
P15	RISE200위클리커버드콜	475720	268	 12,631 
P15	KODEX미국AI테크TOP10+15%프리미엄	483280	3162	 12,090 
				
				
[IRP_29 : 30%]				
IRP	RISE200위클리커버드콜	475720	1390	 10,908 
SO	SK하이닉스	000660	23	 568,000 
[JP]				
JP	어반테스트	6857.T	100	 41,510.0 
JP	도쿄일렉	8035.T	400	 12,505.0 
"""

# 데이터 파싱 함수
def parse_portfolio_text(text):
    sections = {}
    current_section = None
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            current_section = line
            sections[current_section] = []
        elif current_section:
            parts = re.split(r'\t+', line)
            if len(parts) >= 5:
                tag = parts[0].strip()
                name = parts[1].strip()
                code = parts[2].strip()
                qty = int(parts[3].replace(',', '').strip())
                buy_price = float(parts[4].replace(',', '').strip())
                sections[current_section].append({
                    'tag': tag,
                    'name': name,
                    'code': code,
                    'qty': qty,
                    'buy_price': buy_price
                })
    return sections

portfolio_data = parse_portfolio_text(t)

st.subheader("📈 대시보드")

# 현재가 조회 함수 (캐시 적용)
@st.cache_data(ttl=300)
def get_current_price(code):
    try:
        df = fdr.DataReader(code)
        if not df.empty and 'Close' in df.columns:
            return float(df['Close'].iloc[-1])
    except Exception as e:
        pass
    return None

# 데이터 로드 및 분석 함수 정의
def load_data(code, T=60, N=1):
    try:
        day = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
        dd = fdr.DataReader(code, day).reset_index()
        if 'index' in dd.columns:
            dd = dd.rename(columns={'index': 'Date'})
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
        return dd.iloc[start_idx:end_idx].copy()
    except Exception:
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
    peaks_5day, valleys_5day = find_extrema(values_5day)

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
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 0.21, 0.21, 0.21], hspace=0.01)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    ax1.set_title(f"{item}")
    ax1.plot(d['Date'], d['Close'], linewidth=1.4, label='Close')
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)

    ax1.axhline(max_100, linestyle=':', color='black', linewidth=1.0)
    ax1.axhline(min_100, linestyle=':', color='black', linewidth=1.0)
    ax1.plot(min_100_date, min_100, marker='d', color='magenta', markersize=20, zorder=6)

    d_len = len(d)
    periods_config = [
        {'days': 20, 'text_idx': (T-19), 'color': 'black'},
        {'days': 40, 'text_idx': (T-39), 'color': 'blue'},
        {'days': 60, 'text_idx': (T-59), 'color': 'green'},
        {'days': 80, 'text_idx': (T-79), 'color': 'black'},
        {'days': 100, 'text_idx': 1, 'color': 'black'}
    ]
    if d_len > 20:
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
                try:
                    x_pos = dates[config['text_idx']]
                    ax1.annotate('', xy=(x_pos, p_max), xytext=(x_pos, p_min),
                                   arrowprops=dict(arrowstyle='<->', linewidth=1.2, edgecolor=config['color']))
                    ax1.text(x_pos, (p_max + p_min) / 2, f"{gap_pct:.0f}%",
                               ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', fc='white', ec=config['color']))
                except IndexError:
                    pass

    if d_len > 20:
        periods = {'1M': 20, '2M': 40, '3M': 60, '4M': 80}
        colors = ['#FF5733', '#33FF57', '#3357FF', "#EDF51A"]
        for i, (label, offset) in enumerate(periods.items()):
            if d_len > offset:
                idx = d_len - 1 - offset
                if idx >= 0:
                    target_date = d['Date'].iloc[idx]
                    target_price = d['Close'].iloc[idx]
                    ax1.plot(target_date, target_price, 'o', markersize=12,
                               markeredgecolor='black', markerfacecolor=colors[i], zorder=5)

    ax1.text(dates[0], max_100, f' Max {int(max_100):,}', fontsize=12, va='bottom')
    ax1.text(dates[0], min_100, f' Min {int(min_100):,}', fontsize=12, va='top')

    x5_start, x5_end = d5['Date'].iloc[0], d5['Date'].iloc[-1]
    ax1.hlines(y=max_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    ax1.hlines(y=min_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    x5_idx = min(T-7, len(dates)-1)
    x5_text_pos = dates[x5_idx]
    ax1.annotate('', xy=(x5_text_pos, max_5), xytext=(x5_text_pos, CC), arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(x5_text_pos, (max_5 + CC)/2, f'+{gap_up_5:.1f}%', ha='left', fontsize=12, bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.8))
    ax1.annotate('', xy=(x5_text_pos, CC), xytext=(x5_text_pos, min_5), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax1.text(x5_text_pos, (CC + min_5)/2, f'-{gap_dn_5:.1f}%', ha='right', fontsize=12, bbox=dict(boxstyle='round', fc='lightcyan', alpha=0.8))
    ax1.text(d5['Date'].iloc[4], max_5, f' {int(max_5):,}', color='red', fontsize=10, ha='left', va='center')
    ax1.text(d5['Date'].iloc[0], min_5, f' {int(min_5):,}', color='red', fontsize=10, ha='right', va='top')

    ax1_t = ax1.twinx()
    ax1_t.bar(d['Date'], d['Change'], alpha=0.25)
    for i in [-3, -2, -1]:
        ax1_t.text(d['Date'].iloc[i], d['Change'].iloc[i] + 0.1, str(d['Change'].iloc[i]), ha='center', va='bottom', fontsize=11, color='black')
    ax1_t.tick_params(axis='y', labelsize=6)
    for j in range(len(d)):
        ax1.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax1.tick_params(axis='x', rotation=45, labelsize=1)
    ax1.tick_params(axis='y', labelsize=10)
    pos = ax1.get_position()
    ax1.set_position([0.06, pos.y0, 0.9, pos.height])

    # 그래프2
    ax2.plot(d['Date'], d['Close'], linestyle='--', color='pink')
    ax2.plot(d['Date'], d['MA5'], linestyle='-.', color='green', label='MA5')
    ax2.plot(d['Date'], d['MA10'], linestyle='-.', color='black', label='MA10')
    ax2.plot(d['Date'], d['MA20'], linestyle='-', color='magenta', label='MA20')
    ax2.plot(d['Date'], d['MA60'], linestyle='-', color='blue', label='MA60')
    ax2.plot(d['Date'], d['MA120'], linestyle='-', color='black', label='MA120')
    ax2.axhline(round(d['Close'].mean(), 1), color='orange', linestyle='--')
    ax2.plot(min_dates_day, mini_day, "o", color='purple', markersize=5)
    ax2.plot(max_dates_day, maxi_day, "o", color='orange', markersize=5)
    ax2.plot(max_dates_5day, maxi_5day, "o", color='red', markersize=11)
    ax2.plot(min_dates_5day, mini_5day, "o", color='purple', markersize=12)
    if last_cross_close_20_date: ax2.plot(last_cross_close_20_date, last_cross_close_20_value, "d", color='magenta', markersize=20)
    if last_cross_close_60_date: ax2.plot(last_cross_close_60_date, last_cross_close_60_value, "d", color='blue', markersize=15)
    if last_cross_close_120_date: ax2.plot(last_cross_close_120_date, last_cross_close_120_value, "d", color='black', markersize=11)
    for j in range(len(d)):
        ax2.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax2.tick_params(axis='y', labelsize=6)
    pos = ax2.get_position()
    ax2.set_position([0.06, pos.y0, 0.9, pos.height])

    # 그래프3
    ax3.plot(d['Date'], d['MA5'], label='MA5', color='red', linewidth=1.5)
    ax3.plot(d['Date'], d['MA10'], label='MA10', color='blue', linewidth=1.3)
    ax32 = ax3.twinx()
    ax32.bar(d['Date'], d['MA5_d'], color=np.where(d['MA5_d'] >= 0, 'royalblue', 'salmon'), alpha=0.5)
    ax32.axhline(y=0, color='green', linestyle='--', linewidth=2)
    for j in range(len(d)):
        ax3.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax3.tick_params(axis='y', labelsize=6)
    ax32.tick_params(axis='y', labelsize=6)

    # 그래프4
    d['S5_detail'] = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    ax4.plot(d['Date'], d['MA5_d'], label='MA5변화', color='green', linestyle='-', alpha=0.5)
    ax4.legend(loc='upper left', fontsize=12, frameon=False)
    ax4.axhline(y=0, color='orange', linestyle='--', linewidth=1)
    for j in range(len(d)):
        ax4.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax4.tick_params(axis='x', rotation=45)
    for label in ax4.get_xticklabels():
        label.set_fontsize(12)

    ax5 = ax4.twinx()
    ax5.plot(d['Date'], d['S5_detail'], label='S5', color='magenta', linestyle='-.', linewidth=2)
    ax5.plot(d['Date'], d['S10_detail'], label='S10', linestyle='--', color='blue', linewidth=1)
    ax5.set_ylim(89.68, 90.03)
    ax5.set_yticks(np.arange(89.68, 90.03, 0.05))
    ax5.tick_params(axis='y', labelsize=6)

    ax6 = ax4.twinx()
    ax6.plot(d['Date'], d['Close'], label='종가', linestyle='-', color='black', linewidth=2, alpha=0.6)
    ax6.tick_params(axis='y', labelsize=6)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, left=0.05, right=0.95, top=0.95)
    return fig

# ─────────────────────────────────────────
# 세션 상태 초기화 (계좌별 분석 상태 딕셔너리로 관리)
# ─────────────────────────────────────────
if "selected_analysis_by_section" not in st.session_state:
    st.session_state.selected_analysis_by_section = {}

def toggle_analysis(section_name, item, code):
    current = st.session_state.selected_analysis_by_section.get(section_name)
    if current == {"item": item, "code": code}:
        st.session_state.selected_analysis_by_section[section_name] = None
    else:
        st.session_state.selected_analysis_by_section[section_name] = {"item": item, "code": code}

# ─────────────────────────────────────────
# 계좌별 섹션 생성 및 3열 레이아웃 배치
# ─────────────────────────────────────────
for section_name, items in portfolio_data.items():
    st.markdown(f"### {section_name}")
    
    rows = [items[i:i + 3] for i in range(0, len(items), 3)]
    
    for row_items in rows:
        cols = st.columns(3)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                name = item['name']
                code = item['code']
                buy_price = item['buy_price']
                
                current_price = get_current_price(code)
                
                # 링크 생성
                naver_stock_url = f"https://m.stock.naver.com/domestic/stock/{code}/total?domesticStockExchange=KRX"
                naver_news_url = f'https://search.naver.com/search.naver?where=news&query={quote(name)}'
                google_news_url = f"https://news.google.com/search?q={quote(name)}&hl=ko&gl=KR&ceid=KR:ko"
                
                if current_price is not None:
                    return_rate = round((current_price - buy_price) / buy_price * 100, 1)
                    color = "red" if return_rate > 0 else ("blue" if return_rate < 0 else "gray")
                    
                    st.markdown(
                        f"""
                        <div style="padding: 12px; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 5px; background-color: #f9f9f9;">
                            <a href="{naver_stock_url}" target="_blank" style="font-weight: bold; font-size: 1.05em; text-decoration: none; color: #1f77b4;">{name}</a><br>
                            <span style="font-size: 0.85em; color: gray;">매수가: {buy_price:,.0f}원 | 현재가: {current_price:,.0f}원</span><br>
                            수익률: <span style="color:{color}; font-weight:bold;">{return_rate:+.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="padding: 12px; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 5px; background-color: #f9f9f9;">
                            <a href="{naver_stock_url}" target="_blank" style="font-weight: bold; font-size: 1.05em; text-decoration: none; color: #1f77b4;">{name}</a><br>
                            <span style="font-size: 0.85em; color: gray;">매수가: {buy_price:,.0f}원 | 현재가: <span style="color:red;">조회 실패</span></span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                current_active = st.session_state.selected_analysis_by_section.get(section_name)
                is_active = (current_active == {"item": name, "code": code})
                btn_label = "📊 분석닫기" if is_active else "📊 분석"

                with st.container(horizontal=True):
                    if st.button(btn_label, key=f"btn_{code}_{name}_{section_name}"):
                        toggle_analysis(section_name, name, code)
                        st.rerun()
                        
                    st.markdown(f'<a href="{naver_news_url}" target="_blank" title="네이버 뉴스 검색" style="text-decoration: none; font-size: 1.1em; line-height: 2.5em; padding-left: 5px;">Naver</a>', unsafe_allow_html=True)
                    st.markdown(f'<a href="{google_news_url}" target="_blank" title="구글 뉴스 검색" style="text-decoration: none; font-size: 1.1em; line-height: 2.5em; padding-left: 5px;">Google</a>', unsafe_allow_html=True)

                st.write("") # 간격 조정

                # btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
                # with btn_col1:
                #     current_active = st.session_state.selected_analysis_by_section.get(section_name)
                #     is_active = (current_active == {"item": name, "code": code})
                #     btn_label = "📊 분석닫기" if is_active else "📊 분석"
                #     if st.button(btn_label, key=f"btn_{code}_{name}_{section_name}"):
                #         toggle_analysis(section_name, name, code)
                #         st.rerun()
                # with btn_col2:
                #     st.markdown(f'<div style="text-align: center; padding-top: 5px;"><a href="{naver_news_url}" target="_blank" title="네이버 뉴스 검색" style="text-decoration: none; font-size: 1.1em;"> Naver </a></div>', unsafe_allow_html=True)
                # with btn_col3:
                #     st.markdown(f'<div style="text-align: center; padding-top: 5px;"><a href="{google_news_url}" target="_blank" title="구글 뉴스 검색" style="text-decoration: none; font-size: 1.1em;"> Google </a></div>', unsafe_allow_html=True)
                
                # st.write("") # 간격 조정

    # 각 섹션(예: [삼성01], [키움W]) 바로 아래에 해당 섹션의 선택된 종목 분석 차트 렌더링
    active_in_section = st.session_state.selected_analysis_by_section.get(section_name)
    if active_in_section:
        st.markdown("---")
        st.subheader(f"📊 {active_in_section['item']}")
        
        with st.spinner(f"{active_in_section['item']} 데이터를 불러오는 중입니다..."):
            df = load_data(active_in_section["code"])
            if df is not None and not df.empty:
                fig = showV(active_in_section["item"], df)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error(f"{active_in_section['item']} 데이터를 가져오지 못했습니다.")
    
    st.markdown("---") # 섹션별 구분선
