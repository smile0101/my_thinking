import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm

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
#########################  지수 그래프 #######################################################################
def load_data(code):
    try:
        dd = fdr.DataReader(code).tail(200).reset_index()
        if 'index' in dd.columns:
            dd = dd.rename(columns={'index': 'Date'})
        if 'Change' in dd.columns:
            dd['Change'] = round(dd['Change'] * 100, 2)
        else:
            dd['Change'] = round(dd['Close'].pct_change() * 100, 2)
        for n in [5, 10, 20, 60, 120]:
            dd[f'MA{n}'] = dd['Close'].rolling(window=n).mean()
        dd['MA5_d']  = dd['MA5'].diff()
        dd['MA10_d'] = dd['MA10'].diff()
        dd['S5']  = np.degrees(np.arctan(np.gradient(dd['MA5'].values)))
        dd['S10'] = np.degrees(np.arctan(np.gradient(dd['MA10'].values)))

        dd = dd.tail(55).copy()
        dd['Date'] = pd.to_datetime(dd['Date']).dt.strftime('%m.%d')
        return dd
    except Exception:
        print("실패")
        return None


# ── 누적합 기간 컬럼 계산 ─────────────────────────────
def calc_period(df, rows, label):
    sub = df.tail(rows)
    return {
        'Close'  : int(sub['Close'].mean()),
        'Change': round(sub['Change'].sum(), 1), }

def fmt_cell(val, row):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    if row == 'Close':
        return f'{int(val):,}'
    if row == 'Change':
        return f'{val:+.1f}%'
    return str(val)

def clean(text):
    text = text.strip()
    return "" if text in ("-", "", "N/A") else text

def td_after_th(container, keyword, exclude=None):
    if not container:
        return ""
    for th in container.find_all("th"):
        text = th.get_text(" ", strip=True)
        if keyword in text and (not exclude or exclude not in text):
            td = th.find_next_sibling("td")
            if td:
                return clean(td.get_text(strip=True).replace(",", ""))
    return ""


def graph_n(item, d):
    # 이동평균선 간 교차점 찾기 함수
    def find_cross_points(df, col1, col2):
        cross_points = []
        for i in range(1, len(df)):
            if (df[col1].iloc[i] > df[col2].iloc[i] and df[col1].iloc[i-1] <= df[col2].iloc[i-1]) or \
            (df[col1].iloc[i] < df[col2].iloc[i] and df[col1].iloc[i-1] >= df[col2].iloc[i-1]):
                cross_points.append(i - 1)  # 교차점 날짜를 이전 인덱스로 설정
        return cross_points

    # 마지막 교차점만 추출하는 함수
    def extract_last_cross_data(df, cross_points, col1, col2):
        if cross_points:
            last_cross_index = cross_points[-1]
            last_cross_date = df['Date'].iloc[last_cross_index]
            last_cross_value = df[[col1, col2]].iloc[last_cross_index].mean()
            return last_cross_date, last_cross_value
        return None, None

    # 피크와 밸리 계산 함수
    def find_extrema(values):
        peaks, _ = find_peaks(values)
        valleys, _ = find_peaks(-values)
        return peaks, valleys

    # 최대/최소값과 날짜 추출
    def extract_extrema_data(df, values, peaks, valleys):
        maxi = values.iloc[peaks]
        mini = values.iloc[valleys]
        max_dates = df['Date'].iloc[peaks]
        min_dates = df['Date'].iloc[valleys]
        return maxi, mini, max_dates, min_dates

    values_day = d['Close']
    values_5day = d['MA5'].dropna()

    peaks_day, valleys_day = find_extrema(values_day)
    peaks_5day, valleys_5day = find_extrema(values_5day)

    # 최대/최소값과 날짜 추출
    maxi_day, mini_day, max_dates_day, min_dates_day = extract_extrema_data(d, values_day, peaks_day, valleys_day)
    maxi_5day, mini_5day, max_dates_5day, min_dates_5day = extract_extrema_data(d, values_5day, peaks_5day, valleys_5day)
    # 이동평균선 교차점 계산 및 마지막 교차점 추출
    cross_close_20_points = find_cross_points(d, 'Close', 'MA20')
    last_cross_close_20_date, last_cross_close_20_value = extract_last_cross_data(d, cross_close_20_points, 'Close', 'MA20')

    cross_close_60_points = find_cross_points(d, 'Close', 'MA60')
    last_cross_close_60_date, last_cross_close_60_value = extract_last_cross_data(d, cross_close_60_points, 'Close', 'MA60')

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)  # 12, 9.5 / 7.5, 7
    ax2, ax3, ax4 = axs

    CC = int(d['Close'].iloc[-1])
    ax2.set_title(f"{item} "  , fontsize=12, color="blue")

    # [1] HL 그래프 (맨 위)
    ax2.plot(d['Date'], d['Close'], label='Close', color='blue', linewidth=1.5)
    ax2.plot(d['Date'], d['High'], label='High', color='green', linestyle='--', linewidth=1.2)
    ax2.plot(d['Date'], d['Low'], label='Low', color='red', linestyle='--', linewidth=1.2)
    for j in range(len(d)):
        ax2.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=0.8, alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(d['Date'], d['Change'], color='gray', alpha=0.3, label='Change (%)')
    for i in [-3,-2,-1]:
        ax2_twin.text( d['Date'].iloc[i], d['Change'].iloc[i] + 0.1,str(d['Change'].iloc[i]), ha='center',
            va='bottom', fontsize=10, color='black',fontweight='bold')

    # [2] 이동평균선 그래프 (가운데)
    ax3.plot(d['Date'], d['Close'], linestyle='--', color='pink')
    ax3.plot(d['Date'], d['MA5'], linestyle='-.', color='green', label='5일')
    ax3.plot(d['Date'], d['MA20'], linestyle='-', color='magenta')
    ax3.plot(d['Date'], d['MA60'], linestyle='-', color='blue')
    ax3.plot(d['Date'], d['MA120'], linestyle='-', color='black', alpha=0.5)
    ax3.axhline(round(d['Close'].mean(), 1), color='orange', linestyle='--')

    ax3.plot(min_dates_day, mini_day, "o", color='purple', markersize=5)
    ax3.plot(max_dates_day, maxi_day, "o", color='orange', markersize=5)
    ax3.plot(max_dates_5day, maxi_5day, "o", color='red', markersize=11)
    ax3.plot(min_dates_5day, mini_5day, "o", color='purple', markersize=12)

    if last_cross_close_20_date:
        ax3.plot(last_cross_close_20_date, last_cross_close_20_value, "d", color='magenta', markersize=12, label='20일')
    if last_cross_close_60_date:
        ax3.plot(last_cross_close_60_date, last_cross_close_60_value, "d", color='blue', markersize=12, label='60일')

    for j in range(len(d)):
        ax3.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)

    ax3.legend(loc='upper left')

    ax4.plot(d['Date'], d['MA5'], label='5일', color='red', linewidth=1.5)
    ax4.plot(d['Date'], d['MA10'], label='10일', color='blue', linewidth=1.3)
    ax4.axhline(y=d['MA5'].mean(), color='green', linestyle='--', linewidth=2)
    ax42 = ax4.twinx()
    ax42.bar(d['Date'],d['MA5_d'], color=np.where(d['MA5_d'] >= 0, 'royalblue', 'salmon'), alpha=0.5 )
    ax4.legend(loc='upper left')

    for j in range(len(d)):
        ax4.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax4.tick_params(axis='x', rotation=45)
    for label in ax4.get_xticklabels():
        label.set_fontsize(6.6)

    ax2.tick_params(axis='y', labelsize=6)
    ax3.tick_params(axis='y', labelsize=6)
    ax4.tick_params(axis='y', labelsize=6)
    ax2_twin.tick_params(axis='y', labelsize=5)
    ax42.tick_params(axis='y', labelsize=5)
    plt.subplots_adjust(hspace=0.1)
    plt.rcParams['axes.unicode_minus'] = False
    return fig

# ── 지수 목록: (표시이름, 코드) 순서로 정확히 매칭 ─────────────────────
indices = [('코스피', '^KS11'), ('코스닥', '^KQ11'), ('다우', 'DJI'), ('나스닥', 'IXIC')]

with st.spinner("데이터 수집 중..."):
    data = {item: load_data(code) for item, code in indices}

def render_group(group_indices, data):
    """group_indices 안의 지수들을 통합 테이블 1개 + 그래프(2개씩 한 줄)로 출력"""
    rows_label = ['Close', 'Change']
    col_order = None
    item_tables = {}  # item별 일자+기간 테이블 저장

    for item, _ in group_indices:
        df = data[item]
        n = len(df)
        periods = {}
        if n >= 5:  periods['1W'] = calc_period(df, 5, '1W')
        if n >= 10: periods['2W'] = calc_period(df, 10, '2W')
        if n >= 15: periods['3W'] = calc_period(df, 15, '3W')
        if n >= 25: periods['1M'] = calc_period(df, 25, '1M')
        if n >= 50: periods['2M'] = calc_period(df, 50, '2M')

        display_10 = df.tail(10)
        if col_order is None:
            col_order = list(display_10['Date']) + ['1W', '2W', '3W', '1M', '2M']

        day_table = {}
        for _, row in display_10.iterrows():
            d = row['Date']
            day_table[d] = {'Close': int(row['Close']), 'Change': row['Change']}
        for p in ['1W', '2W', '3W', '1M', '2M']:
            day_table[p] = periods[p] if p in periods else {k: None for k in rows_label}

        item_tables[item] = day_table

    html = '''
    <style>
        .etf-table { border-collapse:collapse; font-size:22px; width:100%; }
        .etf-table th, .etf-table td { padding:5px 8px; text-align:center; border:1px solid #ddd; }
        .etf-table th { background:#f0f0f0; font-weight:bold; }
        .etf-table td.row-label { text-align:left; font-weight:bold; background:#f8f8f8; }
        .etf-table td.period { background:#f0fff0; }
        .etf-table td.sep { border-left:2px solid #888 !important; }
    </style>
    <table class="etf-table">
    <thead><tr><th>항목</th>'''

    for col in col_order:
        cls = 'class="sep"' if col == '1W' else ''
        html += f'<th {cls}>{col}</th>'
    html += '</tr></thead><tbody>'

    for item, _ in group_indices:
        day_table = item_tables[item]
        for rlab in rows_label:
            html += f'<tr><td class="row-label">{item}</td>'
            for col in col_order:
                val = day_table[col].get(rlab) if day_table[col] else None
                text = fmt_cell(val, rlab)
                cls_list = []
                if col in ['1W', '2W', '1M']:
                    cls_list.append('period')
                if col == '1W':
                    cls_list.append('sep')
                bg = ''
                if rlab == 'Change':
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        if val > 0:
                            bg = 'background-color:#FFD1DC;'
                html += f'<td class="{" ".join(cls_list)}" style="{bg}">{text}</td>'
            html += '</tr>'

    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)
    # st.divider()

    for i in range(0, len(group_indices), 2):
        row_items = group_indices[i:i+2]
        cols = st.columns([2, 2])
        for col, (item, _) in zip(cols, row_items):
            with col:
                fig = graph_n(item, data[item])
                st.pyplot(fig)

render_group([('코스피', '^KS11'), ('코스닥', '^KQ11')], data)
render_group([('다우', 'DJI'), ('나스닥', 'IXIC')], data)

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
