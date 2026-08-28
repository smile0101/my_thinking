import os
import pandas as pd
from pymongo import MongoClient
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm
matplotlib.use("Agg")  # 화면 없이 이미지로만 렌더링 (Streamlit 의존 제거)
import os, numpy as np, FinanceDataReader as fdr
from scipy.signal import find_peaks
from pymongo import MongoClient
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
from urllib.parse import quote
import requests
from io import StringIO

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="주식", layout="wide")

def set_korean_font():
    plt.rcParams['axes.unicode_minus'] = False

    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux(Streamlit Cloud)
        "C:/Windows/Fonts/malgun.ttf",                       # Windows 로컬
        "/tmp/NanumGothic.ttf",                               # 다운로드 캐시
    ]

    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rc('font', family=font_name)
            return

    # 로컬에 없으면 다운로드 시도
    font_path = "/tmp/NanumGothic.ttf"
    font_url = "https://github.com/googlefonts/nanum-gothic/raw/main/fonts/ttf/NanumGothic.ttf"
    try:
        import urllib.request
        urllib.request.urlretrieve(font_url, font_path)
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    except Exception as e:
        print(f"폰트 다운로드 실패: {e}")

set_korean_font()

@st.cache_resource
def get_mongo_client():
    MONGO_URL = st.secrets["mongo_uri"]
    return MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True)

def get_col(name):
    return get_mongo_client().stock[name]

def get_df(name):
    return pd.DataFrame(list(get_col(name).find()))

def get_vals(df_, code, cols):
    return df_.loc[df_['코드'] == code, cols].iloc[0].to_dict()

def color_format(val):
    try:
        v = float(val)
        color = "red" if v < 0 else "black"
        text = f"{v:,.0f}"   # 정수(콤마 포함)로 표시
    except (TypeError, ValueError):
        color = "black"
        text = val
    return f'<span style="color:{color}">{text}</span>'

df, df_I, df_F = get_df("stock"), get_df("Info"), get_df("Focc")

name_list = df['종목'].tolist()

if 'selected_name' not in st.session_state or st.session_state['selected_name'] not in name_list:
    st.session_state['selected_name'] = name_list[0]

def on_name_change():
    name = st.session_state['selected_name']
    code = df.loc[df['종목'] == name, '코드'].iloc[0]
    st.session_state['selected_code'] = code

col = st.columns([0.3, 2, 8])

with col[0]:
    st.markdown("<h4 style='font-size: 25px; font-weight: bold;'>📊 </h4>", unsafe_allow_html=True)
with col[1]:
    st.selectbox('', name_list, key='selected_name', on_change=on_name_change, label_visibility='collapsed')

if 'selected_code' not in st.session_state:
    on_name_change()


## 데이터 ##
#########################################################################################################
code = st.session_state['selected_code']  ## code
item = st.session_state['selected_name']  ## item

def load_data(code, T=60, N =1):
    try :
        day = (datetime.now() - timedelta(days=300)).strftime("%Y%m%d") #300
        dd = fdr.DataReader(code, day).reset_index()
        # dd = fdr.DataReader(code, '20250101', '20260118').reset_index()
        if 'index' in dd.columns:
            dd = dd.rename(columns={'index': 'Date'})
        if 'Change' in dd.columns:
            dd['Change'] = round(dd['Change'] * 100, 2)
        else:
            dd['Change'] = round(dd['Close'].pct_change() * 100, 2)

        dd = dd.ffill()

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
        print("실패")
        return None

def _fetch_naver_frgn_page(code):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f'https://finance.naver.com/item/frgn.naver?code={code}'
    res = requests.get(url, headers=headers)

    try:
        fk = pd.read_html(StringIO(res.text))[2]
        fk = fk.dropna()
        fk.columns = ['날짜','종가','전일비','등락률','거래량','기관','외국인','보유량','보유율']
    except:
        fk = pd.read_html(StringIO(res.text))[3]
        fk = fk.dropna()
        fk.columns = ['날짜','종가','전일비','등락률','거래량','기관','외국인','보유량','보유율']

    fk['개인'] = -(fk['외국인'] + fk['기관'])
  
    if fk['보유율'].dtype == 'O':
        fk['보유율'] = fk['보유율'].str.replace('%','').astype(float)

    fk['dd'] = pd.to_datetime(fk['날짜']).dt.strftime('%m.%d')
    for col in ['외국인', '기관', '개인']:
        fk[col] = (fk[col] / 1000).round(0).fillna(0).astype(int)

    fk = fk[['날짜','dd', '종가', '외국인', '기관', '개인', '보유율']]

    return fk

dfv = load_data(code)
dfc = dfv.tail(20).copy()
dff = _fetch_naver_frgn_page(code)
dft = dfc[['Date','Close','Change']].merge(dff[['dd', '외국인', '기관', '개인', '보유율']],
    left_on='Date', right_on='dd', how='left').drop(columns='dd')
cols = ['외국인', '기관', '개인']
dft[cols] = dft[cols].fillna(0).astype(int)
dft['보유율'] = dft['보유율'].ffill()
dfc['거래금'] = dfc['Close'] * dfc['Volume']

amt_그제, amt_어제, amt_오늘 = dfc['거래금'].tail(3) / 100000000  # 억원 단위
amt = f"{amt_그제:,.0f} / {amt_어제:,.0f} / {amt_오늘:,.0f}"

V = dft['보유율'].iloc[-1]
try:
    V = float(V)
except (TypeError, ValueError):
    V = float(str(V).replace('%', ''))

CC = dfv['Close'].iloc[-1]
YY = dfv["Close"].iloc[-2]
D5 = dfv["MA5"].iloc[-1]
D20 = dfv["MA20"].iloc[-1]
CD5 = int(CC-D5)
D520 = int(D5-D20)
CY = int(CC-YY)

############################        Info          ###########################################
with col[2]:
    v  = get_vals(df,   code, ['순위', '시가'])
    vi = get_vals(df_I, code, ['유통', 'EPS','ROE', '매출_24', '매출_25', '매출_26',
                                '영익률_24', '영익률_25', '영익률_26', '지분율'])
    ROE     = color_format(vi['ROE'])
    매출_24  = color_format(vi['매출_24'])
    매출_25  = color_format(vi['매출_25'])
    매출_26  = color_format(vi['매출_26'])
    영익률_24 = color_format(vi['영익률_24'])
    영익률_25 = color_format(vi['영익률_25'])
    영익률_26 = vi['영익률_26']
    try:
        EPS = vi['EPS']
        if CC is not None and EPS != 0:
            PER = round(CC / EPS, 1)
        else:
            PER = '-'
    except (ValueError, TypeError):
        PER = '-'

    try:
        시가_int = f"{float(v['시가']):,.0f}"
    except (TypeError, ValueError):
        시가_int = v['시가']

    CCT = f'<span style="color:blue;">{CC:,}</span>'
    유통_Red    = f'<span style="color:red;">{vi["유통"]}</span>'

    try:
        영익률_26_val = f"{float(영익률_26):,.0f}"
    except (TypeError, ValueError):
        영익률_26_val = 영익률_26
    영익률_26_Red = f'<span style="color:red;">{영익률_26_val}</span>'
    
    ts = fdr.DataReader(code).tail(60)
    high_3m = ts['Close'].max()
    low_3m  = ts['Close'].min()
    Hc = (high_3m -CC)/CC*100
    Lc = (CC - low_3m)/low_3m*100
    Hcc = f'<span style="color:blue;">{Hc:.1F}%</span>'
    Lcc = f'<span style="color:red;">{Lc:.1F}%</span>'
    Vc = f'<span style="color:orange;">{V:.2f}% </span>'

    st.markdown(f"""
        <span style="font-size:16px;font-weight:bold;margin-right:30px;">현재 :{CCT}, &nbsp;&nbsp;&nbsp; H: {high_3m:,} ({Hcc}) /L: {low_3m:,} ({Lcc}) </span>
       <span style="font-size:16px;font-weight:bold;margin-right:30px;"> 순위 : {v['순위']}위 / {시가_int}천억   </span>
        <span style="font-size:16px;font-weight:bold;margin-right:30px;">매출 : {매출_24}/{매출_25}/{매출_26}</span>
        <span style="font-size:16px;font-weight:bold;">영업이익 : {영익률_24}/{영익률_25}/{영익률_26_Red}</span> """, unsafe_allow_html=True)

    st.markdown(f""" <span style="font-size:16px;font-weight:bold;margin-right:10px;">유통 : {유통_Red} / PER : {PER} / ROE : {ROE}&nbsp;&nbsp;&nbsp; 
    보유율 : {Vc} &nbsp;&nbsp;&nbsp; 거래금 : {amt} (억원)</span>
    <span style="font-size:12px;"> {vi['지분율']} </span>  """, unsafe_allow_html=True)    

############################        메모          ###########################################
def get_val(df_, col_name, code):
    return df_.loc[df_['코드'] == code, col_name].iloc[0]
Ref = get_val(df, '기준', code)
History = get_val(df, 'History', code)
Memo = get_val(df, 'Memo', code)
 
def update_field(code, field, value):
    get_col("stock").update_one({'코드': code}, {'$set': {field: value}})
 
####  Memo   #####
row_memo = st.columns([0.8, 8])

#  글자 크기 변경 
st.markdown("""  <style> div[data-testid="stTextInput"] input { font-size: 13px !important; 
        height: 45px !important;     /* 글자 크기에 맞춰 입력창 높이도 조정할 경우 사용 */ }  </style> """, unsafe_allow_html=True)
with row_memo[0]:
    st.markdown("<div style='padding-top: 12px;'>📝 Memo</div>", unsafe_allow_html=True)
with row_memo[1]:
    memo_val = st.text_input("", value=Memo, key=f"memo_{code}", label_visibility='collapsed')

# 기준 / History 한줄 
row_rh = st.columns([0.5, 2, 0.7, 6.1])
with row_rh[0]:
    st.markdown("<div style='padding-top: 12px;'>📝 기준</div>", unsafe_allow_html=True)
with row_rh[1]:
    ref_val = st.text_input("", value=Ref, key=f"ref_{code}", label_visibility='collapsed')
with row_rh[2]:
    st.markdown("<div style='padding-top: 12px;'>📝 History</div>", unsafe_allow_html=True)
with row_rh[3]:
    history_val = st.text_input("", value=History, key=f"history_{code}", label_visibility='collapsed')

############################        Btton        ###########################################

def format2(val):
    if val > 0:
        return f'<span style="color:#d63031; font-weight:bold;">▲{val:,}</span>'
    elif val < 0:
        return f'<span style="color:#0984e3; font-weight:bold;">▼{abs(val):,}</span>'
    else:
        return f'<span>0</span>'

btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:15px;margin:2px 20px 2px 0;"

url_think = f'https://www.thinkpool.com/item/{code}'
url_min   = f'https://m.stock.naver.com/fchart/domestic/stock/{code}'
url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}'
url_fn    = f"https://wcomp.fnguide.com/?c_id=AA&menu_type=01&cmp_cd={code}"
url_nv    = f'https://m.stock.naver.com/domestic/stock/{code}/research'
url_ggl   = f"https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko"

row_link = st.columns([0.5, 6, 3, 1])
with row_link[1]:
    st.markdown(
        f'<a href="{url_think}" target="_blank" style="{btn}">Think</a>'
        f'<a href="{url_min}"   target="_blank" style="{btn}">chart</a>'
        f'<a href="{url_tr}"    target="_blank" style="{btn}">Tr</a>'
        f'<a href="{url_fn}"    target="_blank" style="{btn}">Fn</a>'
        f'<a href="{url_nv}"    target="_blank" style="{btn}">Nv</a>'
        f'<a href="{url_ggl}"   target="_blank" style="{btn}">Google</a>',
        unsafe_allow_html=True
    )
with row_link[2]:
    ch_combined = f"{format2(CD5)} / {format2(D520)} &emsp;&emsp;({format2(CY)})"
    st.markdown( f'<div style="font-size: 18px;">{ch_combined}</div>', unsafe_allow_html=True, )



with row_link[3]:
    if st.button("💾 저장", key=f"btn_save_{code}"):
        update_field(code, '기준', ref_val)
        update_field(code, 'History', history_val)
        update_field(code, 'Memo', memo_val)
        st.success("저장되었습니다.")
        st.rerun()
################################### 투자자/ 매몰도  ########################################################

cols1 = st.columns(3)
cols1[0].image(f'https://webchart.thinkpool.com/2021ReNew/CumulationSelling/A{code}.png',
               use_container_width=True, caption="투자자")
cols1[1].image(f'https://ssl.pstatic.net/imgfinance/chart/item/area/week/{code}.png',
               use_container_width=True, caption="5일 주가")
cols1[2].image(f'https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png',
               use_container_width=True, caption="매몰도")

############################        Table        ###########################################
def calc_period(df, start, end, label):
    sub = df.tail(end) if start == 0 else df.iloc[-end:-start]
    return {  'Close': int(round(sub['Close'].mean(), 0)),
        'Change': round(sub['Change'].sum(), 1),       
        '외인': int(sub['외국인'].sum()),
        '기관': int(sub['기관'].sum()),
        '개인': int(sub['개인'].sum()),  }

def fmt_cell(val, row):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    if row == 'Change':
        return f'{val:+.1f}%'
    if row in ('외인', '기관', '개인', '종가'):
        return f'{int(val):,}'
    return str(val)

# ── 종목별 표 HTML 생성 ─────────────────────────────
def build_table_html(df):
    periods = {}
    n = len(df)
    if n >= 5:  periods['1W'] = calc_period(df, 0, 5, '1W')
    if n >= 10: periods['2W'] = calc_period(df, 5, 10, '2W')
    if n >= 15: periods['3W'] = calc_period(df, 10, 15, '3W')
    if n >= 20: periods['1M'] = calc_period(df, 15, 20, '1M')

    rows_label = ['종가', 'Change', '외인', '기관', '개인']
    count_labels = ['Change', '외인', '기관', '개인']  # 종가는 갯수/강조 대상 제외
    display_10 = df.tail(10)

    table = {}
    for _, row in display_10.iterrows():
        d = row['Date']
        table[d] = { '종가': row['Close'],
            'Change': row['Change'],            
            '외인': int(row['외국인']),
            '기관': int(row['기관']),
            '개인': int(row['개인']),
        }

    for p in ['1W', '2W', '3W', '1M']:
        if p in periods:
            table[p] = dict(periods[p])
            table[p]['종가'] = periods[p]['Close']
        else:
            table[p] = {k: None for k in rows_label}

    date_cols = list(display_10['Date'])
    count_row = {}
    for rlab in rows_label:
        if rlab in count_labels:
            cnt = 0
            for d in date_cols:
                v = table[d].get(rlab)
                if v is not None and not (isinstance(v, float) and pd.isna(v)) and v > 0:
                    cnt += 1
            count_row[rlab] = cnt
        else:
            count_row[rlab] = ''
    table['갯수'] = count_row

    col_order = date_cols + ['1W', '2W', '3W', '1M', '갯수']

    html = ''
    html += '<table class="etf-table"><thead><tr><th>항목</th>'
    for col in col_order:
        cls = 'class="sep"' if col in ('1W', '갯수') else ''
        html += f'<th {cls}>{col}</th>'
    html += '</tr></thead><tbody>'

    for rlab in rows_label:
        html += f'<tr><td class="row-label">{rlab}</td>'
        for col in col_order:
            val = table[col].get(rlab) if table[col] else None

            if col == '갯수':
                text = str(val) if val != '' else ''
                bg = 'background-color:#FFCCCC; color:#CC0000;' if (rlab in count_labels and val != '' and val > 5) else ''
            else:
                text = fmt_cell(val, rlab)
                is_num = val is not None and not (isinstance(val, float) and pd.isna(val))
                bg = 'background-color:#FFCCCC; color:#CC0000;' if (rlab in count_labels and is_num and val > 0) else ''

            cls_list = []
            if col in ('1W', '2W', '3W', '1M'):
                cls_list.append('period')
            if col in ('1W', '갯수'):
                cls_list.append('sep')

            html += f'<td class="{" ".join(cls_list)}" style="{bg}">{text}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html

if dfv is None or dfv.empty:
    st.error("데이터를 불러오지 못했습니다.")
else:
    table_html = build_table_html( dft)

    # ── 표 스타일 (한 번만 정의) ─────────────────────────
    st.markdown("""
    <style>
    .etf-table { border-collapse: collapse; width: 100%; font-size: 20px; }
    .etf-table th, .etf-table td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; }
    .etf-table th { background-color: #f2f2f2; font-size: 18px; }
    .etf-table td.row-label { text-align: left; font-weight: bold; background-color: #fafafa; font-size: 18px; }
    .etf-table th.sep, .etf-table td.sep { border-left: 2px solid #333; }
    .etf-table td.period { background-color: #f9f9f9; }
    .etf-table tbody td:not(.row-label) { font-size: 16px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # ── 표 렌더링 ─────────────────────────────────────
    st.markdown(table_html, unsafe_allow_html=True)

####################################### 그래프 ##################################################################
def showV( item, d, T=60):

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
    max_100 = d['Close'].max()
    min_100 = d['Close'].min()
    min_100_idx = d['Close'].values.argmin()          # 최저값 위치(위치 기반 index)
    min_100_date = dates[min_100_idx]                 # 최저값 날짜(x좌표)

    # ## 1주일
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
    fig = plt.figure(figsize=(18.5,11)) #14, 7.5
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 0.21, 0.21, 0.21], hspace=0.01)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # x축 공유
    ax3 = fig.add_subplot(gs[2], sharex=ax1) # x축 공유
    ax4 = fig.add_subplot(gs[3], sharex=ax1) # x축 공유

    ax1.set_title(f"{code}")

    ax1.plot(d['Date'], d['Close'], linewidth=1.4, label='Close')
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)

    ax1.axhline(max_100, linestyle=':', color = 'black', linewidth=1.0)
    ax1.axhline(min_100, linestyle=':', color ='black', linewidth=1.0)
    ax1.plot(min_100_date, min_100, marker='d', color='magenta', markersize=20, zorder=6)  
###############################################################################
    d_len = len(d)
    periods_config = [   
        {'days': 20, 'text_idx': (T-19), 'color': 'black'},
        {'days': 40, 'text_idx': (T-39), 'color': 'blue',},
        {'days': 60, 'text_idx': (T-59), 'color': 'green'},
        {'days': 80, 'text_idx': (T-79), 'color': 'black'},
        {'days': 100, 'text_idx': 1, 'color': 'black'}
    ]
    if d_len > 20:
        for config in periods_config:  ## config값 가져옴
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
                    pass # dates 범위를 벗어날 경우 출력 생략

    if d_len > 20:
        periods = {'1M': 20, '2M': 40, '3M': 60, '4M' : 80 }
        colors = ['#FF5733', '#33FF57', '#3357FF', "#EDF51A" ]

        for i, (label, offset) in enumerate(periods.items()):
            # 데이터 길이가 offset보다 클 때만 마커 표시
            if d_len > offset:
                idx = d_len - 1 - offset
                if idx >= 0:
                    target_date = d['Date'].iloc[idx]
                    target_price = d['Close'].iloc[idx]
                    
                    # 동그라미 마커
                    ax1.plot(target_date, target_price, 'o', markersize=12, 
                            markeredgecolor='black', markerfacecolor=colors[i], zorder=5)

    ax1.text(dates[0], max_100, f' Max {int(max_100):,}', fontsize=12, va='bottom')
    ax1.text(dates[0], min_100, f' Min {int(min_100):,}', fontsize=12, va='top')

    # 1주일(5일) 상세 표시
    x5_start, x5_end = d5['Date'].iloc[0], d5['Date'].iloc[-1]
    ax1.hlines(y=max_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    ax1.hlines(y=min_5, xmin=x5_start, xmax=x5_end, colors='red', linestyles='--', linewidth=1.2)
    
    x5_idx = min(T-7, len(dates)-1)
    x5_text_pos = dates[x5_idx]
    ax1.annotate('', xy=(x5_text_pos, max_5), xytext=(x5_text_pos, CC), arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(x5_text_pos, (max_5 + CC)/2, f'+{gap_up_5:.1f}%', ha='left', fontsize=12, bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.8))
    ax1.annotate('', xy=(x5_text_pos, CC), xytext=(x5_text_pos, min_5), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax1.text(x5_text_pos, (CC + min_5)/2, f'-{gap_dn_5:.1f}%', ha='right', fontsize=12, bbox=dict(boxstyle='round', fc='lightcyan', alpha=0.8))
    ax1.text(d5['Date'].iloc[4], max_5, f' {int(max_5):,}', color = 'red', fontsize=10, ha='left', va='center')
    ax1.text(d5['Date'].iloc[0], min_5, f' {int(min_5):,}', color = 'red', fontsize=10, ha='right', va='top')

    # 거래 변동률
    ax1_t = ax1.twinx()
    ax1_t.bar(d['Date'], d['Change'], alpha=0.25)

    for i in [-3,-2,-1]:
        ax1_t.text( d['Date'].iloc[i], d['Change'].iloc[i] + 0.1,str(d['Change'].iloc[i]), ha='center',
            va='bottom', fontsize=11, color='black')
    ax1_t.tick_params(axis='y', labelsize=6)
    for j in range(len(d)):
        ax1.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax1.tick_params(axis='x', rotation=45, labelsize=1)
    ax1.tick_params(axis='y', labelsize=10) # 6
    pos = ax1.get_position()
    ax1.set_position([0.06, pos.y0, 0.9, pos.height])

    # 그래프2
    ax2.plot(d['Date'], d['Close'], linestyle='--', color='pink')
    ax2.plot(d['Date'], d['MA5'], linestyle='-.', color='green', label='MA5')
    ax2.plot(d['Date'], d['MA10'], linestyle='-.', color='black', label='MA10')
    ax2.plot(d['Date'], d['MA20'], linestyle='-', color='magenta', label='MA20')
    ax2.plot(d['Date'], d['MA60'], linestyle='-', color='blue', label='MA60')
    ax2.plot(d['Date'], d['MA120'], linestyle='-', color='black', label='MA120')
    ax2.axhline(round(d['Close'].mean(),1), color='orange', linestyle='--')
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

# --- 그래프3 (수정) ---
    ax3.plot(d['Date'], d['MA5'], label='MA5', color='red', linewidth=1.5)
    ax3.plot(d['Date'], d['MA10'], label='MA10', color='blue', linewidth=1.3)    
    ax32 = ax3.twinx()
    ax32.bar(d['Date'], d['MA5_d'], color=np.where(d['MA5_d']>=0,'royalblue','salmon'), alpha=0.5)
    ax32.axhline(y=0, color='green', linestyle='--', linewidth=2)
    for j in range(len(d)):
        ax3.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax3.tick_params(axis='y', labelsize=6)
    ax32.tick_params(axis='y', labelsize=6)

    # --- 그래프4 (수정) ---
    d['S5_detail'] = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    
    ax4.plot(d['Date'], d['MA5_d'], label='MA5변화', color='green', linestyle='-', alpha=0.5)
    ax4.legend( loc='upper left', fontsize=12, frameon=False )
    ax4.axhline(y=0 , color='orange', linestyle='--', linewidth=1)
    for j in range(len(d)):
        ax4.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax4.tick_params(axis='x', rotation=45)
    for label in ax4.get_xticklabels():
        label.set_fontsize(12)  ## X좌표 크기


    # 보조축 설정
    ax5 = ax4.twinx()
    ax5.plot(d['Date'], d['S5_detail'], label='S5', color='magenta', linestyle='-.', linewidth=2)
    ax5.plot(d['Date'], d['S10_detail'], label='S10', linestyle='--', color='blue', linewidth=1)
    # ax5.axhline(y=89.90, color='orange', linestyle='--', linewidth=1)
    ax5.set_ylim(89.68, 90.03)
    ax5.set_yticks(np.arange(89.68, 90.03, 0.05))
    ax5.tick_params(axis='y', labelsize=6)

    # 또 다른 보조축 (종가 표시용)
    ax6 = ax4.twinx()
    ax6.plot(d['Date'], d['Close'], label='종가', linestyle='-', color='black', linewidth=2, alpha=0.6)
    ax6.tick_params(axis='y', labelsize=6)

    # --- 전체 레이아웃 정렬 (핵심) ---
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.tight_layout()
    # 만약 여백이 너무 좁다면 아래 코드로 미세조정
    fig.subplots_adjust(hspace=0.05, left=0.05, right=0.95, top = 0.95)

    return fig
##############################################################################################################################

if dfv is None or dfv.empty:
    st.error("데이터를 불러오지 못했습니다. (fdr.DataReader 실패)")
else:

    fig = showV(item, dfv)

    # ── 표 스타일 (한 번만 정의) ─────────────────────────
    st.markdown("""
    <style>
    .etf-table { border-collapse: collapse; width: 100%; font-size: 13px; }
    .etf-table th, .etf-table td { border: 1px solid #ddd; padding: 4px 6px; text-align: center; }
    .etf-table th { background-color: #f2f2f2; }
    .etf-table td.row-label { text-align: left; font-weight: bold; background-color: #fafafa; }
    .etf-table th.sep, .etf-table td.sep { border-left: 2px solid #333; }
    .etf-table td.period { background-color: #f9f9f9; }
    </style>
    """, unsafe_allow_html=True)

    # ── 그래프 렌더링 ──────────────────────────────────
    st.pyplot(fig)
    plt.close(fig)  # 메모리 누수 방지 (matplotlib figure는 명시적으로 닫아줘야 함)


#############################################################################################

def plot_stock_st(df, stock_name):
    set_korean_font()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = range(len(df))

    ax1.plot(x, df['보유율'], marker='o', color='royalblue', label='보유율')
    ax1.set_ylabel("보유율 (%)", color='royalblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['일자'], rotation=45, fontsize=6)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, df['종가'], linestyle='--', color='crimson',
            marker='s', linewidth=2, label='종가')
    ax2.tick_params(axis='y', labelcolor='crimson',labelsize=8)

    plt.title(f"{stock_name} 주가", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
#######################################################################################################

MONGO_URL = st.secrets["mongo_uri"]
try:
    with MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True) as client:
        col   = client.stock.FOCC
        db_df = pd.DataFrame(col.find({"종목명": item}, {"_id": 0}))
except Exception as e:
    st.warning(f"MongoDB 연결 오류: {e}")
    db_df = pd.DataFrame()

# plot_df: page=1 기준으로 구성
plot_df = dff[['날짜','종가','보유율']].head(10).copy()
plot_df['보유율'] = plot_df['보유율'].astype(str).str.replace('%','').astype(float)
plot_df['날짜']   = pd.to_datetime(plot_df['날짜'])
plot_df['일자']   = plot_df['날짜'].dt.strftime('%m.%d')
plot_df['종목명'] = item
plot_df['코드']   = code

if not db_df.empty:
    if '날짜' in db_df.columns:
        db_df['날짜'] = pd.to_datetime(db_df['날짜'])
    if '일자' not in db_df.columns and '날짜' in db_df.columns:
        db_df['일자'] = db_df['날짜'].dt.strftime('%m.%d')
    merged = (pd.concat([db_df, plot_df], ignore_index=True)
                .drop_duplicates(subset=['날짜'])
                .sort_values('날짜')
                .reset_index(drop=True))
else:
    merged = plot_df.sort_values('날짜').reset_index(drop=True)

plot_stock_st(merged, item)

