import os
import pandas as pd
from pymongo import MongoClient
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm
matplotlib.use("Agg")  # 화면 없이 이미지로만 렌더링 (Streamlit 의존 )
import os, numpy as np, FinanceDataReader as fdr
from scipy.signal import find_peaks
from pymongo import MongoClient
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
from urllib.parse import quote
import requests
from io import StringIO
from scipy.signal import argrelextrema

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

def _fetch_naver_frgn_page(code):

    url = f'https://stock.naver.com/api/domestic/detail/{code}/trend'

    headers = { 'User-Agent': 'Mozilla/5.0',
        'Referer': f'https://stock.naver.com/domestic/stock/{code}/price' }

    params = { 'tradeType': 'KRX','startIdx': 0, 'pageSize': 100 }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        fk = pd.DataFrame(data)
    except Exception:
        return pd.DataFrame( columns=['날짜', 'dd', '종가', '외국인', '기관', '개인', '보유율'] )

    if fk.empty:
        return pd.DataFrame( columns=['날짜', 'dd', '종가', '외국인', '기관', '개인', '보유율'] )

    # 새 네이버 API → 기존 컬럼명으로 변경
    fk = fk.rename(columns={ 'bizdate': '날짜', 'closePrice': '종가', 'tradeVolume': '거래량',
                            'foreignerPureBuyQuant': '외국인', 'organPureBuyQuant': '기관', 'individualPureBuyQuant': '개인', 'frgnHoldRatio': '보유율'})

    fk['날짜'] = pd.to_datetime( fk['날짜'].astype(str), format='%Y%m%d', errors='coerce'  )

    # 숫자형 변환
    for c in ['종가', '외국인', '기관', '개인', '보유율']:
        if c in fk.columns:
            fk[c] = pd.to_numeric(fk[c], errors='coerce')

    # 기존 코드와 동일하게 1,000주 단위
    for c in ['외국인', '기관', '개인']:
        fk[c] = (fk[c] / 1000).round(0).fillna(0).astype(int)

    # 기존 dd 형식
    fk['dd'] = fk['날짜'].dt.strftime('%m.%d')

    fk = fk[['날짜', 'dd', '종가', '외국인', '기관', '개인', '보유율'] ].reset_index(drop=True)
    return fk

dfv = load_data(code)  ## 데이터
dfc = dfv.tail(15).copy()
dff = _fetch_naver_frgn_page(code)
dft = dfc.merge(dff[['dd', '외국인', '기관', '개인', '보유율']], 
                left_on='Date', right_on='dd', how='left').drop(columns='dd')
dft['fo'] = dft['외국인'].cumsum()
dft['go'] = dft['기관'].cumsum()
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
D60 = dfv["MA60"].iloc[-1]
CD5 = int(D5/CC*100)
CD20 = int(D20/CC*100)
CD60 = int(D60/CC*100)
CY = int(CC-YY)
TRR = (CC-YY)/YY*100

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

w1, w2, w3, m1, m2 = build(dfv)
keys = ["1W", "2W", "3W", "1M", "2M"]
values = [w1, w2, w3, m1, m2]

all_ch = ", ".join( f"{key}: {fmt_cell(v['Change'], 'Change')}" for key, v in zip(keys, values) if v is not None )

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
        
btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:15px;margin:2px 20px 2px 0;"

url_think = f'https://www.thinkpool.com/item/{code}'
url_nv    = f'https://stock.naver.com/domestic/stock/{code}/price'
url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}'
url_fn    = f"https://wcomp.fnguide.com/?c_id=AA&menu_type=01&cmp_cd={code}"
url_min   = f'https://m.stock.naver.com/domestic/stock/{code}/total?domesticStockExchange=KRX'
url_ggl   = f"https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko"

row_link = st.columns([0.01, 3, 6.5, 1])
with row_link[1]:
    st.markdown(
        f'<a href="{url_think}" target="_blank" style="{btn}">Think</a>'
        f'<a href="{url_nv}"    target="_blank" style="{btn}">Nv</a>'
        f'<a href="{url_tr}"    target="_blank" style="{btn}">Tr</a>'
        f'<a href="{url_fn}"    target="_blank" style="{btn}">Fn</a>'
        f'<a href="{url_min}"   target="_blank" style="{btn}">Np</a>'
        f'<a href="{url_ggl}"   target="_blank" style="{btn}">Goog</a>',
        unsafe_allow_html=True
    )
with row_link[2]:
    ch_combined = f"{all_ch} &emsp;&emsp; {format3(CD5)} / {format3(CD20)} / {format3(CD60)}  &emsp;&emsp;{format2(CY)} / {format1(TRR)} "
    st.markdown( f'<div style="font-size: 18px;">{ch_combined}</div>', unsafe_allow_html=True, )

with row_link[3]:
    if st.button("💾 저장", key=f"btn_save_{code}"):
        update_field(code, '기준', ref_val)
        update_field(code, 'History', history_val)
        update_field(code, 'Memo', memo_val)
        st.success("저장되었습니다.")
        st.rerun()


####################################### 그래프 ##################################################################
def showV( item, d ):

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
    fig = plt.figure(figsize=(12,9)) #14, 7.5
    gs = gridspec.GridSpec(5, 1, height_ratios=[0.3, 0.21, 0.21, 0.21, 0.21], hspace=0.01)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # x축 공유
    ax3 = fig.add_subplot(gs[2], sharex=ax1) # x축 공유
    ax4 = fig.add_subplot(gs[3], sharex=ax1) # x축 공유
    ax7 = fig.add_subplot(gs[4], sharex=ax1) # x축 공유

    ax1.set_title(f"{item}{code}",fontsize=8)
    ##################################################################
    ax1.plot(d['Date'], d['Close'], linewidth=1.4, marker="o",label='Close', color ="red")
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)
    ax1.axhline(round(d['Close'].mean(),1), color='blue', linestyle='--')
    ax1.plot(min_15_date, min_15, marker='d', color='magenta', markersize=20, zorder=6)
    ax1.axhline(y = max_15, color="#A9ED17", linestyle='--')  
    # for i in range(-7, 0):
    #     ax1.text( d['Date'].iloc[i], d['Close'].iloc[i] + 0.1,str(d['Close'].iloc[i]), ha='center',
    #         va='top', fontsize=11, color='red', weight = "bold")

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
        label.set_fontsize(12)  ## X좌표 크기

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


    ##################################################################

    ax7.plot(d['Date'], d['외국인'], label='F', marker='o', linewidth=1, markersize=4, color='red',linestyle=":")
    ax7.plot(d['Date'], d['기관'], label='G', marker='o', linewidth=1, markersize=4, color='blue',linestyle=":")
    ax7.plot(d['Date'], d['개인'], label='I', marker='o', linewidth=1, markersize=4, color='green',linestyle=":")
    ax7.axhline(y=d['fo'].iloc[-1] , color="red", linestyle="-", linewidth=1.5, alpha=0.8)
    ax7.axhline(y=d['go'].iloc[-1] , color="blue", linestyle="-", linewidth=1.5, alpha=0.8)
    ax7.axhline(0, color='gray', linewidth=1.5, linestyle='-')
    for j in range(len(d)):
        ax7.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax7.legend(loc='upper left', fontsize=10, frameon=False)
    ax7.tick_params(axis='y', labelsize=6)

    # 최하단 서브플롯인 ax7에만 X축 레이블 표시 및 회전 설정
    ax7.tick_params(axis='x', rotation=45)
    for label in ax7.get_xticklabels():
        label.set_fontsize(8)

    # --- 전체 레이아웃 정렬 ---
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False) # ax4의 X축 눈금도 숨김 처리하여 겹침 방지

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, left=0.05, right=0.95, top=0.95)

    return fig

##############################################################################################################################

if dfv is None or dfv.empty:
    st.error("데이터를 불러오지 못했습니다. (fdr.DataReader 실패)")
else:

    fig = showV(item, dft)

    # ── 그래프 렌더링 ──────────────────────────────────
    st.pyplot(fig)
    plt.close(fig)  # 메모리 누수 방지 (matplotlib figure는 명시적으로 닫아줘야 함)

################################### 투자자/ 매몰도  ########################################################

cols1 = st.columns(3)
cols1[0].image(f'https://webchart.thinkpool.com/2021ReNew/CumulationSelling/A{code}.png',
               use_container_width=True, caption="투자자")
cols1[1].image(f'https://ssl.pstatic.net/imgfinance/chart/item/area/week/{code}.png',
               use_container_width=True, caption="5일 주가")
cols1[2].image(f'https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png',
               use_container_width=True, caption="매몰도")

#############################        Trend       ##########################################

def plot_trend_subplot(ax, df, close_prices, order_val, item):
    # 1. 저점(Minima) 계산
    minima_indices = argrelextrema(close_prices, np.less, order=order_val)[0]
    if len(minima_indices) >= 2:
        low_indices = sorted(minima_indices, reverse=True)[:3]
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
        high_indices = sorted(maxima_indices, reverse=True)[:3]
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

    # 저점 추세선 및 마커 (문자열 라벨 위치에 매칭되도록 x.iloc 사용)
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
    ax.tick_params(axis="x", rotation=45, labelsize=4)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_yticklabels([])

    sub_title = f"{order_val}. LL: {latest_low:,.0f} | HH: {latest_high:,.0f} | Cha: {cha_value}%"
    ax.set_title(sub_title, fontsize=10, pad=10)


def trend(item, code):
    df = fdr.DataReader(code).tail(50).reset_index()
    # df = fdr.DataReader(code, '20260701','20260825').reset_index()
    date_col = df.columns[0]
    df.rename(columns={date_col: "Date", "Close": "Close"}, inplace=True)

    formatted_dates = [f"{d.month}.{d.day}" for d in pd.to_datetime(df["Date"])]
    df["Date_Str"] = formatted_dates

    close_prices = df["Close"].values

    # 1행 3열 서브플롯 생성 (수정된 부분)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.5))

    # 각각 order_val=1, 2, 3으로 호출
    plot_trend_subplot(ax1, df, close_prices, order_val=1, item=item)
    plot_trend_subplot(ax2, df, close_prices, order_val=2, item=item)
    plot_trend_subplot(ax3, df, close_prices, order_val=3, item=item)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
trend(item, code)
#############################################################################################

def plot_stock_st(df, stock_name):
    set_korean_font()
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
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
