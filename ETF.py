import os
import numpy as np
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import quote
from bs4 import BeautifulSoup
from scipy.signal import find_peaks
import FinanceDataReader as fdr
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="ETF", layout="wide")
st.subheader("📊 ETF")

# ── 폰트 ──────────────────────────────────────────────
def set_korean_font():
    plt.rcParams['axes.unicode_minus'] = False
    font_path = '/tmp/NanumGothic.ttf'
    font_url  = 'https://github.com/googlefonts/nanum-gothic/raw/main/fonts/ttf/NanumGothic.ttf'
    if not os.path.exists(font_path):
        try:
            import urllib.request
            urllib.request.urlretrieve(font_url, font_path)
        except Exception as e:
            print(f"폰트 다운로드 실패: {e}")
            return
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')

set_korean_font()

# ── 외국인 페이지 크롤링 ──────────────────────────────
def _fetch_naver_frgn_page(code):
    headers = {"User-Agent": "Mozilla/5.0"}
    result = []
    for page in range(1, 7):
        res = requests.get(
            f'https://finance.naver.com/item/frgn.naver?code={code}&page={page}',
            headers=headers
        )
        try:
            fk = pd.read_html(StringIO(res.text))[2]
            fk = fk.dropna()
            if fk.shape[1] < 9:
                raise ValueError
        except Exception:
            fk = pd.read_html(StringIO(res.text))[3]
            fk = fk.dropna()

        fk.columns = ['날짜','종가','전일비','등락률','거래량','기관','외국인','보유량','보유율']
        fk['개인'] = -(fk['외국인'] + fk['기관'])
        fk['거래금'] = (fk['종가'] * fk['거래량'] / 100000000).astype(int)

        for c in ['외국인','기관','개인']:
            fk[c] = (pd.to_numeric(fk[c], errors='coerce') / 1000).fillna(0).astype(int)

        fk['날짜']   = fk['날짜'].str.slice(5)
        fk['등락률'] = pd.to_numeric(fk['등락률'].astype(str).str.replace('%',''), errors='coerce')
        fk['보유율'] = pd.to_numeric(fk['보유율'].astype(str).str.replace('%',''), errors='coerce')

        fkv = fk[['날짜','종가','등락률','개인','기관','외국인','보유율','거래금']].copy()
        result.append(fkv)

    return pd.concat(result, ignore_index=True).drop_duplicates(subset='날짜').reset_index(drop=True)

# ── 누적합 기간 컬럼 계산 ─────────────────────────────
def calc_period(df, rows, label):
    sub = df.head(rows)
    return {
        '종가'  : int(sub['종가'].mean()),
        '등락률': round(sub['등락률'].sum(), 1),
        '개인'  : int(sub['개인'].sum()),
        '기관'  : int(sub['기관'].sum()),
        '외국인': int(sub['외국인'].sum()),
        '보유율': round(sub['보유율'].max(), 1),
        '거래금': int(sub['거래금'].max()),
    }

# ── 고저 변동폭 계산 ──────────────────────────────────
def calc_hl(series: pd.Series) -> tuple[float, float]:

    hi, lo, cur = series.max(), series.min(), series.iloc[0]
    hl  = round((hi - lo) / lo * 100, 1)
    hlc = round((hi - cur) / cur * 100, 1)
    return hl, hlc

# ── 셀 포매팅 ─────────────────────────────────────────
def fmt_cell(val, row):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    if row == '종가':
        return f'{int(val):,}'
    if row == '등락률':
        return f'{val:+.1f}%'
    if row in ['개인','기관','외국인']:
        return f'{int(val):+,}'
    if row == '거래금':
        return f'{int(val):,}'
    if row == '보유율':
        return f'{val:.1f}%'
    return str(val)


# ── 1행 : 코드입력 | 종목명 + 시총 ───────────────────
col_input, col_name = st.columns([1, 7])

# 변수 초기화 (col_name 밖에서도 참조 가능하도록)
kk       = ''

with col_input:
    code = st.text_input("종목코드", value="0008T0", label_visibility="collapsed",
                         placeholder="종목코드 입력 (예: 0008T0)")

with col_name:
    if code:
        with st.spinner("종목명 조회 중..."):
            try:
                url  = f'https://finance.naver.com/item/main.naver?code={code}'
                res  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7)
                soup = BeautifulSoup(res.text, "lxml")

                item = ''
                for sel in [
                    "div.wrap_company h2 a",
                    "h2.h_company a",
                    "div.wrap_company h2",
                    "h2.h_company",
                ]:
                    tag = soup.select_one(sel)
                    if tag:
                        item = tag.get_text(strip=True)
                        break

                # 위 셀렉터 모두 실패 시 <title> 태그로 fallback
                if not item:
                    title_tag = soup.find("title")
                    if title_tag:
                        item = title_tag.get_text(strip=True).split(":")[0].strip()

            except Exception as e:
                print(f"[코드→이름] 오류 ({code}): {e}")
                item = ''

        # ── main.naver 파싱 (시가총액 · 구성종목) ──────
        tot = ''
        try:
            url_main = f'https://finance.naver.com/item/main.naver?code={code}'
            res_main = requests.get(url_main, headers={"User-Agent": "Mozilla/5.0"})
            report   = pd.read_html(StringIO(res_main.text))
            tot  = report[5].iloc[0, 1]
            comp = report[3][['구성종목(구성자산)', '구성비중']].dropna()
            comp = comp[comp['구성비중'].str.contains('%', na=False)].head(5).reset_index(drop=True)
            kk   = ', '.join(
                f"{row['구성종목(구성자산)']}({row['구성비중']})"
                for _, row in comp.iterrows()
            )
        except Exception as e:
            print(f"[main 파싱 오류] {e}")

        # 종목명 · 시총
        name_str = item if item else '(종목명 없음)'
        st.markdown(
            f"### {name_str}"
            + (f"&nbsp;&nbsp;<span style='font-size:14px;color:#555;'>시총 {tot}</span>" if tot else ""),
            unsafe_allow_html=True
        )

# ── 2행 : 구성 | 버튼 ────────────────────────
col_comp, col_btn = st.columns([3.5, 2])

with col_comp:
    if kk:
        st.markdown(
            f"<span style='font-size:16px;color:#555;'>구성 : {kk}</span>",
            unsafe_allow_html=True
        )

with col_btn:
    if code and name_str:
        btn = (
            "padding:3px 9px;border:1px solid #bbb;border-radius:4px;"
            "text-decoration:none;font-size:20px;margin:2px 2px 2px 0;white-space:nowrap;"
        )
        url_ggl   = f"https://news.google.com/search?q={quote(name_str)}&hl=ko&gl=KR&ceid=KR:ko"
        url_think = f'https://www.thinkpool.com/item/{code}'
        url_min   = f'https://m.stock.naver.com/fchart/domestic/stock/{code}'
        url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}'
        url_fn    = f"https://wcomp.fnguide.com/?c_id=AA&menu_type=01&cmp_cd={code}"
        url_nv    = f'https://m.stock.naver.com/domestic/stock/{code}/analysis'
        st.markdown(
            f'<div style="text-align:right;">'
            f'<a href="{url_ggl}"   target="_blank" style="{btn}">Google</a>'
            f'<a href="{url_think}" target="_blank" style="{btn}">Think</a>'
            f'<a href="{url_min}"   target="_blank" style="{btn}">chart</a>'
            f'<a href="{url_tr}"    target="_blank" style="{btn}">Tr</a>'
            f'<a href="{url_fn}"    target="_blank" style="{btn}">Fn</a>'
            f'<a href="{url_nv}"    target="_blank" style="{btn}">Nv</a>'
            f'</div>',
            unsafe_allow_html=True
        )

cols1 = st.columns(3)
cols1[0].image(f'https://webchart.thinkpool.com/2021ReNew/CumulationSelling/A{code}.png',
               width='stretch', caption="투자자")
cols1[1].image(f'https://ssl.pstatic.net/imgfinance/chart/item/area/week/{code}.png',
               width='stretch', caption="5일 주가")
cols1[2].image(f'https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png',
                width='stretch', caption="매몰도")

# ── 데이터 테이블 ─────────────────────────────────────
if code:
    with st.spinner("데이터 수집 중..."):
        df = _fetch_naver_frgn_page(code)

    periods = {}
    n = len(df)
    if n >= 5:  periods['1W'] = calc_period(df, 5,  '1W')
    if n >= 21: periods['1M'] = calc_period(df, 21, '1M')
    if n >= 63: periods['3M'] = calc_period(df, 63, '3M')

    rows_label = ['종가','등락률','개인','기관','외국인','보유율','거래금']
    display_10 = df.head(10)

    table = {}
    for _, row in display_10.iterrows():
        d = row['날짜']
        table[d] = {
            '종가'  : int(row['종가']),
            '등락률': row['등락률'],
            '개인'  : int(row['개인']),
            '기관'  : int(row['기관']),
            '외국인': int(row['외국인']),
            '보유율': row['보유율'],
            '거래금': int(row['거래금']),
        }

    for p in ['1W','1M','3M']:
        table[p] = periods[p] if p in periods else {k: None for k in rows_label}

    col_order = list(display_10['날짜']) + ['1W','1M','3M']

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

    for rlab in rows_label:
        html += f'<tr><td class="row-label">{rlab}</td>'
        for col in col_order:
            val  = table[col].get(rlab) if table[col] else None
            text = fmt_cell(val, rlab)
            cls_list = []
            if col in ['1W','1M','3M']:
                cls_list.append('period')
            if col == '1W':
                cls_list.append('sep')
            bg = ''
            if rlab in ['등락률','외국인','기관','개인']:
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    if val > 0:
                        bg = 'background-color:#FFD1DC;'
            html += f'<td class="{" ".join(cls_list)}" style="{bg}">{text}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

col1, col2 = st.columns([3, 2.5])

with col2:
    HLW, HCW = calc_hl(df['종가'].head(5))  if n >= 5  else (None, None)
    HL1M, HC1M  = calc_hl(df['종가'].head(21)) if n >= 21 else (None, None)
    HL3M       = calc_hl(df['종가'].head(63))[0] if n >= 63 else None

    HL = f' HL(W): {HLW}%, HC(W): {HCW}% / HL(M): {HL1M}%, HC(M):{HC1M}% / 3MHL:{HL3M}% '

    if HL:
        st.markdown(
            f"<span style='font-size:20px;color:#555;'>{HL}</span>",
            unsafe_allow_html=True
        )



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

    ###################################################################################
    fig = plt.figure(figsize=(18.5,11)) #14, 7.5
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 0.21, 0.21, 0.21], hspace=0.01)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # x축 공유
    ax3 = fig.add_subplot(gs[2], sharex=ax1) # x축 공유
    ax4 = fig.add_subplot(gs[3], sharex=ax1) # x축 공유

    ax1.set_title(f"{item}")

    ax1.plot(d['Date'], d['Close'], linewidth=1.4, label='Close')
    ax1.plot(d['Date'], d['High'], '--', linewidth=1.0)
    ax1.plot(d['Date'], d['Low'], '--', linewidth=1.0)

    ax1.axhline(max_100, linestyle=':', color = 'black', linewidth=1.0)
    ax1.axhline(min_100, linestyle=':', color ='black', linewidth=1.0)

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
    
    x5_text_pos = dates[(T-7)] ## 1주 위치 30-7 = 23
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
    if last_cross_close_20_date: ax2.plot(last_cross_close_20_date,last_cross_close_20_value,"d",color='magenta',markersize=12)
    if last_cross_close_60_date: ax2.plot(last_cross_close_60_date,last_cross_close_60_value,"d",color='blue',markersize=12)
    if last_cross_close_120_date: ax2.plot(last_cross_close_120_date,last_cross_close_120_value,"d",color='black',markersize=11)
    for j in range(len(d)):
        ax2.axvline(x=d['Date'].iloc[j], color='lightgray', linestyle=':', linewidth=1)
    ax2.tick_params(axis='x',rotation=45,labelsize=1)
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
        label.set_fontsize(6.6)


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

def load_data(code, T=60, N =1):
    try :
        day = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d") #300
        dd = fdr.DataReader(code, day).reset_index()
        # dd = fdr.DataReader(code, '20250101', '20260118').reset_index()
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
        print("실패")
        return None

dfv = load_data(code)
if dfv is None or dfv.empty:
    st.error(f"{item} 데이터 로드 실패")
else:
    fig = showV(item, dfv)
if fig:
    st.pyplot(fig)
    plt.close(fig)   # 메모리 누수 방지용으로 닫아주는 게 좋음
