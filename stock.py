import os
import requests
import pandas as pd
from io import StringIO
from pymongo import MongoClient
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import FinanceDataReader as fdr
from urllib.parse import quote
import matplotlib.font_manager as fm
import numpy as np
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import matplotlib.font_manager as fm


matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="주식", layout="wide")
st.subheader("📊 Stock")

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
# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def custom_metric(label, main_val, sub_val, delta=None, delta_color="normal"):
    display_main = f"{main_val:,.0f}" if isinstance(main_val, (int, float)) else main_val
    display_sub  = f"{sub_val:,.0f}"  if isinstance(sub_val,  (int, float)) else sub_val
    color = "#31333F"
    if delta and delta != "-":
        if delta_color == "inverse":
            color = "red" if "-" not in str(delta) else "blue"
        else:
            color = "red" if "+" in str(delta) or (isinstance(delta, (int, float)) and delta > 0) else "blue"
    st.markdown(f"""
    <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;min-height:80px;">
        <p style="margin:0;font-size:14px;color:#555;font-weight:bold;">{label}</p>
        <p style="margin:0;line-height:1.2;">
            <span style="font-size:14pt;font-weight:bold;color:#111;">{display_main}</span>
            <sup style="font-size:12pt;color:#777;">({display_sub})</sup>
        </p>
        <p style="margin:0;font-size:20px;color:{color};font-weight:500;">{delta if delta else ""}</p>
    </div>""", unsafe_allow_html=True)

def color_format(val):
    color = "red" if val < 0 else "black"
    return f'<span style="color:{color}">{val:+.1f}%</span>'

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

def format_jibun(jibun_str):
    """'/' 기준으로 분리해 최대 3줄 반환"""
    if not jibun_str or str(jibun_str).strip() == '':
        return ''
    parts = [p.strip() for p in str(jibun_str).split('/') if p.strip()]
    return '\n'.join(parts[:3])

# ─────────────────────────────────────────
# MongoDB 연결
# ─────────────────────────────────────────
def get_mongo_col():
    MONGO_URL = st.secrets["mongo_uri"]
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True)
    return client, client.forin.stock_info

# ─────────────────────────────────────────
# MongoDB 읽기
# ─────────────────────────────────────────
@st.cache_data(ttl=30)
def load_mongo():
    client, col = get_mongo_col()
    with client:
        df = pd.DataFrame(col.find({}, {"_id": 0}))
    if df.empty:
        st.error("MongoDB에 데이터가 없습니다.")
        st.stop()

    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df['기준값']   = df['기준값'].fillna(0).astype(int)
    df['Memo']    = df['Memo'].fillna('')
    if '관심' not in df.columns:
        df['관심'] = 0
    df['관심'] = df['관심'].fillna(0).astype(int)

    for c in ['매출_24','매출_25','매출_26',
              '영익_24','영익_25','영익_26',
              '영익률_24','영익률_25','영익률_26',
              'EPS','ROE','유통']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if '지분율' in df.columns:
        df['지분율'] = df['지분율'].fillna('')

    return df

# ─────────────────────────────────────────
# MongoDB 저장
# ─────────────────────────────────────────
def save_data(category, stock_name, value):
    client, col = get_mongo_col()
    with client:
        if category == "ref_prices":
            try:
                value = int(float(value)) if str(value).replace('.','',1).isdigit() else 0
            except (ValueError, TypeError):
                value = 0
            col.update_one({"종목명": stock_name}, {"$set": {"기준값": value}})
        elif category == "memos":
            col.update_one({"종목명": stock_name}, {"$set": {"Memo": value}})
        elif category == "interest":
            col.update_one({"종목명": stock_name}, {"$set": {"관심": int(value)}})
    st.cache_data.clear()
    st.toast(f"'{stock_name}' 저장!", icon="💾")

# ─────────────────────────────────────────
# 콜백 함수들
# ─────────────────────────────────────────
def update_stock():
    new_name = st.session_state['stock_selector']
    row = df[df['종목명'] == new_name].iloc[0]
    st.session_state['selected_code'] = row['종목코드']
    st.session_state['selected_name'] = new_name

def on_ref_change():
    name    = st.session_state['selected_name']
    new_val = st.session_state.get(f"ref_{name}", "")
    save_data("ref_prices", name, new_val)

def on_interest_pills(stock_name):
    """pills 선택 변경 시 저장 (None = 선택 해제 → 0으로 저장)"""
    val = st.session_state[f"pills_interest_{stock_name}"]
    new_val = int(val) if val is not None else 0
    save_data("interest", stock_name, new_val)

# ─────────────────────────────────────────
# 네이버 수급 페이지 파싱 헬퍼
# ─────────────────────────────────────────
def _fetch_naver_frgn_page(stock_code, page):
    """네이버 외국인 수급 페이지 1개를 DataFrame으로 반환"""
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(
        f'https://finance.naver.com/item/frgn.naver?code={stock_code}&page={page}',
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

    if fk['보유율'].dtype == 'O':
        fk['보유율'] = fk['보유율'].str.replace('%','').astype(float)

    return fk.reset_index(drop=True)

# ─────────────────────────────────────────
# ★ 외인 점유율 (FOCC) 
# ─────────────────────────────────────────
@st.cache_data(ttl=100)
def fetch_supply_data(stock_name, stock_code, df_json):
    excel_df = pd.read_json(StringIO(df_json), dtype={'종목코드': str})

    # ── page=1 (최근 1M) / page=2 (이전 2M) ──────────────
    fk_p1 = _fetch_naver_frgn_page(stock_code, 1)   # 최근 데이터
    fk_p2 = _fetch_naver_frgn_page(stock_code, 2)   # 이전 데이터

    # 기존 로직에서 사용하던 fk → page=1 데이터 기준
    fk = fk_p1.copy()

    dk = fk.head(10).reset_index(drop=True)   # 수급 테이블 표시용 (10행)

    target = excel_df[excel_df['종목코드'] == stock_code].iloc[0]
    m_rank = target['순위']
    amm    = target['시총']

    FO = int((dk['외국인'] > 0).sum())
    GV = int((dk['기관']   > 0).sum())
    IN = int((dk['개인']   > 0).sum())
    FC = dk['보유율'].iloc[0]

    info1 = f"{m_rank}위 / {amm}천억"
    info3 = f"외인:{FO} / 기관:{GV} / 개인:{IN} (보유:{FC})"

    MONGO_URL = st.secrets["mongo_uri"]
    try:
        with MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, tls=True, tlsInsecure=True) as client:
            col   = client.forin.FOCC
            db_df = pd.DataFrame(col.find({"종목명": stock_name}, {"_id": 0}))
    except Exception as e:
        st.warning(f"MongoDB 연결 오류: {e}")
        db_df = pd.DataFrame()

    # plot_df: page=1 기준으로 구성
    plot_df = dk[['날짜','종가','보유율']].copy()
    plot_df['보유율'] = plot_df['보유율'].astype(str).str.replace('%','').astype(float)
    plot_df['날짜']   = pd.to_datetime(plot_df['날짜'])
    plot_df['일자']   = plot_df['날짜'].dt.strftime('%m.%d')
    plot_df['종목명'] = stock_name
    plot_df['코드']   = stock_code

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

    return info1, info3, dk, merged, fk_p1, fk_p2

# ─────────────────────────────────────────
# 보유율 & 종가 그래프
# ─────────────────────────────────────────
def plot_stock_st(df, stock_name):
    set_korean_font()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(df))

    ax1.plot(x, df['보유율'], marker='o', color='royalblue', label='보유율')
    ax1.set_ylabel("보유율 (%)", color='royalblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['일자'], rotation=45)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, df['종가'], linestyle='--', color='crimson',
             marker='s', linewidth=2, label='종가')
    ax2.set_ylabel("종가 (원)", color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title(f"{stock_name} 주가", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────
# 데이터 로드 + 세션 초기화
# ─────────────────────────────────────────
df = load_mongo()

if 'selected_name' not in st.session_state:
    st.session_state['selected_name'] = df['종목명'].iloc[0]

# ═════════════════════════════════════════
# 1단 배열
# ═════════════════════════════════════════
cool = st.columns([2, 1, 2.2, 1.5, 3])

# ── cool[0]: 종목 선택 / 관심 체크박스 7개 ──────────
with cool[0]:
    name_list = df['종목명'].tolist()

    if st.session_state['selected_name'] not in name_list:
        st.session_state['selected_name'] = name_list[0]
        st.session_state['selected_code'] = ( df[df['종목명'] == name_list[0]].iloc[0]['종목코드'] )

    try:
        current_index = name_list.index(st.session_state['selected_name'])
    except ValueError:
        current_index = 0

    # 종목 선택 셀렉박스
    item = st.selectbox( "종목 선택", name_list, index=current_index, key='stock_selector', 
                        on_change=update_stock, label_visibility='collapsed' )

    if 'selected_code' not in st.session_state:
        st.session_state['selected_code'] = ( df[df['종목명'] == item].iloc[0]['종목코드'] )

    cur_interest = int(df[df['종목명'] == item].iloc[0].get('관심', 0))
    default_pill = str(cur_interest) if cur_interest > 0 else None

    st.pills(
        "관심",
        options=["1","2","3","4","5","6","7"],
        default=default_pill,
        key=f"pills_interest_{item}",
        on_change=lambda: on_interest_pills(item),
        label_visibility="collapsed",
    )

code = st.session_state['selected_code']  ##### code

# 선택 종목 row_data 전역 확정
row_data = df[df['종목명'] == item].iloc[0]

# ─────────────────────────────────────────
# row_data에서 값 안전하게 읽기
# ─────────────────────────────────────────
def _get(col_name, suffix='', fmt="{:.2f}"):
    if col_name not in row_data.index:
        return '-'
    v = row_data[col_name]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    try:
        return fmt.format(float(v)) + suffix
    except Exception:
        return str(v)

# ── (cool[2] 전 주가 데이터 ─────────────────
# @st.cache_data(ttl=100)
def get_stock_data(code):
    return fdr.DataReader(code).tail(60)

ts = get_stock_data(code)

CC = high_1w = high_1m = high_3m = None
low_1w = low_1m = low_3m = None
changes = [0, 0, 0]

if not ts.empty:
    CC      = ts['Close'].iloc[-1]
    high_1w = ts['Close'].tail(5).max()
    high_1m = ts['Close'].tail(20).max()
    high_3m = ts['Close'].max()
    low_1w  = ts['Close'].tail(5).min()
    low_1m  = ts['Close'].tail(20).min()
    low_3m  = ts['Close'].min()
    changes = [
        ts['Change'].iloc[-1] * 100,
        ts['Change'].iloc[-2] * 100,
        ts['Change'].iloc[-3] * 100,
    ]

# 수급 데이터  ★ fk_p1, fk_p2 추가 반환 ★
df_json = df.to_json()
info1, info3, info4, plot_df, fk_p1, fk_p2 = fetch_supply_data(item, code, df_json)

if CC:
    vol_3d = [
        f"{int((ts['Close'].iloc[i] * ts['Volume'].iloc[i]) / 100000000)}억"
        for i in [-5, -4, -3, -2, -1]
    ]
    info2 = " / ".join(vol_3d)
else:
    info2 = "-"

try:
    EPS = row_data['EPS']
    if CC is not None and EPS != 0:
        PER = round(CC / EPS, 1)
    else:
        PER = '-'
except (ValueError, TypeError):
    PER = '-'
# ── cool[1]: 유통 / PER / ROE ──────────────────────
with cool[1]:
    st.markdown(
        f"""
        <div style="font-size:16px;line-height:2.3;padding-top:4px;">
            <b>유통</b>&nbsp;{_get('유통', '%', '{:.2f}')}<br>
            <b>PER</b>&nbsp;&nbsp;{PER}<br>
            <b>ROE</b>&nbsp;&nbsp;{_get('ROE', '%', '{:.2f}')}
        </div>
        """,
        unsafe_allow_html=True
    )


# ── cool[2]: 현재가 + 등락률 + 시총순위 + 거래량 ──────
with cool[2]:
    cc_str = f"{CC:,.0f}" if CC else "-"
    st.markdown(
        f"""
        <div style="font-size:13px;line-height:2.0;padding-top:2px;">
            <span style="font-size:17px;font-weight:bold;">{cc_str}</span>
            &nbsp;&nbsp;
            그제 {color_format(changes[2])}
            &nbsp; 어제 {color_format(changes[1])}
            &nbsp; 오늘 {color_format(changes[0])}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:15px;font-weight:bold;margin:2px 0;'>{info1}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:12px;margin:2px 0;color:#555;'>{info2}</p>",
        unsafe_allow_html=True
    )

# ── cool[3]: 링크 버튼 ────────────────────────────
with cool[3]:
    btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:15px;margin:2px 2px 2px 0;"
    url_think = f'https://www.thinkpool.com/item/{code}'
    url_min   = f'https://m.stock.naver.com/fchart/domestic/stock/{code}'
    url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}'
    url_fn    = f"https://wcomp.fnguide.com/?c_id=AA&menu_type=01&cmp_cd={code}"
    url_nv    = f'https://m.stock.naver.com/domestic/stock/{code}/research'
    url_ggl   = f"https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko"

    st.markdown(
        f'<a href="{url_think}" target="_blank" style="{btn}">Think</a>'
        f'<a href="{url_min}" target="_blank" style="{btn}">chart</a><br>'
        f'<a href="{url_tr}"    target="_blank" style="{btn}">Tr</a>'
        f'<a href="{url_fn}"    target="_blank" style="{btn}">Fn</a>'
        f'<a href="{url_nv}"    target="_blank" style="{btn}">Nv</a><br>'
        f'<a href="{url_ggl}"   target="_blank" style="{btn}">Google</a>',
        unsafe_allow_html=True
    )

# ── cool[4]: 재무 데이터프레임 ─────────────────────
with cool[4]:
    fin_df = pd.DataFrame({
        '구분': ['매출', '영익', '익율'],
        '24년': [
            _get('매출_24', fmt='{:.0f}'),
            _get('영익_24', fmt='{:.0f}'),
            _get('영익률_24', fmt='{:.2f}'),
        ],
        '25년': [
            _get('매출_25', fmt='{:.0f}'),
            _get('영익_25', fmt='{:.0f}'),
            _get('영익률_25', fmt='{:.2f}'),
        ],
        '26년': [
            _get('매출_26', fmt='{:.0f}'),
            _get('영익_26', fmt='{:.0f}'),
            _get('영익률_26', fmt='{:.2f}'),
        ],
    }).set_index('구분')

    st.dataframe(
        fin_df.style
            .set_properties(**{'text-align': 'center', 'font-size': '12px'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '12px')]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
            ]),
        use_container_width=True,
        height=115,
    )

# ═════════════════════════════════════════
# 2단 배열: 기준가 + 고저가 + 지분율
# ═════════════════════════════════════════
cols = st.columns([1.5, 2, 2, 2, 2, 2, 2, 2])

with cols[0]:
    saved_ref = str(row_data['기준값']) if row_data['기준값'] != 0 else ""
    ref_input = st.text_input(
        "기준가",
        value=saved_ref,
        key=f"ref_{item}",
        on_change=on_ref_change
    )
    if CC and ref_input.replace('.','',1).isdigit() and float(ref_input) > 0:
        diff  = ((CC - float(ref_input)) / float(ref_input)) * 100
        color = "blue" if diff >= 0 else "red"
        st.markdown(f"{CC - float(ref_input):,.0f} (:{color}[{diff:+.2f}%])")

with cols[1]:
    difh_1w = high_1w - CC
    custom_metric("1주최고", high_1w, difh_1w, f"-{(difh_1w/CC)*100:.1f}%", "inverse")
with cols[2]:
    difl_1w = CC - low_1w
    custom_metric("1주최저", low_1w, difl_1w, f"+{(difl_1w/low_1w)*100:.1f}%")
with cols[3]:
    difh_1m = high_1m - CC
    custom_metric("1달최고", high_1m, difh_1m, f"-{(difh_1m/CC)*100:.1f}%", "inverse")
with cols[4]:
    difl_1m = CC - low_1m
    custom_metric("1달최저", low_1m, difl_1m, f"+{(difl_1m/low_1m)*100:.1f}%")
with cols[5]:
    difh_3m = high_3m - CC
    custom_metric("분기최고", high_3m, difh_3m, f"-{(difh_3m/CC)*100:.1f}%", "inverse")
with cols[6]:
    difl_3m = CC - low_3m
    custom_metric("분기최저", low_3m, difl_3m, f"+{(difl_3m/low_3m)*100:.1f}%")

with cols[7]:
    jibun_raw = row_data['지분율'] if '지분율' in row_data.index else ''
    if pd.isna(jibun_raw) if isinstance(jibun_raw, float) else False:
        jibun_raw = ''

    parts = [p.strip() for p in str(jibun_raw).split('/') if p.strip()]
    if parts:
        html_lines = '<br>'.join(
            f'<span style="font-size:14px;color:#333;">{p}</span>'
            for p in parts[:3]
        )
        st.markdown(
            f'<div style="line-height:2.0;padding-top:6px;">{html_lines}</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────
# 차트 이미지
# ─────────────────────────────────────────
cols1 = st.columns(3)
cols1[0].image(f'https://webchart.thinkpool.com/2021ReNew/CumulationSelling/A{code}.png',
               use_container_width=True, caption="투자자")
cols1[1].image(f'https://ssl.pstatic.net/imgfinance/chart/item/area/week/{code}.png',
               use_container_width=True, caption="5일 주가")
cols1[2].image(f'https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png',
               use_container_width=True, caption="매몰도")

# ─────────────────────────────────────────
# 수급 테이블 + 차트
# ─────────────────────────────────────────

tab1, tab2 = st.columns([1.1, 2])

with tab1:
    display_df = info4[['날짜','종가','등락률','외국인','기관','개인','보유율']].copy()
    display_df['날짜']   = display_df['날짜'].str.slice(5)
    display_df['종가']   = pd.to_numeric(display_df['종가'], errors='coerce').fillna(0).astype(int)
    display_df['등락률'] = display_df['등락률'].str.replace('%','').astype(float)
    for c in ['외국인','기관','개인']:
        display_df[c] = pd.to_numeric(display_df[c], errors='coerce') / 1000

    styled = (display_df.style.hide(axis="index")
        .map(lambda v: 'background-color:#FFD1DC'
             if isinstance(v, (int, float)) and v > 0 else '',
             subset=['등락률','외국인','기관','개인'])
        .format(precision=1)
        .set_properties(**{'text-align':'center'})
        .set_table_styles([
            {'selector':'th','props':[('text-align','center')]},
            {'selector':'td','props':[('text-align','center')]}
        ]))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:15px;font-weight:bold;color:#31333F;padding-top:10px;">{info3}</p>',
        unsafe_allow_html=True
    )

    top5 = display_df.iloc[:5]
    bot5 = display_df.iloc[5:]
    sum_labels = ['등락률','외국인','기관','개인']

    summary_data = []
    for title, grp in [("최근", top5), ("이전", bot5)]:
        r = {'구간': title}
        for label in sum_labels:
            r[label] = grp[label].sum()
        summary_data.append(r)

    summary_df = pd.DataFrame(summary_data).set_index('구간')

    # Sum 행 추가 (최근 + 이전 합계)
    sum_row = pd.DataFrame(
        {col: [summary_df[col].sum()] for col in summary_df.columns},
        index=['Sum'] )
    summary_df = pd.concat([summary_df, sum_row])

    def color_val(val):
        color = "#0000FF" if val > 0 else "#FF0000" if val < 0 else "#000000"
        return f'color: {color}'

    st.dataframe(
        summary_df.style.map(color_val).format("{:,.0f}"),
        use_container_width=True
    )

with tab2:
    plot_stock_st(plot_df, item)

    # ─────────────────────────────────────────
    # ★ 1M / 2M 구간별 수급 요약 DataFrame ★
    # fk_p1 = page=1 (최근 1M), fk_p2 = page=2 (이전 2M)
    # ─────────────────────────────────────────

    def _prepare_summary_df(fk_raw):
        """수급 DataFrame을 display용으로 변환 (단위: 천주, 등락률 %)"""
        df_tmp = fk_raw[['등락률','외국인','기관','개인']].copy()
        # 등락률: '%' 제거 후 float
        df_tmp['등락률'] = (df_tmp['등락률'].astype(str)
                            .str.replace('%','', regex=False)
                            .astype(float))
        for c in ['외국인','기관','개인']:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors='coerce') / 1000
        return df_tmp

    p1_df = _prepare_summary_df(fk_p1)
    p2_df = _prepare_summary_df(fk_p2)

    sum_labels_m = ['등락률','외국인','기관','개인']

    monthly_data = []
    for title, grp in [("1M", p1_df), ("2M", p2_df)]:
        r = {'구간': title}
        for label in sum_labels_m:
            r[label] = grp[label].sum()
        monthly_data.append(r)

    monthly_df = pd.DataFrame(monthly_data).set_index('구간')

    # Sum 행 추가 (1M + 2M 합계)
    sum_row_m = pd.DataFrame(
        {col: [monthly_df[col].sum()] for col in monthly_df.columns},
        index=['Sum']
    )
    monthly_df = pd.concat([monthly_df, sum_row_m])

    def color_val_m(val):
        color = "#0000FF" if val > 0 else "#FF0000" if val < 0 else "#000000"
        return f'color: {color}'

    st.dataframe(
        monthly_df.style.map(color_val_m).format("{:,.0f}"),
        use_container_width=True
    )

# ─────────────────────────────────────────
# 메모
# ─────────────────────────────────────────
st.subheader("📝 Memo")
saved_memo = df[df['종목명'] == item].iloc[0]['Memo']
memo_val = st.text_area(
    "종목 메모", value=saved_memo, key=f"memo_{item}", height=100
)

if st.button("💾 메모 저장", key=f"btn_memo_{item}"):
    save_data("memos", item, memo_val)

####################################################################################################
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
        label.set_fontsize(12)


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
