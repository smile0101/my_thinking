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
import koreanize_matplotlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="기본정보", layout="wide")
st.subheader("📊 Stock")

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
        st.session_state['selected_code'] = (
            df[df['종목명'] == name_list[0]].iloc[0]['종목코드']
        )

    try:
        current_index = name_list.index(st.session_state['selected_name'])
    except ValueError:
        current_index = 0

    # 종목 선택 셀렉박스
    item = st.selectbox(
        "종목 선택",
        name_list,
        index=current_index,
        key='stock_selector',
        on_change=update_stock,
        label_visibility='collapsed'
    )

    if 'selected_code' not in st.session_state:
        st.session_state['selected_code'] = (
            df[df['종목명'] == item].iloc[0]['종목코드']
        )

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

code = st.session_state['selected_code']

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
    eps_val = float(EPS)
    if CC is not None and eps_val != 0:
        PER = round(CC / eps_val, 1)
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
    url_fn    = f'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?gicode=A{code}'
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
st.divider()
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
        index=['Sum']
    )
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

# ─────────────────────────────────────────
# Plotly 차트
# ─────────────────────────────────────────
@st.cache_data(ttl=600)
def showV_plotly(stock_name, stock_code):
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
            return dd.tail(30).copy()
        except Exception:
            print(stock_name)
            return None

    d = load_data(stock_code)
    if d is None or d.empty:
        return None
    d_max = d['Close'].max()
    d_min = d['Close'].min()
    dgap  = (d_max - d_min) / d_min * 100
    
#### 레이아웃
    fig = make_subplots( rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,
        row_heights=[0.3, 0.32, 0.25, 0.13],
        specs=[ [{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}], ] )

    # Chart 1: Price + Change bar
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['High'],  name='High',  line=dict(color='red',   width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Low'],   name='Low',   line=dict(color='green', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(x=d['Date'], y=d['Change'], name='Change(%)', marker_color='rgba(150,150,150,0.3)'), row=1, col=1, secondary_y=True)

    d_len = len(d)
    if d_len > 20:
        configs = [
            {'days': 20, 'T': 1,  'm_color': '#FF5733', 'line_color': 'red'},
            {'days': 40, 'T': 20, 'm_color': '#33FF57', 'line_color': 'green'},
            {'days': 60, 'T': 40, 'm_color': '#3357FF', 'line_color': 'blue'},
        ]
        for conf in configs:
            offset = conf['days']
            cha    = conf['T']
            if d_len > offset:
                k            = 30 - offset
                x_pos        = d['Date'].iloc[k]
                x_end        = d['Date'].iloc[k + 5]
                target_price = d['Close'].iloc[k]
                d_sub  = d.iloc[-offset:-cha]
                p_max  = d_sub['Close'].max()
                p_min  = d_sub['Close'].min()
                gap_pct = (p_max - p_min) / p_min * 100
                for y_val in [p_max, p_min]:
                    fig.add_shape(type="line",
                        x0=x_pos, y0=y_val, x1=x_end, y1=y_val,
                        line=dict(color=conf['line_color'], width=1, dash="dot"),
                        row=1, col=1)
                fig.add_annotation(
                    x=x_pos, y=p_max, ax=x_pos, ay=p_min,
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    text="", showarrow=True,
                    arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor=conf['line_color'],
                    row=1, col=1)
                fig.add_annotation(
                    x=x_pos, y=target_price,
                    xref="x1", yref="y1",
                    text=f"{gap_pct:.0f}%",
                    showarrow=False,
                    font=dict(size=18, color=conf['line_color']),
                    bgcolor="white", bordercolor=conf['line_color'],
                    borderwidth=1, borderpad=2,
                    row=1, col=1)

    fig.add_annotation(x=d['Date'].iloc[1], y=d_max, text=f" Max {int(d_max):,}",
        showarrow=False, yanchor="bottom", xanchor="left", font=dict(size=16), row=1, col=1)
    fig.add_annotation(x=d['Date'].iloc[1], y=d_min, text=f" Min {int(d_min):,}",
        showarrow=False, yanchor="top", xanchor="left", font=dict(size=16), row=1, col=1)
    fig.add_annotation(
        x=d['Date'].iloc[1], y=(d_max + d_min) / 2, text=f"{dgap:.0f}%",
        showarrow=False, font=dict(size=18, color='black'),
        bgcolor="yellow", bordercolor='black', borderwidth=1, borderpad=2, row=1, col=1)

    # Chart 2: MA + peaks/valleys
    ma_colors = {'MA5': 'green', 'MA20': 'magenta', 'MA60': 'blue', 'MA120': 'darkgray'}
    for ma, color in ma_colors.items():
        fig.add_trace(go.Scatter(x=d['Date'], y=d[ma], name=ma, line=dict(color=color, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close2', line=dict(color='black', width=1.5, dash='dash')), row=2, col=1)
    peaks,   _ = find_peaks(d['Close'])
    valleys, _ = find_peaks(-d['Close'])
    fig.add_trace(go.Scatter(x=d['Date'].iloc[peaks],   y=d['Close'].iloc[peaks],   mode='markers', marker=dict(color='red',    size=14), name='Peaks'),   row=2, col=1)
    fig.add_trace(go.Scatter(x=d['Date'].iloc[valleys], y=d['Close'].iloc[valleys], mode='markers', marker=dict(color='purple', size=14), name='Valleys'), row=2, col=1)

    # Chart 3: MA5 / MA10 + MA5 Diff bar
    fig.add_trace(go.Scatter(x=d['Date'], y=d['MA5'],  name='MA5',  line=dict(color='red')),  row=3, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['MA10'], name='MA10', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Bar(
        x=d['Date'], y=d['MA5_d'], name='MA5 Diff',
        marker_color=['royalblue' if v >= 0 else 'salmon' for v in d['MA5_d']],
        opacity=0.6
    ), row=3, col=1, secondary_y=True)

    # Chart 4: Angle (S5, S10)
    d['S5_detail']  = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'],     name='Close_Shadow', line=dict(color='black', width=1), opacity=0.4), row=4, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['S5_detail'],  name='S5 (Angle)',  line=dict(color='magenta', dash='dashdot')), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['S10_detail'], name='S10 (Angle)', line=dict(color='blue',    dash='dash')),    row=4, col=1, secondary_y=True)

###  크기
    fig.update_layout(
        height=950, title_text=f"📊 {stock_name}({stock_code})", showlegend=False,
        template="plotly_white", margin=dict(l=10, r=10, t=25, b=10),
    )
    fig.update_xaxes(
        tickangle=-45, tickformat="%m.%d",
        tickfont=dict(color="black", size=12, family="Arial"),
        row=4, col=1
    )
    fig.update_yaxes(range=[89.68, 90.03], row=4, col=1, secondary_y=True)

    return fig

st.divider()
fig_plotly = showV_plotly(item, code)
if fig_plotly:
    st.plotly_chart(fig_plotly, use_container_width=True)
else:
    st.error("Plotly 차트 데이터를 불러올 수 없습니다.")
