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

    # 재무 컬럼 숫자 변환
    for c in ['매출_24','매출_25','매출_26',
              '영익_24','영익_25','영익_26',
              '영익률_24','영익률_25','영익률_26',
              'PER','ROE','유통']:
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
def on_filter_change():
    st.session_state['interest_filter'] = st.session_state['_interest_filter_sel']

def update_stock():
    new_name = st.session_state['stock_selector']
    row = df[df['종목명'] == new_name].iloc[0]
    st.session_state['selected_code'] = row['종목코드']
    st.session_state['selected_name'] = new_name

def on_ref_change():
    name    = st.session_state['selected_name']
    new_val = st.session_state.get(f"ref_{name}", "")
    save_data("ref_prices", name, new_val)

def on_interest_chk():
    """체크박스 클릭 시: 관심 0이면 1로, 1~7이면 +1, 7이면 0으로 순환"""
    name    = st.session_state['selected_name']
    old_val = st.session_state.get(f"_interest_val_{name}", 0)
    chk_now = st.session_state[f"chk_interest_{name}"]
    if chk_now:
        # 체크됨 → 기존값 기준 +1 순환 (최대 7)
        new_val = min(old_val + 1, 7) if old_val < 7 else 1
    else:
        # 체크 해제 → 0
        new_val = 0
    save_data("interest", name, new_val)
    st.session_state[f"_interest_val_{name}"] = new_val

# ─────────────────────────────────────────
# 통합 수급 함수
# ─────────────────────────────────────────
@st.cache_data(ttl=6000)
def fetch_supply_data(stock_name, stock_code, df_json):
    excel_df = pd.read_json(StringIO(df_json), dtype={'종목코드': str})

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(
        f'https://finance.naver.com/item/frgn.naver?code={stock_code}',
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

    dk = fk.head(10).reset_index(drop=True)

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
            col   = client.forin.stocks
            db_df = pd.DataFrame(col.find({"종목명": stock_name}, {"_id": 0}))
    except Exception as e:
        st.warning(f"MongoDB 연결 오류: {e}")
        db_df = pd.DataFrame()

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

    return info1, info3, dk, merged

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
if 'interest_filter' not in st.session_state:
    st.session_state['interest_filter'] = 0

# ═════════════════════════════════════════
# 1단 배열: cool = [2, 1, 2.2, 2, 3]
# ═════════════════════════════════════════
cool = st.columns([2, 1, 2.2, 2, 3])

# ── cool[0]: 관심 필터 / 종목 선택 / 관심 체크박스 ──
with cool[0]:

    # ① 관심 필터 셀렉박스 (0=전체, 1~7)
    filt = st.session_state['interest_filter']
    st.selectbox(
        "관심 필터",
        options=list(range(8)),
        format_func=lambda x: "전체 목록" if x == 0 else f"관심 {x}",
        index=filt,
        key='_interest_filter_sel',
        on_change=on_filter_change,
        label_visibility='collapsed'
    )

    # 필터에 따른 종목 목록 결정
    filt = st.session_state['interest_filter']
    if filt == 0:
        name_list = df['종목명'].tolist()
    else:
        name_list = df[df['관심'] == filt]['종목명'].tolist()
        if not name_list:
            st.caption(f"관심 {filt}인 종목이 없습니다.")
            name_list = df['종목명'].tolist()

    # selected_name이 목록 밖이면 첫 번째로 초기화
    if st.session_state['selected_name'] not in name_list:
        st.session_state['selected_name'] = name_list[0]
        st.session_state['selected_code'] = (
            df[df['종목명'] == name_list[0]].iloc[0]['종목코드']
        )

    try:
        current_index = name_list.index(st.session_state['selected_name'])
    except ValueError:
        current_index = 0

    # ② 종목 선택 셀렉박스
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

    # ③ 관심 체크박스 (작게) — 클릭마다 +1 순환, 해제하면 0
    # row_data는 아래 전역에서 정의 — 여기서는 cur_interest만 직접 조회
    cur_interest = int(df[df['종목명'] == item].iloc[0].get('관심', 0))
    st.session_state[f"_interest_val_{item}"] = cur_interest

    checked = cur_interest > 0
    st.checkbox(
        f"⭐ {cur_interest}" if cur_interest > 0 else "☆ 관심",
        value=checked,
        key=f"chk_interest_{item}",
        on_change=on_interest_chk
    )

code = st.session_state['selected_code']

# 선택 종목 row_data 전역 확정 (item/code 결정 후 한 번만 조회)
row_data = df[df['종목명'] == item].iloc[0]

# ─────────────────────────────────────────
# row_data에서 값 안전하게 읽기 (pandas Series용)
# ─────────────────────────────────────────
def _get(col_name, suffix='', fmt="{:.2f}"):
    """row_data(pandas Series)에서 컬럼 값을 안전하게 포맷팅"""
    if col_name not in row_data.index:
        return '-'
    v = row_data[col_name]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''   # NaN이면 빈칸
    try:
        return fmt.format(float(v)) + suffix
    except Exception:
        return str(v)

# ── cool[1]: 유통 / PER / ROE ──────────────
with cool[1]:
    st.markdown(
        f"""
        <div style="font-size:13px;line-height:2.3;padding-top:4px;">
            <b>유통</b>&nbsp;{_get('유통', '%', '{:.2f}')}<br>
            <b>PER</b>&nbsp;&nbsp;{_get('PER', '', '{:.2f}')}<br>
            <b>ROE</b>&nbsp;&nbsp;{_get('ROE', '%', '{:.2f}')}
        </div>
        """,
        unsafe_allow_html=True
    )

# ── 주가 데이터 (cool[2] 전에 로드) ──────────
@st.cache_data(ttl=600)
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

# 수급 데이터
df_json = df.to_json()
info1, info3, info4, plot_df = fetch_supply_data(item, code, df_json)

if CC:
    vol_3d = [
        f"{int((ts['Close'].iloc[i] * ts['Volume'].iloc[i]) / 100000000)}억"
        for i in [-5, -4, -3, -2, -1]
    ]
    info2 = " / ".join(vol_3d)
else:
    info2 = "-"

# ── cool[2]: 현재가 + 등락률 + 시총순위 + 거래량 ──
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

# ── cool[3]: 링크 버튼 ─────────────────────────
with cool[3]:
    btn = "padding:3px 9px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:12px;margin:2px 2px 2px 0;"
    url_think = f'https://www.thinkpool.com/item/{code}'
    url_tr    = f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}'
    url_fn    = f'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?gicode=A{code}'
    url_nv    = f'https://m.stock.naver.com/domestic/stock/{code}/research'
    url_ggl   = f"https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko"

    st.markdown(
        f'<a href="{url_think}" target="_blank" style="{btn}">Think</a><br>'
        f'<a href="{url_tr}"    target="_blank" style="{btn}">Tr</a>'
        f'<a href="{url_fn}"    target="_blank" style="{btn}">Fn</a>'
        f'<a href="{url_nv}"    target="_blank" style="{btn}">Nv</a><br>'
        f'<a href="{url_ggl}"   target="_blank" style="{btn}">Google</a>',
        unsafe_allow_html=True
    )

# ── cool[4]: 재무 데이터프레임 ─────────────────
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
            .set_properties(**{'text-align': 'right', 'font-size': '12px'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '12px')]},
                {'selector': 'td', 'props': [('text-align', 'right')]},
            ]),
        use_container_width=True,
        height=140,
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
    # 지분율: '/' 기준으로 분리, 빈 항목 제거, 최대 3줄
    jibun_raw = row_data['지분율'] if '지분율' in row_data.index else ''
    if pd.isna(jibun_raw) if isinstance(jibun_raw, float) else False:
        jibun_raw = ''

    parts = [p.strip() for p in str(jibun_raw).split('/') if p.strip()]
    if parts:
        html_lines = '<br>'.join(
            f'<span style="font-size:11px;color:#333;">{p}</span>'
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
# 메모
# ─────────────────────────────────────────
st.subheader("📝 Memo")
saved_memo = df[df['종목명'] == item].iloc[0]['Memo']
memo_val = st.text_area(
    "종목 메모", value=saved_memo, key=f"memo_{item}", height=100
)

if st.button("💾 메모 저장", key=f"btn_memo_{item}"):
    save_data("memos", item, memo_val)
