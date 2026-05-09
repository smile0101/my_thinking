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
import certifi
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

# ─────────────────────────────────────────
# 엑셀 읽기 / 쓰기
# ─────────────────────────────────────────
EXCEL_FILE = 'stock.xlsx'

@st.cache_data
def load_excel():
    df = pd.read_excel(EXCEL_FILE, dtype={'종목코드': str})
    df['기준값'] = df['기준값'].fillna(0).astype(int)
    df['Memo']  = df['Memo'].fillna('')
    if '관심' not in df.columns:
        df['관심'] = 0
    df['관심'] = df['관심'].fillna(0).astype(int)
    return df

def save_excel(df):
    df.to_excel(EXCEL_FILE, index=False)
    st.cache_data.clear()

def save_data(category, stock_name, value):
    st.cache_data.clear()          # 읽기 전에 캐시 먼저 클리어
    df = load_excel()              # 최신 엑셀 읽기
    if category == "ref_prices":
        try:
            value = int(float(value)) if str(value).replace('.', '', 1).isdigit() else 0
        except (ValueError, TypeError):
            value = 0
        df.loc[df['종목명'] == stock_name, '기준값'] = value
    elif category == "memos":
        df.loc[df['종목명'] == stock_name, 'Memo'] = value
    elif category == "interest":
        if '관심' not in df.columns:
            df['관심'] = 0
        df.loc[df['종목명'] == stock_name, '관심'] = int(value)
    save_excel(df)                 # 저장 (내부에서 cache_data.clear() 재실행)
    st.toast(f"'{stock_name}' 저장 완료!", icon="💾")

# ─────────────────────────────────────────
# 통합 수급 함수
# ─────────────────────────────────────────
@st.cache_data(ttl=6000)
def fetch_supply_data(stock_name, stock_code, excel_df_json):
    excel_df = pd.read_json(StringIO(excel_df_json), dtype={'종목코드': str})

    # ── 네이버 수급 데이터 ──────────────────
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
        fk['보유율'] = fk['보유율'].str.replace('%', '').astype(float)

    dk = fk.head(10).reset_index(drop=True)

    # ── 수급 요약 정보 ──────────────────────
    target = excel_df[excel_df['종목코드'] == stock_code].iloc[0]
    m_rank = target['순위']
    amm    = target['시총']

    FO = int((dk['외국인'] > 0).sum())
    GV = int((dk['기관']   > 0).sum())
    IN = int((dk['개인']   > 0).sum())
    FC = dk['보유율'].iloc[0]

    info1 = f"{m_rank}위/ {amm}천억"
    info3 = f"외인:{FO}/기관:{GV}/개인:{IN}(보유:{FC})"

    # ── MongoDB Atlas 연결 ──────────────────
    MONGO_URL = st.secrets["mongo_uri"]
    try:
        with MongoClient(
            MONGO_URL,
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsInsecure=True
        ) as client:
            col   = client.forin.stocks
            db_df = pd.DataFrame(col.find({"종목명": stock_name}, {"_id": 0}))
    except Exception as e:
        st.warning(f"MongoDB 연결 오류: {e}")
        db_df = pd.DataFrame()

    # ── plot용 df 구성 ──────────────────────
    plot_df = dk[['날짜','종가','보유율']].copy()
    plot_df['보유율'] = plot_df['보유율'].astype(str).str.replace('%', '').astype(float)
    plot_df['날짜']   = pd.to_datetime(plot_df['날짜'])
    plot_df['일자']   = plot_df['날짜'].dt.strftime('%m.%d')
    plot_df['종목명'] = stock_name
    plot_df['코드']   = stock_code

    # ── DB 데이터 병합 ──────────────────────
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
df = load_excel()

if 'selected_name' not in st.session_state:
    st.session_state['selected_name'] = df['종목명'].iloc[0]

def update_stock():
    new_name     = st.session_state['stock_selector']
    selected_row = df[df['종목명'] == new_name].iloc[0]
    st.session_state['selected_code'] = selected_row['종목코드']
    st.session_state['selected_name'] = new_name

# ─────────────────────────────────────────
# Selectbox
# ─────────────────────────────────────────
cool = st.columns([2, 1.5, 2.2, 2, 2.5])

try:
    current_index = df['종목명'].tolist().index(st.session_state['selected_name'])
except ValueError:
    current_index = 0

item = cool[0].selectbox(
    "Choice", df['종목명'].tolist(),
    index=current_index,
    key='stock_selector',
    on_change=update_stock
)

if 'selected_code' not in st.session_state:
    st.session_state['selected_code'] = (
        df[df['종목명'] == st.session_state['selected_name']].iloc[0]['종목코드']
    )

code = st.session_state['selected_code']

# ─────────────────────────────────────────
# 주가 데이터
# ─────────────────────────────────────────
@st.cache_data(ttl=600)
def get_stock_data(code):
    return fdr.DataReader(code).tail(60)

ts = get_stock_data(code)

CC = high_1w = high_1m = high_3m = None
low_1w = low_1m = low_3m = None
changes = [0, 0, 0]

if not ts.empty:
    CC       = ts['Close'].iloc[-1]
    high_1w  = ts['Close'].tail(5).max()
    high_1m  = ts['Close'].tail(20).max()
    high_3m  = ts['Close'].max()
    low_1w   = ts['Close'].tail(5).min()
    low_1m   = ts['Close'].tail(20).min()
    low_3m   = ts['Close'].min()
    changes  = [
        ts['Change'].iloc[-1] * 100,
        ts['Change'].iloc[-2] * 100,
        ts['Change'].iloc[-3] * 100
    ]

# ─────────────────────────────────────────
# 통합 수급 데이터 호출
# ─────────────────────────────────────────
excel_df_json = df.to_json()
info1, info3, info4, plot_df = fetch_supply_data(item, code, excel_df_json)

if CC:
    vol_3d = [
        f"{int((ts['Close'].iloc[i] * ts['Volume'].iloc[i]) / 100000000)}억"
        for i in [-5, -4, -3, -2, -1]
    ]
    info2 = " / ".join(vol_3d)
else:
    info2 = "-"

# ─────────────────────────────────────────
# ThinkPool 링크 + 관심 체크박스
# ─────────────────────────────────────────
url = f'https://www.thinkpool.com/item/{code}'
with cool[1]:
    st.markdown(
        f'<a href="{url}" target="_blank" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; text-decoration:none;">Think</a>'
        f' &nbsp;&nbsp; '
        f'<span>{CC}</span>',
        unsafe_allow_html=True
    )

    # 관심 체크박스 5개 (체크 수 = 관심 값)
    current_interest = int(df[df['종목명'] == item].iloc[0].get('관심', 0))

    new_interest = st.select_slider(
    "관심",
    options=[0, 1, 2, 3, 4, 5],
    value=current_interest,
    key=f"interest_{item}" )
    if new_interest != current_interest:
        save_data("interest", item, new_interest)
        st.rerun()
     
    # cols_chk = st.columns(5)
    # new_interest = 0
    # for i in range(5):
    #     checked = cols_chk[i].checkbox(
    #         str(i + 1),
    #         value=(i < current_interest),
    #         key=f"interest_{item}_{i}"
    #     )
    #     if checked:
    #         new_interest = i + 1  # 마지막으로 체크된 번호가 값

    # if new_interest != current_interest:
    #     save_data("interest", item, new_interest)
    #     st.rerun()
    

# ─────────────────────────────────────────
# 등락률 (그제 / 어제 / 오늘)
# ─────────────────────────────────────────
with cool[2]:
    st.markdown(
        f"###### 그제 {color_format(changes[2])} &nbsp;&nbsp;어제 {color_format(changes[1])} &nbsp;&nbsp;오늘 {color_format(changes[0])}",
        unsafe_allow_html=True
    )

with cool[3]:
    st.markdown(f"<p style='font-size:16px;font-weight:bold;'>{info1}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:13px;font-weight:bold;'>{info2}</p>", unsafe_allow_html=True)

with cool[4]:
    st.markdown(
        f'<a href="https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}" target="_blank" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; text-decoration:none;">Tr</a>'
        f' &nbsp;&nbsp; '
        f'<a href="https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?gicode=A{code}" target="_blank" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; text-decoration:none;">Fn</a>'
        f' &nbsp;&nbsp; '
        f'<a href="https://m.stock.naver.com/domestic/stock/{code}/research" target="_blank" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; text-decoration:none;">Nv</a>'
        f' &nbsp;&nbsp; '
        f'<a href="https://www.samsungpop.com/mbw/trading/domesticStock.do?cmd=stockInvestorList" target="_blank" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; text-decoration:none;">투자자</a>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────
# 기준가 + 고저가 메트릭
# ─────────────────────────────────────────
cols = st.columns([1.5, 2, 2, 2, 2, 2, 2, 2])

with cols[0]:
    row       = df[df['종목명'] == item].iloc[0]
    saved_ref = str(row['기준값']) if row['기준값'] != 0 else ""
    ref_input = st.text_input("기준가", value=saved_ref, key=f"ref_{item}")

    if st.button("💾 저장", key=f"btn_ref_{item}"):
        save_data("ref_prices", item, ref_input)

    if CC and ref_input.replace('.', '', 1).isdigit() and float(ref_input) > 0:
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
    co1, _ = st.columns(2)
    co1.link_button('google', f"https://news.google.com/search?q={quote(item)}&hl=ko&gl=KR&ceid=KR:ko")
    co1.link_button('naver',  f"https://finance.naver.com/item/news.naver?code={code}")

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
    display_df['등락률'] = display_df['등락률'].str.replace('%', '').astype(float)
    for col in ['외국인', '기관', '개인']:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce') / 1000

    styled = (display_df.style.hide(axis="index")
        .map(lambda v: 'background-color:#FFD1DC'
             if isinstance(v, (int, float)) and v > 0 else '',
             subset=['등락률','외국인','기관','개인'])
        .format(precision=1)
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    st.markdown(
        f"""<p style="font-size:18px;font-weight:bold;color:#31333F;padding-top:10px;">{info3}</p>""",
        unsafe_allow_html=True
    )

    # 상위 5일 / 하위 5일 분리
    top5 = display_df.iloc[:5]
    bot5 = display_df.iloc[5:]
    sum_labels = ['등락률', '외국인', '기관', '개인']

    summary_data = []
    for title, grp in [("최근", top5), ("이전", bot5)]:
        row = {'구간': title}
        for label in sum_labels:
            row[label] = grp[label].sum()
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data).set_index('구간')

    def color_val(val):
        color = "#0000FF" if val > 0 else "#FF0000" if val < 0 else "#000000"
        return f'color: {color}'

    st.dataframe(
        summary_df.style
            .map(color_val)
            .format("{:,.0f}"),
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
