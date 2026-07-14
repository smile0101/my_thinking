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
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import plotly.graph_objects as go
import FinanceDataReader as fdr

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

            dd = dd.tail(30).copy()
            # ✅ 주말 제거: Date를 문자열로 변환 → Plotly가 카테고리 축으로 처리
            dd['Date'] = pd.to_datetime(dd['Date']).dt.strftime('%m.%d')
            return dd
        except Exception:
            print(stock_name)
            return None

    d = load_data(stock_code)
    if d is None or d.empty:
        return None

    #### 레이아웃
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,
        row_heights=[0.3, 0.32, 0.25, 0.13],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ]
    )

    # Chart 1: Price + Change bar
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['High'],  name='High',  line=dict(color='red',   width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Low'],   name='Low',   line=dict(color='green', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(x=d['Date'], y=d['Change'], name='Change(%)', marker_color='rgba(150,150,150,0.3)'), row=1, col=1, secondary_y=True)

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
        tickangle=-45,
        # ✅ tickformat 제거 (문자열 축이므로 불필요)
        tickfont=dict(color="black", size=12, family="Arial"),
        row=4, col=1
    )
    fig.update_yaxes(range=[89.68, 90.03], row=4, col=1, secondary_y=True)

    return fig

st.divider()
fig_plotly = showV_plotly(item, code)
if fig_plotly:
    st.plotly_chart(fig_plotly, width='stretch')
else:
    st.error("Plotly 차트 데이터를 불러올 수 없습니다.")