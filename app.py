import os
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# 페이지 설정
st.set_page_config(page_icon="♥", page_title="지수", layout="wide")
st.subheader("📊 지수") 

def load_data(code, T=60, N=1):
  try:
    # day = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
    # dd = fdr.DataReader(code, day).reset_index()
    dd = fdr.DataReader(code).tail(200).reset_index()

    if 'index' in dd.columns:
      dd = dd.rename(columns={'index': 'Date'})
    if 'Change' in dd.columns:
      dd['Change'] = round(dd['Change'] * 100, 2)
    else:
      dd['Change'] = round(dd['Close'].pct_change() * 100, 2)

    dd = dd.ffill()

    end_idx = -(N - 1) if N > 1 else None
    start_idx = -(T + N - 1)
    dd['Date'] = pd.to_datetime(dd['Date']).dt.strftime('%m.%d')
    return dd.iloc[start_idx:end_idx].copy()
  except Exception as e:
    print(f'데이터 로드 실패 ({code}): {e}')
    return None


# ── 누적합 기간 컬럼 계산 ─────────────────────────────
def calc_period(df, rows, label):
  sub = df.tail(rows)
  return {
      'Close': int(sub['Close'].mean()),
      'Change': round(sub['Change'].sum(), 1),
  }

def fmt_cell(val, row):
  if val is None or (isinstance(val, float) and pd.isna(val)):
    return ''
  if row == 'Close':
    return f'{int(val):,}'
  if row == 'Change':
    return f'{val:+.1f}%'
  return str(val)

# ── 종목별 표 HTML 생성 ─────────────────────────────
def build_table_html( df):
  periods = {}
  n = len(df)
  if n >= 5: periods['1W'] = calc_period(df, 5, '1W')
  if n >= 10: periods['2W'] = calc_period(df, 10, '2W')
  if n >= 15: periods['3W'] = calc_period(df, 15, '3W')
  if n >= 25: periods['1M'] = calc_period(df, 25, '1M')
  if n >= 50: periods['2M'] = calc_period(df, 50, '2M')

  rows_label = ['Close', 'Change']
  display_10 = df.tail(10)

  table = {}
  for _, row in display_10.iterrows():
    d = row['Date']
    table[d] = {
        'Close': int(row['Close']),
        'Change': row['Change'],
    }

  for p in ['1W', '2W', '3W', '1M', '2M']:
    table[p] = (
        periods[p] if p in periods else {k: None for k in rows_label}
    )

  col_order = list(display_10['Date']) + ['1W', '2W', '3W', '1M', '2M']

#########################################################################

  html = '<table class="etf-table"><thead><tr><th>항목</th>'

  for col in col_order:
    cls = 'class="sep"' if col == '1W' else ''
    html += f'<th {cls}>{col}</th>'
  html += '</tr></thead><tbody>'

  for rlab in rows_label:
    html += f'<tr><td class="row-label">{rlab}</td>'
    for col in col_order:
      val = table[col].get(rlab) if table[col] else None
      text = fmt_cell(val, rlab)
      cls_list = []
      if col in ['1W', '2W', '3W', '1M', '2M']:
        cls_list.append('period')
      if col == '1W':
        cls_list.append('sep')
      bg = ''
      if rlab == 'Change':
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
          if val > 0:
            bg = 'background-color:#FFD1DC;'

      class_attr = (
          f'class="{" ".join(cls_list)}"' if cls_list else ''
      )
      html += f'<td {class_attr} style="{bg}">{text}</td>'
    html += '</tr>'

  html += '</tbody></table>'
  return html

##############################################################################

def plot_trend_subplot(ax, df, close_prices, order_val, item):

    count_map = {1: 5, 2: 4, 3: 3}
    limit_count = count_map.get(order_val, 3)  # 기본값은 3개

    # 1. 저점(Minima) 계산
    minima_indices = argrelextrema(close_prices, np.less, order=order_val)[0]
    if len(minima_indices) >= 2:
        low_indices = sorted(minima_indices, reverse=True)[:limit_count]
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
        high_indices = sorted(maxima_indices, reverse=True)[:limit_count]
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

    # 저점 추세선 및 마커
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
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_yticklabels([])

    sub_title = f"{order_val}.{item} 저점: {latest_low:,.0f} | 고점: {latest_high:,.0f} | Cha: {cha_value}%"
    ax.set_title(sub_title, fontsize=10, pad=10)


def trend(item, code):
    df = fdr.DataReader(code).tail(50).reset_index()
    # df = fdr.DataReader(code, '20260601','20260815').reset_index()
    date_col = df.columns[0]
    df.rename(columns={date_col: "Date", "Close": "Close"}, inplace=True)

    formatted_dates = [f"{d.month}.{d.day}" for d in pd.to_datetime(df["Date"])]
    df["Date_Str"] = formatted_dates

    close_prices = df["Close"].values

    # 한글 폰트 설정
    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
    plt.rcParams["axes.unicode_minus"] = False

    # 1행 3열 서브플롯 생성 (수정된 부분)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    plot_trend_subplot(ax1, df, close_prices, order_val=1, item=item)
    plot_trend_subplot(ax2, df, close_prices, order_val=2, item=item)
    plot_trend_subplot(ax3, df, close_prices, order_val=3, item=item)

    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)

################################################################################################
stock_list = [
    {   "item": "코스피",
        "code": "^KS11",
        "url":  'https://t1.daumcdn.net/media/finance/chart/kr/stock/d/KGG01P.png?',},

    {
        "item": "코스닥",
        "code": "^KQ11",
        "url": "https://t1.daumcdn.net/media/finance/chart/kr/stock/d/QGG01P.png?timestamp=202603021557",},

    {
        "item": "다우지수",
        "code": "DJI",
        "url": "https://ssl.pstatic.net/imgfinance/chart/world/continent/DJI@DJI.png"},

    {
        "item": "나스닥",
        "code": "IXIC",
        "url": "https://ssl.pstatic.net/imgfinance/chart/world/continent/NAS@IXIC.png",}]


for info in stock_list:
    item_name = info["item"]
    item_code = info["code"]
    item_url = info["url"]
    
    # 데이터 로드 (테이블 및 그래프 공용)
    df_data = load_data(item_code, T=60, N=1)
    
    st.subheader(item_name)

    if df_data is not None and not df_data.empty:

        col_left, col_right = st.columns([1, 2], gap="medium")
        
        # 좌측: 이미지 (높이를 350px로 고정하고 박스 안에 맞춤)
        with col_left:
             st.markdown( f""" <div style="width: 100%;">  <img src="{item_url}" style="width: 600px; height: 150px; object-fit: fill;" /> </div> """,
            unsafe_allow_html=True )
        # 우측: 테이블 (동일하게 높이 350px 및 스크롤 지정)
        with col_right:
            table_html = build_table_html(df_data)
            styled_table_html = f"""
            <div style="width: 100%; height: 150px; max-height: 150px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 4px; padding: 5px; box-sizing: border-box;">
                {table_html}
            </div>
            """
            st.markdown(styled_table_html, unsafe_allow_html=True)        
        
        # 2. Trend 그래프 렌더링
        trend(item_name, item_code)

        st.markdown("---")
    else:
        st.warning(f"[{item_name}] ({item_code}) 데이터 실패")

keys = {

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
