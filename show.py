import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from scipy.signal import find_peaks


st.set_page_config(page_title="Graph", layout="wide")
st.subheader("📊Graph")

df = pd.read_json('code.json', encoding='utf-8')
if 'selected_item' not in st.session_state:
    st.session_state['selected_item'] = df['item'].iloc[0]

def update_stock():

    new_item = st.session_state['stock_selector']
    selected_row = df[df['item'] == new_item].iloc[0]
    st.session_state['selected_code'] = selected_row['code']
    st.session_state['selected_item'] = new_item

# 3. Selectbox 구성
cool = st.columns([2, 2,2])

# 현재 저장된 이름의 인덱스 찾기
try:
    current_index = df['item'].tolist().index(st.session_state['selected_item'])
except ValueError:
    current_index = 0

item = cool[0].selectbox("Choice", df['item'].tolist(), index=current_index,
    key='stock_selector',  # 세션 키 지정
    on_change=update_stock)# 값이 바뀔 때 즉시 함수 실행 

if 'selected_code' not in st.session_state:
    st.session_state['selected_code'] = df[df['item'] == st.session_state['selected_item']].iloc[0]['code']

code = st.session_state['selected_code']


@st.cache_data(ttl=600)
def showV_plotly(item, code):
    def load_data(code):
        try :
            # day = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d") #300
            # dd = fdr.DataReader(code, day).reset_index()
            dd = fdr.DataReader(code).tail(300).reset_index()
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
            return dd.tail(100).copy()

        except : print(item)

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
    
    d = load_data(code) 
    if d is None or d.empty:
        return None
   
    # 지표 계산 (기존과 동일)
    d_max = d['Close'].max()
    d_min = d['Close'].min()
    dgap = (d_max - d_min) / d_min * 100

###### 1. 서브플롯 생성 (4행 1열) 레이아웃
    fig = make_subplots( rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,
        row_heights=[0.3, 0.28, 0.22, 0.15],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]] )

    # --- Chart 1: Price and Change ---
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['High'], name='High', line=dict(color='red', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Low'], name='Low', line=dict(color='green', width=2, dash='dash')), row=1, col=1)

    # 거래 변동률 (Bar) - secondary_y
    fig.add_trace(go.Bar(x=d['Date'], y=d['Change'], name='Change(%)', marker_color='rgba(150, 150, 150, 0.3)'), row=1, col=1, secondary_y=True)

    d_len = len(d)

    if d_len > 20:
        configs = [
            {'days': 20, 'T': 1, 'm_color': '#FF5733', 'line_color': 'red'},
            {'days': 40, 'T': 20, 'm_color': '#33FF57', 'line_color': 'green'},
            {'days': 60, 'T': 40, 'm_color': '#3357FF', 'line_color': 'blue'},
        ]

        for conf in configs:
            offset = conf['days']
            cha = conf['T']

            if d_len > offset:  
                k = 100 - offset          
                x_pos = d['Date'].iloc[k]
                x_end = d['Date'].iloc[k+5]
                target_price = d['Close'].iloc[k] # 마커가 찍힐 해당 날짜의 종가  

                d_sub = d.iloc[-offset:-cha]
                p_max = d_sub['Close'].max()
                p_min = d_sub['Close'].min() 
                gap_pct = (p_max - p_min) / p_min * 100

                for y_val in [p_max, p_min]:
                    fig.add_shape(type="line",
                        x0=x_pos, y0=y_val, x1=x_end, y1=y_val,
                        line=dict(color=conf['line_color'], width=1, dash="dot"),
                        row=1, col=1
                    )

                # 양방향 화살표 (마커와 동일한 x_pos에 위치)
                fig.add_annotation(
                    x=x_pos, y=p_max, ax=x_pos, ay=p_min,
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    text="", showarrow=True,
                    arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor=conf['line_color'],
                    row=1, col=1
                )
                
                # % 수치 텍스트
                fig.add_annotation(
                    x=x_pos, y= target_price,
                    xref="x1",   # ⭐ 추가 (1번 차트 기준)
                    yref="y1",
                    text=f"{gap_pct:.0f}%",
                    showarrow=False,
                    font=dict(size=18, color=conf['line_color']),
                    bgcolor="white", bordercolor=conf['line_color'],
                    borderwidth=1, borderpad=2,
                    row=1, col=1
                )


    # 3. 전체 Max / Min 텍스트 (차트 맨 왼쪽에 고정)
    fig.add_annotation(x=d['Date'].iloc[1], y=d_max, text=f" Max {int(d_max):,}", 
                    showarrow=False, yanchor="bottom", xanchor="left", font=dict(size=16), row=1, col=1)
    fig.add_annotation(x=d['Date'].iloc[1], y=d_min, text=f" Min {int(d_min):,}", 
                    showarrow=False, yanchor="top", xanchor="left", font=dict(size=16), row=1, col=1)
    fig.add_annotation(
    x=d['Date'].iloc[1], y=(d_max + d_min) / 2, text=f"{dgap:.0f}%", showarrow=False, font=dict(size=18, color='black'),
    bgcolor="yellow", bordercolor='black', borderwidth=1, borderpad=2,    row=1, col=1)

    # --- Chart 2: Moving Averages & Cross Points ---
    ma_colors = {'MA5': 'green', 'MA20': 'magenta', 'MA60': 'blue', 'MA120': 'darkgray'}
    for ma, color in ma_colors.items():
        fig.add_trace(go.Scatter(x=d['Date'], y=d[ma], name=ma, line=dict(color=color, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name=ma, line=dict(color='black', width=1.5, dash='dash')), row=2, col=1)
    # 극점 표시 (Scatter markers)
    peaks, _ = find_peaks(d['Close'])
    valleys, _ = find_peaks(-d['Close'])
    fig.add_trace(go.Scatter(x=d['Date'].iloc[peaks], y=d['Close'].iloc[peaks], mode='markers', marker=dict(color='red', size=14), name='Peaks'), row=2, col=1)
    fig.add_trace(go.Scatter(x=d['Date'].iloc[valleys], y=d['Close'].iloc[valleys], mode='markers', marker=dict(color='purple', size=14), name='Valleys'), row=2, col=1)

    # --- Chart 3: MA Diff Bar ---
    fig.add_trace(go.Scatter(x=d['Date'], y=d['MA5'], name='MA5', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['MA10'], name='MA10', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Bar(
        x=d['Date'], y=d['MA5_d'], 
        name='MA5 Diff',
        marker_color=['royalblue' if val >= 0 else 'salmon' for val in d['MA5_d']],
        opacity=0.6
    ), row=3, col=1, secondary_y=True)

    # --- Chart 4: Angles (S5, S10) ---
    d['S5_detail'] = d['S5'].clip(lower=89.7)
    d['S10_detail'] = d['S10'].clip(lower=89.7)
    
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close_Shadow', line=dict(color='black', width=1), opacity=0.4), row=4, col=1)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['S5_detail'], name='S5 (Angle)', line=dict(color='magenta', dash='dashdot')), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=d['Date'], y=d['S10_detail'], name='S10 (Angle)', line=dict(color='blue', dash='dash')), row=4, col=1, secondary_y=True)

### 크기 레이아웃
    fig.update_layout( height=1000, title_text=f"📊 {item}({code})", showlegend=False, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), )

    fig.update_xaxes( tickangle=-45, tickformat="%m.%d", tickfont=dict(color="black", size=12,family="Arial" ), row=4, col=1 )  # 마지막 subplot 기준
    
    # Y축 범위 조정 (각도 차트)
    fig.update_yaxes(range=[89.68, 90.03], row=4, col=1, secondary_y=True)
    
    return fig

# Streamlit 실행 부분
fig_plotly = showV_plotly(st.session_state['selected_item'], code)

if fig_plotly:
    st.plotly_chart(fig_plotly, width='stretch')
else:
    st.error("데이터를 불러올 수 없습니다.")

ss1, ss2, ss3 = st.columns([1, 1, 1])
with ss1:
    st.link_button(label=' Think', url= f'https://www.thinkpool.com/item/{code}')
with ss2:
    st.link_button(label=' Naver ', url= f'https://m.stock.naver.com/domestic/stock/{code}/analysis')
with ss3:
    st.link_button(label=' Tr', url= f'https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}')
