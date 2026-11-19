import io
import FinanceDataReader as fdr
import streamlit as st
import yfinance as yf
import mplfinance as mpf

# 페이지 설정 (반드시 최상단에 위치)
st.set_page_config(page_title="Report", layout="wide")
st.markdown(f"### 📊 관심 종목")

# 이미지 URL 딕셔너리 (문법 오류 수정 완료)
chart_urls = {
    '코스피': 'https://ssl.pstatic.net/imgfinance/chart/sise/siseMainKOSPI.png',
    '코스닥':'https://ssl.pstatic.net/imgfinance/chart/sise/siseMainKOSDAQ.png',
    '달러': 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month3/FX_USDKRW.png',
    '투자자(코스피)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSPI.png?',
    '투자자(코스닥)' : 'https://ssl.pstatic.net/imgfinance/chart/sise/trendUitradeDayKOSDAQ.png',
    'WTI' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month3/OIL_CL.png',
    '일본증시': 'https://ssl.pstatic.net/imgfinance/chart/world/month3/NII@NI225.png',
    '인도증시'  : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/INI@BSE30.png',
    '항생증시' : 'https://ssl.pstatic.net/imgfinance/chart/world/month3/HSI@HSI.png',
    'BTC' : 'https://imagechart.upbit.com/d/mini/BTC.png',
    '엔화' : 'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month3/FX_JPYKRW.png',
    '국내금' :'https://ssl.pstatic.net/imgfinance/chart/marketindex/area/month3/CMDT_GD.png',
}

# 딕셔너리 아이템들을 리스트로 변환 (이름, URL 쌍)
items = list(chart_urls.items())

# 3개씩 묶어서 반복문 실행
for i in range(0, len(items), 3):
    # 한 행에 들어갈 3개의 컬럼 생성
    cols = st.columns(3)
    
    # 3개씩 자르기 (마지막 행에 개수가 모자라도 에러가 나지 않도록 슬라이싱 처리됨)
    chunk = items[i:i+3]
    
    for j, (title, url) in enumerate(chunk):
        with cols[j]:
            st.markdown(f"##### {title}")
            st.image(url, use_container_width=True)

#################################################################################

t = """
[관심]
KODEX미국달러선물레버리지	261250
SK텔레콤	017670
SK스퀘어	402340
삼성인버스2XWTIETN	530134
KODEX2차전지산업레버리지	462330
KODEX200타겟위클리커버드콜	498400
SOL팔란티어커버드콜OTM채권혼합	0040Y0
RISE코리아밸류업위클리고정커버드콜	0094M0



"""

def parse_stock_data(text):
  sectors = {}
  current_sector = None
  for line in text.strip().split("\n"):
    line = line.strip()
    if not line:
      continue
    if line.startswith("[") and line.endswith("]"):
      current_sector = line[1:-1]
      sectors[current_sector] = []
    else:
      parts = line.split()
      if len(parts) >= 2:
        name, code = parts[0], parts[1]
        sectors[current_sector].append((name, code))
  return sectors


stock_dict = parse_stock_data(t)

# 3. 등락률에 색상(빨강/파랑)을 입혀주는 HTML 생성 함수
def get_colored_change_str(changes):
  colored_parts = []
  for val in changes:
    if val > 0:
      color = "red"
      text_val = f"+{val:.1f}%"
    elif val < 0:
      color = "blue"
      text_val = f"{val:.1f}%"
    else:
      color = "gray"
      text_val = f"{val:.1f}%"
    colored_parts.append(
        f"<span style='color:{color}; font-weight:bold;'>{text_val}</span>"
    )
  return " / ".join(colored_parts)


# 4. 최근 3거래일 거래대금 형식 함수 ('억'을 맨 마지막에 한 번만 표시)
def get_colored_mm_str(mm_list):
  mm_parts = [f"{val:,}" for val in mm_list]
  joined_mm = " / ".join(mm_parts)
  return f"<span style='color:black; font-weight:bold;'>({joined_mm}억)</span>"


# 5. 섹터별 순회 및 출력
for sector_name, stocks in stock_dict.items():
  st.markdown(f"### [{sector_name}]")

  # 한 줄에 3개씩 배치하기 위해 3개씩 묶어서 처리
  for i in range(0, len(stocks), 3):
    row_stocks = stocks[i : i + 3]
    cols = st.columns(3)  # 항상 3개의 컬럼 생성

    for j, (name, code) in enumerate(row_stocks):
      with cols[j]:
        try:
          # 데이터 수집 및 등락률 계산
          dd = fdr.DataReader(code).tail(5)
          if not dd.empty:
            if "Change" in dd.columns:
              dd["Change"] = round(dd["Change"] * 100, 1)
            else:
              dd["Change"] = round(dd["Close"].pct_change() * 100, 1)

            # 거래대금(억원 단위) 계산 컬럼 추가
            dd["MM"] = (
                (dd["Close"] * dd["Volume"] / 100000000).round(0).astype(int)
            )

            # 최근 3거래일 등락률 추출 및 패딩
            changes = dd["Change"].tail(3).tolist()
            while len(changes) < 3:
              changes.insert(0, 0.0)

            # 최근 3거래일 거래대금 추출 및 패딩
            mm_values = dd["MM"].tail(3).tolist()
            while len(mm_values) < 3:
              mm_values.insert(0, 0)

            # 색상/형식 적용된 HTML 문자열 생성
            change_display = get_colored_change_str(changes)
            mm_display = get_colored_mm_str(mm_values)
          else:
            change_display = "<span style='color:gray;'>데이터 없음</span>"
            mm_display = ""
        except Exception as e:
          change_display = "<span style='color:gray;'>조회 실패</span>"
          mm_display = ""

        # 요청하신 형식: 종목명 + 컬러 등락률 + 검정색 거래대금(맨 끝에만 '억' 표시)
        st.markdown(
            f" ✔️ {name} &nbsp; {change_display} &nbsp;"
            f" {mm_display}",
            unsafe_allow_html=True,
        )

        # 씽크풀 차트 이미지 출력
        chart_url = (
            f"https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png"
        )
        st.image(chart_url, use_container_width=True)
        tv_url = f"https://kr.tradingview.com/chart/Y3Tq45pg/?symbol=KRX%3A{code}"
        st.caption(f"[Chart]({tv_url})")

    st.markdown("---")  # 줄 바꿈 구분선

    tickers = ["PLTR", "NVDA",  "TSLA", "AAPL", "MSFT", "GOOG" ]


@st.cache_data(ttl=600)
def make_chart(ticker):
    df = yf.download(ticker, period="1mo", auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)  # 멀티인덱스 대응
    buf = io.BytesIO()
    mpf.plot(df, type="candle", mav=(5, 10), volume=True, title=f"{ticker}",
             savefig=dict(fname=buf, format="png", dpi=100,
                          bbox_inches="tight", pad_inches=0.05))  # 바깥 여백 제거
    buf.seek(0)
    return buf.getvalue()


items = [(t, make_chart(t)) for t in tickers]

for i in range(0, len(items), 3):
    cols = st.columns(3)
    for col, (title, png) in zip(cols, items[i:i + 3]):
        with col:
            st.image(png, use_container_width=True) 
