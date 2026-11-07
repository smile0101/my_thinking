import FinanceDataReader as fdr
import streamlit as st

# 페이지 설정 (반드시 최상단에 위치)
st.set_page_config(page_title="Report", layout="wide")

# 1. 원본 데이터 정의
t = """
[반도체]
삼성전자	005930
SK하이닉스	000660
LG전자	066570
한미반도체	042700
이수페타시스	007660
삼성전기	009150
대덕전자	353200

[로봇]
두산로보틱스	454910
레인보우로보틱스	277810
에스피지	058610

[조선]
HD현대중공업	329180
한화오션	042660
보성파워텍	006910

[원전]
두산에너빌리티	034020
한전기술	052690

[전력]
HD현대일렉트릭	267260
효성중공업	298040
LSELECTRIC	010120

[전선]
산일전기	062040
대한전선	001440
일진전기	103590
가온전선	000500

[방산]
한화에어로스페이스	012450
LIG넥스원	079550
퍼스텍	010820

[바이오]
삼성바이오로직스	207940
셀트리온	068270
알테오젠	196170
에이비엘바이오	298380
리가켐바이오	141080
HLB	028300

[금융]
KB금융	105560
신한지주	055550
JB금융지주	175330

[뷰티]
에이피알	278470
달바글로벌	483650
파마리서치	214450
휴젤	145020
실리콘투	257720
브이티	018290

[통신]
SK스퀘어	402340
SK텔레콤	017670
KT	030200
[엔터]
하이브	352820
에스엠	041510
와이지엔터테인먼트	122870
[푸드]
삼양식품	003230
빙그레	005180
[전지]
LG에너지솔루션	373220
POSCO홀딩스	005490
삼성SDI	096770
포스코퓨처엠	003670
에코프로비엠	247540
에코프로	086520

[기초자산]
KODEXWTI(H)	261220
KODEXWTI인버스(H)	271050	
KODEX달러레버리지	261250
KODEX달러인버스2X	261260	
TIGER일본엔선물	292560
ACEKRX금현물	411060
KODEX골드인버스(H)	280940
KODEX은선물(H)	144600
TIGER리츠부동산	329200
KODEX건설	117700

[외국]
ACE미빅테크TOP7레버리지(합성)	465610
ACE미빅테크TOP7인버스(합성)	465620
KODEX미휴머노이드로봇	0038A0
TIGER미우주테크	0183J0
SOL팔란티어커버드콜	0040Y0
TIGER차이나전기차레버리지(합성)	456680
TIGER차이나항셍테크레버리지(H)	438320
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

st.title("📊 관심 종목")


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
  st.markdown(f"## [{sector_name}]")

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
            f"**✔️ {name} ({code})** &nbsp; {change_display} &nbsp;"
            f" {mm_display}",
            unsafe_allow_html=True,
        )

        # 씽크풀 차트 이미지 출력
        chart_url = (
            f"https://webchart.thinkpool.com/2021ReNew/stock1day_volume/A{code}.png"
        )
        st.image(chart_url, use_container_width=True)

    st.markdown("---")  # 줄 바꿈 구분선
