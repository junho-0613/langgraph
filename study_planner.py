"""
study_planner.py — LangGraph + Streamlit AI 학습 플래너 (단일 파일 버전)

설치:  pip install langgraph langchain-openai streamlit python-dotenv requests
실행:  streamlit run study_planner.py

API 키는 이 파일과 같은 폴더의 .env 에서 읽습니다.
    OPENAI_API_KEY=sk-...
    NEWSDATA_API_KEY=pub_...
.env 가 없으면 앱 사이드바의 "API 키 설정"에 직접 입력해도 됩니다.

[PART 1] 에이전트 (LangGraph, Supervisor 패턴)
    START → supervisor_node ─┬─ plan_agent_node ──────┐
                             ├─ news_agent_node ──────┼─→ final_node → END
                             └─ progress_agent_node ──┘
[PART 2] Streamlit 채팅 UI
"""

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional, TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# #############################################################################
# [PART 1] 에이전트
# #############################################################################

# -----------------------------------------------------------------------------
# 1. 환경 변수 로드 (.env)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # 실행 위치(cwd)와 상관없이 파일 옆의 .env를 읽는다

MODEL_NAME = "gpt-4o-mini"
NEWSDATA_URL = "https://newsdata.io/api/1/latest"

_llm_cache: dict[str, ChatOpenAI] = {}


def get_llm() -> ChatOpenAI:
    """모든 에이전트가 공유하는 LLM(Large Language Model, 대규모 언어 모델) 인스턴스."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY가 없습니다. 파일 옆에 .env를 만들어 OPENAI_API_KEY=sk-... 를 넣거나, "
            "사이드바의 'API 키 설정'에 입력해 주세요."
        )
    if key not in _llm_cache:
        _llm_cache[key] = ChatOpenAI(model=MODEL_NAME, temperature=0.3, api_key=key)
    return _llm_cache[key]


# -----------------------------------------------------------------------------
# 2. 상태(State) 정의
# -----------------------------------------------------------------------------
class PlannerState(TypedDict):
    user_input: str        # 사용자가 채팅창에 입력한 메시지
    subject: str           # 과목명
    exam_date: str         # 시험 날짜 (YYYY-MM-DD 문자열)
    study_hours: int       # 하루 학습 가능 시간
    next_agent: str        # supervisor가 결정한 다음 에이전트: "plan" | "news" | "progress"
    plan_result: str       # plan_agent_node 결과
    news_result: str       # news_agent_node 결과
    progress_result: str   # progress_agent_node 결과
    final_answer: str      # final_node가 취합한 최종 답변


# -----------------------------------------------------------------------------
# 3. 날짜 유틸리티
# -----------------------------------------------------------------------------
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def _fmt(d: date) -> str:
    """2026-10-27(화) 형식"""
    return f"{d.strftime('%Y-%m-%d')}({WEEKDAY_KO[d.weekday()]})"


def _parse_date(exam_date_str: str) -> Optional[date]:
    try:
        return date.fromisoformat(exam_date_str)
    except (TypeError, ValueError):
        return None


def _days_left(exam_date_str: str) -> Optional[int]:
    exam = _parse_date(exam_date_str)
    return (exam - date.today()).days if exam else None


def _dday_label(days: Optional[int]) -> str:
    if days is None:
        return "날짜 미확인"
    if days > 0:
        return f"D-{days}"
    if days == 0:
        return "D-Day"
    return f"시험일 {-days}일 경과"


def _build_calendar(exam_date_str: str) -> str:
    """
    오늘부터 시험일까지의 달력을 미리 계산해 프롬프트에 넣는다 (LLM의 날짜·요일 계산 오류 방지).
    - 21일 이하: 하루씩 나열   - 21일 초과: 1주 단위 블록
    """
    exam = _parse_date(exam_date_str)
    today = date.today()
    if exam is None:
        return "(시험 날짜를 해석할 수 없습니다)"
    days = (exam - today).days
    if days <= 0:
        return "(시험일이 오늘이거나 이미 지났습니다)"

    if days <= 21:
        return "\n".join(
            f"- {_fmt(today + timedelta(days=i))}" + ("  ← 시험일" if i == days else "")
            for i in range(days + 1)
        )

    lines, start, week = [], today, 1
    while start < exam:
        end = min(start + timedelta(days=6), exam - timedelta(days=1))
        lines.append(f"- {week}주차: {_fmt(start)} ~ {_fmt(end)}")
        start, week = end + timedelta(days=1), week + 1
    lines.append(f"- 시험일: {_fmt(exam)}")
    return "\n".join(lines)


def _context_block(state: PlannerState) -> str:
    """모든 에이전트 프롬프트에 공통으로 들어가는 학습 조건"""
    days = _days_left(state["exam_date"])
    return (
        f"- 과목: {state['subject']}\n"
        f"- 오늘 날짜: {_fmt(date.today())}\n"
        f"- 시험 날짜: {state['exam_date']} ({_dday_label(days)})\n"
        f"- 하루 학습 가능 시간: {state['study_hours']}시간"
    )


# -----------------------------------------------------------------------------
# 4. Supervisor 노드 — 어떤 에이전트에게 넘길지 결정
# -----------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """Supervisor의 라우팅 결정 (structured output 스키마)"""

    next_agent: Literal["plan", "news", "progress"] = Field(
        description=(
            "plan: 학습 계획·스케줄 수립 / "
            "news: 과목 관련 최신 뉴스·학습 자료 검색 / "
            "progress: 오늘 학습한 내용 기반 진도 체크·피드백"
        )
    )
    reason: str = Field(default="", description="선택 이유 (한 문장)")


SUPERVISOR_PROMPT = """당신은 AI 학습 플래너의 Supervisor입니다.
사용자의 메시지를 읽고 아래 세 에이전트 중 가장 적합한 하나를 선택하세요.

- plan     : 학습 계획, 스케줄, 공부 순서, 시간 배분을 세워 달라는 요청
- news     : 과목과 관련된 최신 뉴스, 학습 자료, 참고 기사, 트렌드를 찾아 달라는 요청
- progress : 오늘(또는 최근) 공부한 내용을 보고하면서 진도 점검, 피드백, 다음에 할 일을 묻는 요청

판단이 애매하면 plan을 선택하세요."""


def _keyword_route(text: str) -> str:
    """LLM 호출이 실패했을 때 쓰는 키워드 기반 대체 라우팅"""
    t = text.lower()
    if any(k in t for k in ["뉴스", "자료", "기사", "최신", "트렌드", "news", "article"]):
        return "news"
    if any(k in t for k in ["오늘", "공부했", "학습했", "끝냈", "완료", "진도", "피드백", "복습했", "풀었"]):
        return "progress"
    return "plan"


def supervisor_node(state: PlannerState) -> dict:
    user_input = state["user_input"]
    try:
        router = get_llm().with_structured_output(RouteDecision)
        decision = router.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=f"과목: {state['subject']}\n사용자 메시지: {user_input}"),
            ]
        )
        next_agent = getattr(decision, "next_agent", None) or _keyword_route(user_input)
    except EnvironmentError:
        raise  # 키 자체가 없으면 그대로 올려서 UI에서 안내
    except Exception:
        next_agent = _keyword_route(user_input)
    return {"next_agent": next_agent}


def route_after_supervisor(state: PlannerState) -> Literal["plan", "news", "progress"]:
    """조건부 엣지: next_agent 값으로 분기"""
    return state["next_agent"]


# -----------------------------------------------------------------------------
# 5. plan_agent_node — 일별 학습 계획 생성
# -----------------------------------------------------------------------------
PLAN_PROMPT = """당신은 시험 대비 학습 계획을 설계하는 전문 학습 코치입니다.
주어진 과목, 시험 날짜, 하루 학습 가능 시간을 바탕으로 현실적이고 구체적인 일별 학습 계획을 세우세요.

작성 규칙
1. 전체 기간을 '개념 학습 → 문제 풀이/적용 → 복습·총정리 → 시험 직전 점검' 흐름으로 설계합니다.
2. 일별 계획은 마크다운 표로 작성합니다. 열: 날짜(요일) | 학습 내용 | 시간 배분
3. 날짜와 요일은 아래 제공된 달력을 그대로 사용하고, 임의로 계산하지 않습니다.
4. 하루 학습량은 '하루 학습 가능 시간'을 넘지 않게 배분합니다. (예: 개념 2h + 문제 1h)
5. 달력이 주차 단위로 주어지면 주차별 목표와 요일 패턴(평일/주말)을 제시합니다.
6. 매주 최소 1회 복습일을 두고, 시험 전날은 가벼운 점검으로 구성합니다.
7. 마지막에 '핵심 전략' 3가지를 짧게 제시합니다.
8. 한국어로 답하고, 불필요한 서론은 생략합니다."""


def plan_agent_node(state: PlannerState) -> dict:
    prompt = (
        f"{_context_block(state)}\n\n"
        f"[달력]\n{_build_calendar(state['exam_date'])}\n\n"
        f"[사용자 요청]\n{state['user_input']}"
    )
    response = get_llm().invoke([SystemMessage(content=PLAN_PROMPT), HumanMessage(content=prompt)])
    return {"plan_result": response.content}


# -----------------------------------------------------------------------------
# 6. news_agent_node — NewsData API로 최신 학습 자료 검색
# -----------------------------------------------------------------------------
NEWS_PROMPT = """당신은 학습 자료 큐레이터입니다.
아래 검색 결과를 보고 해당 과목을 공부하는 학생에게 도움이 되는 자료를 골라 정리하세요.

작성 규칙
1. 자료마다 번호([1], [2] …)를 그대로 유지하고, 제목 / 한 줄 핵심 요약 / 학습에 활용하는 방법을 적습니다.
2. 과목 학습과 관련이 낮은 기사는 제외하거나 '관련성 낮음'으로 짧게만 언급합니다.
3. URL은 작성하지 않습니다. (원문 링크는 별도로 첨부됩니다)
4. 마지막에 '이번 주 참고 포인트'를 2~3줄로 정리합니다.
5. 한국어로 답합니다."""


def _fetch_news(query: str, size: int = 5) -> list[dict]:
    """NewsData API 호출. 실패 시 예외를 던진다."""
    params = {
        "apikey": os.getenv("NEWSDATA_API_KEY", "").strip(),
        "q": query[:100],          # q는 최대 512자 — 과목명은 짧으므로 100자로 안전하게 자름
        "language": "ko,en",       # 무료 플랜은 언어 최대 5개
        "size": size,              # 무료 플랜 기본값 10 (1~50)
    }
    resp = requests.get(NEWSDATA_URL, params=params, timeout=15)
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"NewsData 응답을 해석할 수 없습니다 (HTTP {resp.status_code})")

    if data.get("status") != "success":
        results = data.get("results")
        message = results.get("message") if isinstance(results, dict) else data
        raise RuntimeError(f"NewsData API 오류 (HTTP {resp.status_code}): {message}")
    return data.get("results", []) or []


def news_agent_node(state: PlannerState) -> dict:
    subject = state["subject"]

    if not os.getenv("NEWSDATA_API_KEY", "").strip():
        return {
            "news_result": (
                "⚠️ NEWSDATA_API_KEY가 없어 자료 검색을 할 수 없습니다. "
                "https://newsdata.io 에서 무료 키를 발급받아 .env 또는 사이드바에 입력해 주세요."
            )
        }

    try:
        articles = _fetch_news(subject)
    except Exception as e:
        return {"news_result": f"⚠️ 뉴스 검색 중 오류가 발생했습니다: {e}"}

    if not articles:
        return {
            "news_result": (
                f"'{subject}' 관련 최신 자료를 찾지 못했습니다. "
                "과목명을 더 일반적인 키워드(예: 영어 표기)로 바꿔서 다시 시도해 보세요."
            )
        }

    article_lines = []
    for i, a in enumerate(articles, 1):
        source = a.get("source_name") or a.get("source_id") or "출처 미상"
        desc = (a.get("description") or "요약 없음")[:300]
        article_lines.append(
            f"[{i}] 제목: {a.get('title')}\n"
            f"    출처: {source} | 날짜: {a.get('pubDate')}\n"
            f"    요약: {desc}"
        )

    prompt = (
        f"{_context_block(state)}\n"
        f"- 사용자 요청: {state['user_input']}\n\n"
        f"[검색 결과]\n" + "\n\n".join(article_lines)
    )
    response = get_llm().invoke([SystemMessage(content=NEWS_PROMPT), HumanMessage(content=prompt)])

    # 원문 링크는 LLM 출력에 맡기지 않고 응답 데이터에서 직접 붙인다 (링크 변형 방지)
    link_lines = []
    for i, a in enumerate(articles, 1):
        title = (a.get("title") or "제목 없음").replace("[", "(").replace("]", ")")
        link_lines.append(f"{i}. [{title}]({a.get('link')})")

    return {"news_result": f"{response.content}\n\n**🔗 원문 링크**\n" + "\n".join(link_lines)}


# -----------------------------------------------------------------------------
# 7. progress_agent_node — 진도 체크 및 피드백
# -----------------------------------------------------------------------------
PROGRESS_PROMPT = """당신은 학습 진도를 점검하고 피드백을 주는 학습 코치입니다.
사용자가 오늘 학습한 내용을 보고 아래 형식으로 답하세요.

### 📍 진도 점검
시험까지 남은 기간과 하루 학습 가능 시간을 고려해, 현재 진도가 적절한지(빠름/적정/느림) 근거와 함께 평가합니다.

### ✅ 잘한 점
### ⚠️ 보완할 점
빠진 개념이나 약점이 있다면 구체적인 보완 방법을 제시합니다.

### 🎯 내일 학습 제안
하루 학습 가능 시간 안에서 구체적인 항목과 시간 배분을 제안합니다.

마지막에 한 줄로 짧게 격려합니다. 한국어로, 마크다운 형식으로 답합니다."""


def progress_agent_node(state: PlannerState) -> dict:
    prompt = f"{_context_block(state)}\n\n[오늘 학습한 내용 (사용자 입력)]\n{state['user_input']}"
    response = get_llm().invoke([SystemMessage(content=PROGRESS_PROMPT), HumanMessage(content=prompt)])
    return {"progress_result": response.content}


# -----------------------------------------------------------------------------
# 8. final_node — 결과 취합
# -----------------------------------------------------------------------------
AGENT_LABEL = {
    "plan": "📅 학습 계획 에이전트",
    "news": "📰 학습 자료 에이전트",
    "progress": "📝 진도 체크 에이전트",
}


def final_node(state: PlannerState) -> dict:
    agent = state.get("next_agent", "plan")
    body = {
        "plan": state.get("plan_result", ""),
        "news": state.get("news_result", ""),
        "progress": state.get("progress_result", ""),
    }.get(agent) or "결과를 생성하지 못했습니다. 다시 시도해 주세요."

    days = _days_left(state["exam_date"])
    header = (
        f"**{AGENT_LABEL.get(agent, '에이전트')}** · {state['subject']} · "
        f"시험 {state['exam_date']} ({_dday_label(days)}) · 하루 {state['study_hours']}시간"
    )
    return {"final_answer": f"{header}\n\n---\n\n{body}"}


# -----------------------------------------------------------------------------
# 9. 그래프 구성
# -----------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(PlannerState)

    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("plan_agent_node", plan_agent_node)
    builder.add_node("news_agent_node", news_agent_node)
    builder.add_node("progress_agent_node", progress_agent_node)
    builder.add_node("final_node", final_node)

    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges(
        "supervisor_node",
        route_after_supervisor,
        {
            "plan": "plan_agent_node",
            "news": "news_agent_node",
            "progress": "progress_agent_node",
        },
    )
    # 각 에이전트는 완료 후 final_node로 이동
    builder.add_edge("plan_agent_node", "final_node")
    builder.add_edge("news_agent_node", "final_node")
    builder.add_edge("progress_agent_node", "final_node")
    builder.add_edge("final_node", END)

    return builder.compile()


planner_graph = build_graph()


# -----------------------------------------------------------------------------
# 10. 외부 호출용 함수
# -----------------------------------------------------------------------------
def run_planner(user_input: str, subject: str, exam_date, study_hours: int) -> str:
    """
    user_input  : 사용자의 채팅 메시지
    subject     : 과목명
    exam_date   : datetime.date / datetime.datetime / "YYYY-MM-DD" 문자열
    study_hours : 하루 학습 가능 시간
    return      : 마크다운 형식의 최종 답변 (final_answer)
    """
    if isinstance(exam_date, (date, datetime)):
        exam_date_str = exam_date.strftime("%Y-%m-%d")
    else:
        exam_date_str = str(exam_date)

    initial_state: PlannerState = {
        "user_input": user_input,
        "subject": subject,
        "exam_date": exam_date_str,
        "study_hours": int(study_hours),
        "next_agent": "",
        "plan_result": "",
        "news_result": "",
        "progress_result": "",
        "final_answer": "",
    }
    result = planner_graph.invoke(initial_state)
    return result["final_answer"]


# #############################################################################
# [PART 2] Streamlit 채팅 UI
# #############################################################################
def main() -> None:
    st.set_page_config(page_title="AI 학습 플래너", page_icon="📚", layout="centered")
    today = date.today()

    # ----- 사이드바: 학습 조건 입력 -------------------------------------------
    with st.sidebar:
        st.header("⚙️ 학습 설정")

        subject = st.text_input("과목명", placeholder="예: 선형대수학, 물리화학, 토익")
        exam_date = st.date_input(
            "시험 날짜",
            value=today + timedelta(days=30),
            min_value=today,
            format="YYYY-MM-DD",
        )
        study_hours = st.slider("하루 학습 가능 시간 (시간)", min_value=1, max_value=12, value=3, step=1)

        # 시험까지 남은 날짜 자동 계산
        days_left = (exam_date - today).days
        st.divider()
        if days_left > 0:
            st.metric("시험까지 남은 날짜", f"D-{days_left}")
            st.caption(
                f"시험일: {exam_date.strftime('%Y-%m-%d')} ({WEEKDAY_KO[exam_date.weekday()]}) · "
                f"확보 가능 학습 시간: 총 {days_left * study_hours}시간"
            )
        else:
            st.metric("시험까지 남은 날짜", "D-Day")
            st.caption("오늘이 시험일입니다. 화이팅!")

        # .env에 키가 없을 때만 보이는 대체 입력 (입력값은 현재 실행 중인 앱에서만 사용)
        st.session_state.setdefault("keys_from_sidebar", False)
        missing_key = not os.getenv("OPENAI_API_KEY") or not os.getenv("NEWSDATA_API_KEY")
        if missing_key or st.session_state.keys_from_sidebar:
            st.divider()
            with st.expander("🔑 API 키 설정 (.env가 없을 때)", expanded=not os.getenv("OPENAI_API_KEY")):
                openai_key = st.text_input("OPENAI_API_KEY", type="password")
                news_key = st.text_input("NEWSDATA_API_KEY (선택)", type="password")
                if openai_key.strip():
                    os.environ["OPENAI_API_KEY"] = openai_key.strip()
                    st.session_state.keys_from_sidebar = True
                if news_key.strip():
                    os.environ["NEWSDATA_API_KEY"] = news_key.strip()
                    st.session_state.keys_from_sidebar = True

        st.divider()
        with st.expander("💡 이렇게 물어보세요", expanded=False):
            st.markdown(
                "- **학습 계획**: `시험까지 학습 계획 세워줘`\n"
                "- **학습 자료**: `관련 최신 학습 자료 찾아줘`\n"
                "- **진도 체크**: `오늘 1~3장 개념 정리하고 연습문제 20개 풀었어`"
            )

        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    # ----- 메인: 채팅 인터페이스 ----------------------------------------------
    st.title("📚 AI 학습 플래너")
    st.caption(
        "학습 계획 수립 · 최신 학습 자료 검색 · 진도 체크 — "
        "Supervisor가 요청을 분석해 알맞은 에이전트에게 전달합니다."
    )

    # 대화 기록 유지
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info("사이드바에서 과목·시험 날짜·학습 시간을 설정한 뒤, 아래 채팅창에 요청을 입력하세요.")

    # 이전 대화 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 새 입력 처리
    if prompt := st.chat_input("예: 학습 계획 세워줘 / 최신 학습 자료 찾아줘 / 오늘 ○○ 공부했어"):
        if not subject.strip():
            st.warning("사이드바에 과목명을 먼저 입력해 주세요.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY가 없습니다. .env 파일 또는 사이드바 'API 키 설정'에 입력해 주세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("에이전트가 답변을 준비하는 중..."):
                    try:
                        answer = run_planner(prompt, subject.strip(), exam_date, study_hours)
                    except Exception as e:
                        answer = f"⚠️ 답변 생성 중 오류가 발생했습니다: {e}"
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


def _running_in_streamlit() -> bool:
    """`streamlit run`으로 실행 중인지 확인 (그냥 python으로 실행하면 False)"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        main()
    else:
        print("이 파일은 Streamlit 앱입니다. 터미널에서 다음 명령으로 실행하세요:")
        print("  streamlit run study_planner.py")