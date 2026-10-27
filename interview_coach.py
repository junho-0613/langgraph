"""
interview_coach.py - LangChain + Streamlit 면접 준비 에이전트 (단일 파일 버전)

실행:  streamlit run interview_coach.py
       (python interview_coach.py 로 실행해도 자동으로 streamlit run 으로 다시 띄웁니다)

준비:  같은 폴더에 .env 파일을 만들고  OPENAI_API_KEY=sk-...  한 줄을 넣어 주세요.
설치:  pip install langchain langchain-openai langchain-classic streamlit python-dotenv
       (LangChain 0.3.x 환경이면 langchain-classic 은 설치하지 않아도 됩니다)

[A] 에이전트 부분 (원래 agent.py)  -  Tool 3개, system_instruction, ConversationBufferMemory,
                                     create_tool_calling_agent + AgentExecutor, run_interview_coach()
[B] 화면 부분     (원래 app.py)    -  Streamlit 채팅 UI, 사이드바 자소서/채용공고 입력, session_state 대화 기록
"""

import os
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# LangChain 0.3.x 는 langchain 패키지에, 1.x 는 langchain-classic 패키지에
# AgentExecutor / create_tool_calling_agent / ConversationBufferMemory 가 있다.
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.memory import ConversationBufferMemory
except ImportError:  # LangChain 1.x
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_classic.memory import ConversationBufferMemory

# ConversationBufferMemory / AgentExecutor 관련 deprecation 경고만 숨긴다 (과제 요구사항이므로 그대로 사용).
try:
    from langchain_core._api import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# 0. `python interview_coach.py` 로 실행했을 때 자동으로 `streamlit run` 으로 재실행
# --------------------------------------------------------------------------- #
def _ensure_streamlit_runtime() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return
    if get_script_run_ctx(suppress_warning=True) is None:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]])
        sys.exit(0)


_ensure_streamlit_runtime()

st.set_page_config(page_title="면접 준비 에이전트", page_icon="🎯", layout="wide")


# =========================================================================== #
# [A] 에이전트 부분
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 1. 환경 변수 (.env) 로드
# --------------------------------------------------------------------------- #
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY 를 찾을 수 없습니다. "
        "이 파일과 같은 폴더에 `.env` 파일을 만들고 `OPENAI_API_KEY=sk-...` 를 입력한 뒤 다시 실행하세요."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# 2. LLM
# --------------------------------------------------------------------------- #
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)


# --------------------------------------------------------------------------- #
# 3. 시스템 인스트럭션
# --------------------------------------------------------------------------- #
system_instruction = """당신은 10년 경력의 대기업 인사담당자이자 면접 코치입니다.
자소서와 채용공고를 분석해 실제 면접에서 나올 법한 질문을 생성하고
답변에 구체적이고 건설적인 피드백을 제공합니다.
- 예상 질문은 채용공고의 핵심 역량과 자소서 내용을 반영
- 답변 피드백은 STAR 기법 기준으로 평가
- 개선이 필요한 부분은 구체적인 개선 예시와 함께 제시
- 지원자가 자신감을 가질 수 있도록 강점도 함께 언급"""


def _to_text(content) -> str:
    """LLM 응답 content 가 문자열이 아닌 블록 리스트로 올 때도 문자열로 변환한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _ask_llm(prompt: str) -> str:
    """Tool 내부에서 사용하는 LLM 호출 헬퍼. 시스템 인스트럭션(인사담당자 페르소나)을 항상 함께 보낸다."""
    response = llm.invoke(
        [
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt),
        ]
    )
    return _to_text(response.content)


# --------------------------------------------------------------------------- #
# 4. Tools
# --------------------------------------------------------------------------- #
@tool
def analyze_resume_and_job(resume: str, job_posting: str) -> str:
    """자기소개서(resume)와 채용공고(job_posting)를 분석한다.
    채용공고의 핵심 역량·자격 요건, 자소서에 드러난 지원자의 강점과 경험,
    둘 사이의 매칭 정도(강점 / 보완 필요 부분), 면접에서 검증될 가능성이 높은 포인트를 정리해 반환한다.
    사용자가 자소서와 채용공고를 제공하며 분석을 요청하거나, 예상 질문 생성 전에 분석이 필요할 때 사용한다."""
    prompt = f"""다음 자기소개서와 채용공고를 분석해 주세요.

[자기소개서]
{resume}

[채용공고]
{job_posting}

아래 항목으로 정리해 주세요.
1. 채용공고의 핵심 역량·자격 요건 (중요도 순)
2. 자기소개서에 드러난 지원자의 강점과 핵심 경험
3. 채용공고 요구사항과 자소서 내용의 매칭 분석
   - 잘 맞는 부분
   - 부족하거나 보완이 필요한 부분
4. 면접에서 검증될 가능성이 높은 포인트 (면접관이 파고들 만한 부분)"""
    return _ask_llm(prompt)


@tool
def generate_interview_questions(analysis: str) -> str:
    """자소서·채용공고 분석 결과(analysis)를 바탕으로 실제 면접에서 나올 법한 예상 면접 질문 5가지를 생성한다.
    채용공고의 핵심 역량과 자소서 내용을 반영한 질문과 각 질문의 의도를 함께 반환한다.
    사용자가 예상 질문을 요청할 때 사용하며, 분석 결과가 아직 없다면 analyze_resume_and_job 을 먼저 호출한다."""
    prompt = f"""다음 분석 결과를 바탕으로 실제 면접에서 나올 법한 예상 면접 질문 5가지를 만들어 주세요.

[분석 결과]
{analysis}

조건:
- 채용공고의 핵심 역량과 자소서 내용을 반드시 반영할 것
- 질문 유형을 다양하게 구성할 것 (경험·역량 검증, 직무 적합성, 상황 대처, 지원 동기, 약점 보완 등)
- 자소서의 구체적인 경험을 파고드는 질문을 최소 2개 포함할 것
- 각 질문 아래에 "질문 의도"를 한 줄로 덧붙일 것
- 1번부터 5번까지 번호를 붙여 출력할 것"""
    return _ask_llm(prompt)


@tool
def feedback_answer(question: str, answer: str) -> str:
    """면접 질문(question)에 대한 지원자의 답변(answer)을 STAR 기법
    (Situation 상황 / Task 과제 / Action 행동 / Result 결과) 기준으로 평가하고 피드백한다.
    잘한 점(강점), STAR 요소별 평가, 개선이 필요한 부분과 구체적인 개선 예시 문장, 총평을 반환한다.
    사용자가 특정 질문에 대한 자신의 답변을 제시하며 피드백을 요청할 때 사용한다."""
    prompt = f"""다음 면접 질문에 대한 지원자의 답변을 STAR 기법 기준으로 평가해 주세요.

[면접 질문]
{question}

[지원자 답변]
{answer}

STAR 기법 평가 기준:
- S (Situation): 상황·배경이 구체적으로 제시되었는가
- T (Task): 본인의 역할과 해결해야 할 과제가 명확한가
- A (Action): 본인이 실제로 취한 행동이 구체적이고 주도적인가
- R (Result): 결과가 정량적·정성적으로 제시되었고, 배운 점이 있는가

아래 형식으로 작성해 주세요.
1. 잘한 점 (강점) - 지원자가 자신감을 가질 수 있도록 구체적으로
2. STAR 요소별 평가 - 각 요소마다 O / △ / X 와 근거
3. 개선이 필요한 부분 - 각 항목마다 "이렇게 바꿔보세요" 형태의 구체적인 개선 예시 문장 포함
4. 총평 및 점수 (10점 만점)"""
    return _ask_llm(prompt)


tools = [analyze_resume_and_job, generate_interview_questions, feedback_answer]


# --------------------------------------------------------------------------- #
# 5. 프롬프트 / 메모리 / 에이전트
#    Streamlit 은 화면이 바뀔 때마다 이 파일 전체를 다시 실행하므로,
#    에이전트(와 그 안의 ConversationBufferMemory)는 session_state 에 한 번만 만들어 두고 재사용한다.
# --------------------------------------------------------------------------- #
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def build_agent_executor() -> AgentExecutor:
    # 대화 기록 유지 (memory_key 는 프롬프트의 chat_history placeholder 와 일치해야 함)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 터미널에서 Tool 호출 과정을 확인할 수 있음
        handle_parsing_errors=True,
        max_iterations=8,
    )


if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = build_agent_executor()


# --------------------------------------------------------------------------- #
# 6. 외부 호출용 함수
# --------------------------------------------------------------------------- #
def run_interview_coach(user_input: str) -> str:
    """사용자 입력을 받아 에이전트를 실행하고 최종 답변 문자열을 반환한다.
    대화 기록은 ConversationBufferMemory 에 자동으로 누적된다."""
    result = st.session_state.agent_executor.invoke({"input": user_input})
    return _to_text(result["output"])


def reset_memory() -> None:
    """대화 기록(메모리)을 초기화한다. ('대화 초기화' 버튼에서 사용)"""
    st.session_state.agent_executor.memory.clear()


# =========================================================================== #
# [B] 화면 부분 (Streamlit)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 7. session_state 초기화
# --------------------------------------------------------------------------- #
DEFAULT_STATE = {
    "messages": [],         # 화면에 표시할 대화 기록 [{"role": "user"|"assistant", "content": str}]
    "resume": "",           # 저장된 자기소개서
    "job_posting": "",      # 저장된 채용공고
    "docs_sent": None,      # 에이전트에 마지막으로 전달한 (자소서, 채용공고) — 내용이 바뀌었을 때만 다시 첨부
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# 8. 사이드바: 자소서 / 채용공고 입력, 빠른 실행, 대화 초기화
# --------------------------------------------------------------------------- #
quick_input = None  # 빠른 실행 버튼을 누르면 채팅 입력 대신 사용할 메시지

with st.sidebar:
    st.header("📄 지원 서류")
    st.text_area("자기소개서", key="resume", height=220, placeholder="자기소개서 내용을 붙여넣으세요.")
    st.text_area(
        "채용공고",
        key="job_posting",
        height=220,
        placeholder="채용공고(주요 업무·자격 요건·우대 사항)를 붙여넣으세요.",
    )
    docs_ready = bool(st.session_state.resume.strip() and st.session_state.job_posting.strip())

    st.divider()
    st.subheader("⚡ 빠른 실행")
    if st.button("📊 자소서·채용공고 분석", disabled=not docs_ready):
        quick_input = "제 자기소개서와 채용공고를 분석해 주세요."
    if st.button("❓ 예상 면접 질문 5가지 생성", disabled=not docs_ready):
        quick_input = "제 자기소개서와 채용공고를 바탕으로 예상 면접 질문 5가지를 만들어 주세요."
    if not docs_ready:
        st.caption("자기소개서와 채용공고를 모두 입력하면 버튼이 활성화됩니다.")

    st.divider()
    with st.expander("💡 이렇게 물어보세요", expanded=False):
        st.markdown(
            "- **분석**: `자소서랑 채용공고 분석해줘`\n"
            "- **예상 질문**: `예상 면접 질문 5개 만들어줘`\n"
            "- **답변 피드백**: `2번 질문에 이렇게 답할게요. 답변: ...`"
        )

    if st.button("🗑️ 대화 초기화"):
        reset_memory()
        st.session_state.messages = []
        st.session_state.docs_sent = None
        st.rerun()


# --------------------------------------------------------------------------- #
# 9. 메인: 채팅 인터페이스
# --------------------------------------------------------------------------- #
st.title("🎯 면접 준비 에이전트")
st.caption("자소서·채용공고 분석 → 예상 면접 질문 생성 → STAR 기법 답변 피드백")

if not st.session_state.messages:
    st.info("사이드바에 자기소개서와 채용공고를 입력한 뒤, 빠른 실행 버튼을 누르거나 아래 채팅창에 요청을 입력하세요.")

# 이전 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 새 입력 처리
user_input = st.chat_input("예: 1번 질문에 대한 제 답변 피드백해 주세요. 답변: ...") or quick_input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 자소서·채용공고는 처음(또는 내용이 바뀌었을 때)만 메시지에 첨부한다.
    # 한 번 보내면 ConversationBufferMemory 에 남으므로 에이전트가 이후 대화에서도 참고할 수 있다.
    docs = (st.session_state.resume.strip(), st.session_state.job_posting.strip())
    attach_docs = all(docs) and docs != st.session_state.docs_sent
    agent_input = user_input
    if attach_docs:
        agent_input += f"\n\n[자기소개서]\n{docs[0]}\n\n[채용공고]\n{docs[1]}"

    with st.chat_message("assistant"):
        with st.spinner("면접 코치가 답변을 준비하는 중..."):
            try:
                answer = run_interview_coach(agent_input)
                if attach_docs:
                    st.session_state.docs_sent = docs
            except Exception as e:
                answer = f"⚠️ 답변 생성 중 오류가 발생했습니다: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})