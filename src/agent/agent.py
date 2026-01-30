"""
ComfyUI RAG Agent - LangGraph ReAct Agent

LangGraph를 사용한 ReAct 패턴 에이전트.
자연어 요청을 받아 RAG 도구들을 조합하여 워크플로우 검색/실행을 수행한다.
"""
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import get_tools, set_client

SYSTEM_PROMPT = """당신은 ComfyUI 워크플로우 전문가 에이전트입니다.
사용자의 요청을 분석하여 적절한 워크플로우를 찾고, 필요하면 파라미터를 조정하여 실행합니다.

## 사용 가능한 도구
- search_workflows: 워크플로우 검색 (이미지/비디오 생성). 필터: model, tags, status, date_from/to
- search_knowledge: 기술 문서/지식 검색 (아키텍처, 개발 이력 등)
- get_workflow_info: 특정 워크플로우 상세 정보 조회
- push_workflow: 워크플로우를 PC 실행 대기열에 추가 (dry_run 지원)
- check_outbox: 전송 대기열 상태 확인

## 도구 응답 형식
모든 도구는 JSON 문자열을 반환합니다. 반드시 파싱해서 results/meta/signals/error를 확인하세요.

## 작업 흐름
1. 사용자 요청을 이해하고, 적절한 검색 쿼리를 구성
2. search_workflows로 후보 워크플로우를 검색
3. 필요시 get_workflow_info로 상세 정보 확인
4. 사용자에게 결과를 설명하고, 실행 여부 확인
5. push_workflow(dry_run=true)로 검증 후, 승인 시 dry_run=false로 실행

## 플래닝 단계
- Creative Plan: 무드/조명/카메라/스타일 키워드 정리
- Technical Plan: 해상도, FPS, 프레임 수, 모델, 스텝을 결정
- Creative와 Technical을 분리해 요약하고, 부족한 정보는 질문으로 보완

## 판단 기준 (signals 기반)
- signals.low_confidence == true 이면 push 금지. 아래 3가지를 질문:
  1. T2V(텍스트→비디오) / I2V(이미지→비디오) 중 무엇인가요?
  2. 목표 해상도와 길이(프레임 수)는?
  3. 선호하는 모델 계열(Wan 2.2 / Flux / SDXL)?
- signals.ambiguous == true 이면 상위 2개 후보의 차이를 비교 설명하고, "어떤 것을 선택하시겠어요?" 등으로 사용자에게 선택을 유도
- error가 있으면 추측하지 말고 error.code와 message를 요약해서 안내

## 실행 전 확인 (guardrail)
- push_workflow는 항상 dry_run=true부터 호출
- dry_run 결과를 사용자에게 보여주고 승인 후에만 dry_run=false로 실제 실행
- 사용자가 명시적으로 '진행/실행' 의사를 표현했을 때만 실제 push

## 검증 루프 (max_retries=2)
- dry_run에서 error가 있으면 같은 작업을 최대 2회까지만 재시도
- 2회 실패 시 즉시 Human-in-the-loop으로 에스컬레이션

## 주의사항
- 검색 시 영어 키워드가 더 정확한 결과를 반환합니다
- 기술적 질문에는 search_knowledge를 활용합니다
- 한국어로 응답합니다
"""


def create_comfyui_agent(
    model_name: str = "gemini-2.5-pro",
    rag_url: str = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
):
    """
    ComfyUI RAG 에이전트를 생성한다.

    인증: Google AI Studio API 키 사용.
    - 환경변수 GOOGLE_API_KEY 또는 api_key 인자를 사용

    Args:
        model_name: Gemini 모델명
        rag_url: RAG API URL (기본: 환경변수 RAG_API_URL 또는 localhost:7001)
        temperature: LLM temperature
        api_key: Google AI Studio API 키 (선택)

    Returns:
        LangGraph CompiledGraph (invoke/stream 가능)
    """
    if rag_url:
        from ..sdk.client import ComfyRagClient
        set_client(ComfyRagClient(base_url=rag_url))

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
    )

    agent = create_react_agent(
        model=llm,
        tools=get_tools(),
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run(query: str, **kwargs) -> str:
    """
    편의 함수: 에이전트를 생성하고 쿼리를 실행한다.

    Args:
        query: 사용자 질의
        **kwargs: create_comfyui_agent에 전달할 인자

    Returns:
        에이전트의 최종 응답 텍스트
    """
    agent = create_comfyui_agent(**kwargs)
    result = agent.invoke({"messages": [("user", query)]})

    messages = result["messages"]
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            content = msg.content
            # 리스트 형태(멀티모달 등)인 경우 텍스트만 추출하여 결합
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts)
            return content

    return "에이전트가 응답을 생성하지 못했습니다."
