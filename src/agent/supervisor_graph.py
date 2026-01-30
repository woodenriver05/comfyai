"""
Phase 3 (Core): Supervisor + Planner + Retriever + Architect + Executor 그래프.

노드 흐름:
  START → supervisor → planner → supervisor → retriever → grade_results
        → supervisor → architect → supervisor → executor → respond → END

- Supervisor: 상태 기반 라우팅 (intent 분류 + 누락 필드 체크)
- Planner: Creative/Technical 계획 + 영어 검색 쿼리 생성
- Retriever: search_workflows 호출
- Grade: signals 기반 품질 판정 (good/fair/poor), poor면 최대 2회 재시도
- Architect: 워크플로우 JSON 경로 정규화 + 모델 매핑 + 하드웨어 최적화
- Executor: 기존 ReAct 에이전트에 계획+검색 결과를 주입하여 실행
"""
from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .state import AgentState, CreativePlan, TechnicalPlan
from .prompts import (
    HUMAN_REVIEW_SUMMARY_PROMPT,
    PLANNER_PROMPT,
    SUPERVISOR_INTENT_PROMPT,
    VALIDATOR_PROMPT,
)
from .tools import search_workflows, get_workflow_info, set_client
from .architect import architect_normalize


# ---------------------------------------------------------------------------
# Pydantic 구조화 출력 스키마
# ---------------------------------------------------------------------------

class IntentOut(BaseModel):
    intent: str = Field(description="generate | search | knowledge | execute | unknown")


class PlanOut(BaseModel):
    creative_mood: str = Field(description="분위기 키워드")
    creative_lighting: str = Field(description="조명 스타일")
    creative_camera: str = Field(description="카메라 움직임/앵글")
    creative_style: str = Field(description="예술 스타일")
    technical_resolution: str = Field(default="1280x720", description="해상도")
    technical_fps: int = Field(default=24, description="FPS")
    technical_frames: int = Field(default=81, description="총 프레임")
    technical_model: str = Field(default="wan2.2", description="모델명")
    technical_steps: int = Field(default=30, description="스텝 수")
    technical_pipeline: str = Field(default="T2V", description="T2V 또는 I2V")
    retrieval_query: str = Field(description="영어 키워드 검색 쿼리")


# ---------------------------------------------------------------------------
# LLM 헬퍼
# ---------------------------------------------------------------------------

_llm_cache = {}


def _get_llm(model_name: str = "gemini-2.5-pro", temperature: float = 0.3):
    key = (model_name, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return _llm_cache[key]


# ---------------------------------------------------------------------------
# 노드 함수들
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """상태 기반 라우팅. 누락된 필드에 따라 next_node를 결정."""

    # 1) user_query 추출 (첫 진입 시)
    if not state.get("user_query"):
        messages = state.get("messages", [])
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if not query:
            return {"next_node": "__end__"}

        # intent 분류
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_INTENT_PROMPT),
            ("human", "{query}"),
        ])
        chain = prompt | llm.with_structured_output(IntentOut)
        try:
            result = chain.invoke({"query": query})
            intent = result.intent if result.intent in ("generate", "search", "knowledge", "execute") else "unknown"
        except Exception:
            intent = "generate"  # 기본값

        return {
            "user_query": query,
            "intent": intent,
            "retrieval_retries": 0,
            "next_node": _decide_next(intent, {}, {}, None, False, False, None, None, 0),
        }

    # 2) 이후 라우팅
    intent = state.get("intent", "generate")
    creative = state.get("creative_plan", {})
    technical = state.get("technical_plan", {})
    grade = state.get("retrieval_grade")
    architect_done = state.get("architect_done", False)
    validator_done = state.get("validator_done", False)
    human_approved = state.get("human_approved")
    executor_out = state.get("executor_output")
    retries = state.get("retrieval_retries", 0)

    next_node = _decide_next(
        intent, creative, technical, grade,
        architect_done, validator_done, human_approved,
        executor_out, retries,
    )
    return {"next_node": next_node}


def _decide_next(
    intent, creative, technical, grade,
    architect_done, validator_done, human_approved,
    executor_out, retries,
) -> str:
    """라우팅 결정 테이블."""
    # executor가 이미 실행됨
    if executor_out:
        return "respond"

    # knowledge/execute/search → 바로 executor
    if intent in ("knowledge", "execute", "search"):
        return "executor"

    # generate: planner → retriever → architect → validator → human_review → executor
    if not creative or not technical:
        return "planner"

    if grade is None:
        return "retriever"

    if grade == "poor" and retries < 2:
        return "retriever"

    # grade가 good/fair이거나 재시도 소진 → architect
    if not architect_done:
        return "architect"

    # architect 완료 → validator
    if not validator_done:
        return "validator"

    # validator 완료 → human_review (승인 대기)
    if human_approved is None:
        return "human_review"

    # human이 피드백을 줬으면 planner로 돌아감 (재계획)
    if human_approved is False:
        return "planner"

    # 모두 통과 → executor
    return "executor"


def planner_node(state: AgentState, model_name: str, temperature: float) -> dict:
    """Creative + Technical 계획 생성."""
    llm = _get_llm(model_name, temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_PROMPT),
        ("human", "{user_query}"),
    ])
    chain = prompt | llm.with_structured_output(PlanOut)

    user_query = state.get("user_query", "")
    try:
        out: PlanOut = chain.invoke({"user_query": user_query})
    except Exception as e:
        return {
            "error_log": [f"PLANNER_ERROR: {e}"],
            "creative_plan": {"mood": "cinematic", "lighting": "natural", "camera": "static", "style": "realistic"},
            "technical_plan": {"resolution": "1280x720", "fps": 24, "frames": 81, "model": "wan2.2", "steps": 30, "pipeline": "T2V"},
            "retrieval_query": user_query,
        }

    creative: CreativePlan = {
        "mood": out.creative_mood,
        "lighting": out.creative_lighting,
        "camera": out.creative_camera,
        "style": out.creative_style,
    }
    technical: TechnicalPlan = {
        "resolution": out.technical_resolution,
        "fps": out.technical_fps,
        "frames": out.technical_frames,
        "model": out.technical_model,
        "steps": out.technical_steps,
        "pipeline": out.technical_pipeline,
    }

    # Planner 결과를 메시지에도 추가 (Studio 시각화용)
    plan_summary = (
        f"[Planner]\n"
        f"Creative: {out.creative_mood} / {out.creative_lighting} / {out.creative_camera} / {out.creative_style}\n"
        f"Technical: {out.technical_resolution}, {out.technical_fps}fps, {out.technical_frames}frames, "
        f"{out.technical_model}, {out.technical_steps}steps, {out.technical_pipeline}\n"
        f"Search query: {out.retrieval_query}"
    )

    return {
        "creative_plan": creative,
        "technical_plan": technical,
        "retrieval_query": out.retrieval_query,
        "messages": [AIMessage(content=plan_summary)],
    }


def retriever_node(state: AgentState) -> dict:
    """search_workflows 호출. 재시도 시 model 필터 제거."""
    query = state.get("retrieval_query") or state.get("user_query", "")
    retries = state.get("retrieval_retries", 0)

    # 기본 파라미터
    params = {
        "query": query,
        "limit": 5,
        "diversify": True,
        "mmr_lambda": 0.6,
        "mmr_pool": 80,
    }

    # 첫 시도: technical plan의 모델로 필터
    if retries == 0:
        tech = state.get("technical_plan", {})
        model_filter = tech.get("model") if isinstance(tech, dict) else None
        if model_filter:
            params["model"] = model_filter

    # 재시도: 쿼리를 user_query로 확장 (더 넓은 검색)
    if retries > 0:
        params["query"] = state.get("user_query", query)

    try:
        result_str = search_workflows.invoke(params)
        data = json.loads(result_str)
    except Exception as e:
        data = {"error": {"code": "RETRIEVER_ERROR", "message": str(e)}, "results": [], "signals": {"no_results": True}}

    return {"retrieval_results": data}


def grade_results_node(state: AgentState) -> dict:
    """검색 결과 품질 판정. signals 기반."""
    results = state.get("retrieval_results", {})
    signals = results.get("signals", {})
    retries = state.get("retrieval_retries", 0)

    if signals.get("no_results") or signals.get("low_confidence"):
        grade = "poor"
    elif signals.get("ambiguous"):
        grade = "fair"
    else:
        grade = "good"

    # 결과를 메시지에 추가 (Studio 시각화용)
    result_list = results.get("results", [])
    count = len(result_list)
    top_score = signals.get("top_score")
    msg = f"[Retriever] {count}개 결과, top_score={top_score}, grade={grade}, retry={retries}"
    if result_list:
        top_names = [r.get("workflow_name", r.get("workflow_id", "?"))[:30] for r in result_list[:3]]
        msg += f"\nTop: {', '.join(top_names)}"

    update = {
        "retrieval_grade": grade,
        "messages": [AIMessage(content=msg)],
    }

    if grade == "poor":
        update["retrieval_retries"] = retries + 1
        # 재시도용 쿼리 확장
        update["retrieval_query"] = state.get("user_query", "")

    return update


def executor_node(state: AgentState) -> dict:
    """기존 ReAct 에이전트를 실행. 계획+검색 결과를 컨텍스트로 주입."""
    from .agent import create_comfyui_agent

    agent = create_comfyui_agent()

    # 컨텍스트 메시지 구성
    context_parts = []

    creative = state.get("creative_plan")
    if creative and isinstance(creative, dict):
        context_parts.append(f"Creative Plan: {json.dumps(creative, ensure_ascii=False)}")

    technical = state.get("technical_plan")
    if technical and isinstance(technical, dict):
        context_parts.append(f"Technical Plan: {json.dumps(technical, ensure_ascii=False)}")

    retrieval = state.get("retrieval_results")
    if retrieval and isinstance(retrieval, dict):
        result_list = retrieval.get("results", [])
        if result_list:
            top_items = []
            for r in result_list[:3]:
                wid = r.get("workflow_id", "?")
                name = r.get("workflow_name", "")
                score = r.get("score", 0)
                top_items.append(f"  - {name} (id={wid}, score={score:.3f})")
            context_parts.append("검색된 워크플로우:\n" + "\n".join(top_items))

    # 메시지 빌드
    enriched = []
    if context_parts:
        enriched.append(SystemMessage(content="[Planner/Retriever 결과]\n" + "\n".join(context_parts)))

    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            enriched.append(msg)

    if not enriched:
        enriched.append(HumanMessage(content=state.get("user_query", "도움이 필요합니다")))

    try:
        result = agent.invoke({"messages": enriched})
        return {
            "messages": result["messages"],
            "executor_output": "done",
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"에이전트 실행 중 오류 발생: {e}")],
            "executor_output": "error",
            "error_log": [f"EXECUTOR_ERROR: {e}"],
        }


def architect_node(state: AgentState) -> dict:
    """워크플로우 JSON 경로 정규화 + 모델 매핑 + 하드웨어 최적화."""
    results = state.get("retrieval_results", {})
    result_list = results.get("results", [])

    if not result_list:
        return {
            "architect_done": True,
            "architect_changes": ["NO_RESULTS: 정규화할 워크플로우 없음"],
            "messages": [AIMessage(content="[Architect] 검색 결과 없음, 정규화 건너뜀")],
        }

    # 상위 결과의 workflow_json을 정규화
    top_result = result_list[0]
    workflow_json = top_result.get("workflow_json")

    if not workflow_json or not isinstance(workflow_json, dict):
        # workflow_json이 없으면 get_workflow_info로 가져오기 시도
        wid = top_result.get("workflow_id")
        if wid:
            try:
                info_str = get_workflow_info.invoke({"workflow_id": wid})
                info_data = json.loads(info_str)
                workflow_json = info_data.get("workflow", {}).get("workflow_json")
            except Exception:
                pass

    if not workflow_json or not isinstance(workflow_json, dict):
        return {
            "architect_done": True,
            "architect_changes": ["NO_JSON: 워크플로우 JSON을 가져올 수 없음"],
            "messages": [AIMessage(content="[Architect] 워크플로우 JSON 없음, 정규화 건너뜀")],
        }

    # architect_normalize 실행
    normalized, changes = architect_normalize(workflow_json)

    msg_parts = [f"[Architect] {len(changes)}건 변경"]
    for c in changes[:5]:
        msg_parts.append(f"  {c}")
    if len(changes) > 5:
        msg_parts.append(f"  ... 외 {len(changes) - 5}건")

    return {
        "normalized_workflow": normalized,
        "architect_changes": changes,
        "architect_done": True,
        "messages": [AIMessage(content="\n".join(msg_parts))],
    }


def validator_node(state: AgentState) -> dict:
    """워크플로우 JSON 구조 검증. LLM 없이 규칙 기반."""
    workflow = state.get("normalized_workflow")
    issues: list[str] = []

    if not workflow or not isinstance(workflow, dict):
        # 워크플로우 없으면 통과 처리 (knowledge/search 등)
        return {
            "validation_passed": True,
            "validation_issues": ["NO_WORKFLOW: 정규화된 워크플로우 없음 — 자동 통과"],
            "validator_done": True,
            "messages": [AIMessage(content="[Validator] 워크플로우 없음 — 검증 건너뜀, 통과 처리")],
        }

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        if not class_type:
            issues.append(f"[{node_id}] class_type 누락")

        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        # 모델 참조 검증: 빈 문자열 체크
        for key in ("ckpt_name", "lora_name", "vae_name", "unet_name", "clip_name", "control_net_name"):
            val = inputs.get(key)
            if val is not None and isinstance(val, str) and val.strip() == "":
                issues.append(f"[{node_id}:{class_type}] {key} 빈 문자열")

        # 파라미터 범위 검증
        for key in ("steps", "cfg"):
            val = inputs.get(key)
            if val is not None and isinstance(val, (int, float)) and val <= 0:
                issues.append(f"[{node_id}:{class_type}] {key}={val} (양수 필요)")

        for key in ("width", "height"):
            val = inputs.get(key)
            if val is not None and isinstance(val, int) and val % 8 != 0:
                issues.append(f"[{node_id}:{class_type}] {key}={val} (8의 배수 필요)")

        # 노드 연결 검증: inputs의 list 참조가 존재하는 노드를 가리키는지
        for key, val in inputs.items():
            if isinstance(val, list) and len(val) == 2:
                ref_id = str(val[0])
                if ref_id not in workflow:
                    issues.append(f"[{node_id}:{class_type}] {key} → 존재하지 않는 노드 {ref_id}")

    passed = len(issues) == 0
    msg_parts = [f"[Validator] {'PASS' if passed else 'FAIL'} — {len(issues)}건 이슈"]
    for issue in issues[:5]:
        msg_parts.append(f"  {issue}")
    if len(issues) > 5:
        msg_parts.append(f"  ... 외 {len(issues) - 5}건")

    return {
        "validation_passed": passed,
        "validation_issues": issues,
        "validator_done": True,
        "messages": [AIMessage(content="\n".join(msg_parts))],
    }


def human_review_node(state: AgentState) -> dict:
    """사용자에게 실행 전 최종 확인 요약을 제공.

    LangGraph interrupt는 Studio/서버 환경에서만 동작하므로,
    여기서는 요약 메시지를 생성하고 awaiting_human 상태를 설정.
    자동 모드(interrupt 미지원)에서는 auto-approve 처리.
    """
    creative = state.get("creative_plan", {})
    technical = state.get("technical_plan", {})
    changes = state.get("architect_changes", [])
    validation = {
        "passed": state.get("validation_passed", False),
        "issues": state.get("validation_issues", []),
    }

    # 상위 검색 결과 요약
    results = state.get("retrieval_results", {})
    result_list = results.get("results", [])
    top_result = "없음"
    if result_list:
        r = result_list[0]
        top_result = f"{r.get('workflow_name', '?')} (id={r.get('workflow_id', '?')})"

    summary = (
        f"[Human Review] 실행 전 최종 확인\n"
        f"Creative: {json.dumps(creative, ensure_ascii=False)}\n"
        f"Technical: {json.dumps(technical, ensure_ascii=False)}\n"
        f"검색 워크플로우: {top_result}\n"
        f"Architect 변경: {len(changes)}건\n"
        f"Validator: {'PASS' if validation['passed'] else 'FAIL'}"
    )

    if validation.get("issues"):
        summary += f" ({len(validation['issues'])}건 이슈)"

    summary += "\n\n이대로 실행할까요? 수정할 부분이 있으면 말씀해주세요."

    # Auto-approve: interrupt 미지원 환경에서는 자동 승인
    # Studio 환경에서는 이 노드 후 interrupt를 걸어 사용자 입력을 대기
    return {
        "human_approved": True,  # auto-approve (interrupt 미지원 시)
        "awaiting_human": False,
        "messages": [AIMessage(content=summary)],
    }


def respond_node(state: AgentState) -> dict:
    """최종 응답 정리. messages는 이미 executor에서 채워짐."""
    return {}


# ---------------------------------------------------------------------------
# 그래프 조립
# ---------------------------------------------------------------------------

def _route_supervisor(state: AgentState) -> str:
    return state.get("next_node", "__end__")


def create_supervisor_graph(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.3,
    rag_url: Optional[str] = None,
):
    """Supervisor + Planner + Retriever + Executor 멀티에이전트 그래프 생성."""
    if rag_url:
        from ..sdk.client import ComfyRagClient
        set_client(ComfyRagClient(base_url=rag_url))

    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", lambda s: planner_node(s, model_name, temperature))
    graph.add_node("retriever", retriever_node)
    graph.add_node("grade_results", grade_results_node)
    graph.add_node("architect", architect_node)
    graph.add_node("validator", validator_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("executor", executor_node)
    graph.add_node("respond", respond_node)

    # 진입점
    graph.set_entry_point("supervisor")

    # Supervisor 조건부 라우팅
    graph.add_conditional_edges("supervisor", _route_supervisor, {
        "planner": "planner",
        "retriever": "retriever",
        "architect": "architect",
        "validator": "validator",
        "human_review": "human_review",
        "executor": "executor",
        "respond": "respond",
        "__end__": END,
    })

    # 고정 엣지
    graph.add_edge("planner", "supervisor")
    graph.add_edge("retriever", "grade_results")
    graph.add_edge("grade_results", "supervisor")
    graph.add_edge("architect", "supervisor")
    graph.add_edge("validator", "supervisor")
    graph.add_edge("human_review", "supervisor")
    graph.add_edge("executor", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
