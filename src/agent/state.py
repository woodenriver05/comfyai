"""
ComfyUI Multi-Agent State 정의

Supervisor → Planner → Retriever → Executor 그래프의 공유 상태.
"""
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class CreativePlan(TypedDict, total=False):
    mood: str       # 전체 분위기 (cinematic, dark, moody 등)
    lighting: str   # 조명 스타일 (low-key, neon accent 등)
    camera: str     # 카메라 움직임/앵글 (slow dolly, close-up 등)
    style: str      # 예술 스타일 (cyberpunk, anime 등)


class TechnicalPlan(TypedDict, total=False):
    resolution: str  # 목표 해상도 (1280x720 등)
    fps: int         # 초당 프레임
    frames: int      # 총 프레임 수
    model: str       # 추천 모델 (wan2.2, flux, sdxl 등)
    steps: int       # 추천 스텝 수
    pipeline: str    # T2V 또는 I2V


class AgentState(TypedDict, total=False):
    # LangGraph Studio 호환 메시지 목록
    messages: Annotated[list[BaseMessage], add_messages]

    # L1 Supervisor
    user_query: str
    intent: Literal["generate", "search", "knowledge", "execute", "unknown"]
    next_node: str

    # L2 Planner
    creative_plan: CreativePlan
    technical_plan: TechnicalPlan
    retrieval_query: str

    # L2 Retriever
    retrieval_results: Dict[str, Any]
    retrieval_grade: Literal["good", "fair", "poor"]
    retrieval_retries: int

    # L2 Architect
    normalized_workflow: Dict[str, Any]  # 경로 치환 완료된 워크플로우 JSON
    architect_changes: List[str]         # 변경 이력 로그
    architect_done: bool

    # L3 Validator
    validation_passed: bool
    validation_issues: List[str]
    validator_done: bool

    # L3 Human-in-the-loop
    human_approved: bool                 # 사용자 최종 승인
    human_feedback: str                  # 사용자 피드백 텍스트
    awaiting_human: bool                 # interrupt 대기 상태

    # Executor
    executor_output: str

    # 에러 로그
    error_log: List[str]
