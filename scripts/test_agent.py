"""
ComfyUI RAG Agent - Smoke Test (Phase 3.5: JSON Structured & Guardrails)

사전 조건:
1. RAG API 서버 실행 중: uvicorn src.api.main:app --port 7001
2. GOOGLE_API_KEY 환경변수 설정 (Google AI Studio API Key)
3. pip install langchain langchain-google-genai langgraph
"""
import json
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)


def _parse_json(result: str) -> dict:
    """도구 반환 JSON을 파싱하고 구조를 검증"""
    try:
        data = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"FAILED to parse JSON: {result}")
        raise e
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    return data

def _check_guardrails(query: str, response_text: str) -> None:
    """signals 기반 가드레일 준수 여부를 간단히 점검 (스모크용)"""
    from src.agent.tools import search_workflows
    try:
        raw = search_workflows.invoke({"query": query, "limit": 3})
        data = _parse_json(raw)
    except Exception as e:
        print(f"  [WARN] Guardrail check skipped: {e}")
        return
    if data.get("error"):
        print(f"  [WARN] Guardrail check skipped: API error {data['error']}")
        return
    signals = data.get("signals", {})

    if signals.get("no_results") or signals.get("low_confidence"):
        clarification_hints = ["T2V", "I2V", "해상도", "모델", "선택", "확인", "어떤", "원하"]
        hits = sum(1 for k in clarification_hints if k in response_text)
        if hits < 2:
            print(f"  [WARN] Low confidence but agent did not ask clarifying questions (hits={hits})")

    if signals.get("ambiguous"):
        choice_hints = ["선택", "차이", "비교", "후보", "추천", "번째", "각각", "반면", "vs", "또는"]
        if not any(h in response_text for h in choice_hints):
            print("  [WARN] Ambiguous results but agent did not prompt user choice")



def test_tools_directly():
    """도구가 JSON 구조를 올바르게 반환하고, 핵심 가드레일이 작동하는지 단위 테스트"""
    print("=" * 60)
    print("TEST 1: Tool Direct Invocation & Guardrails")
    print("=" * 60)

    from src.agent.tools import (
        search_workflows,
        search_knowledge,
        get_workflow_info,
        push_workflow,
        check_outbox,
    )

    # [1] 워크플로우 검색 (기본)
    print("\n[1] search_workflows('video generation')")
    result = search_workflows.invoke({"query": "video generation", "limit": 3, "diversify": True})
    data = _parse_json(result)
    
    assert "results" in data, "Missing 'results' key"
    assert "meta" in data, "Missing 'meta' key"
    assert "signals" in data, "Missing 'signals' key"
    assert data.get("error") is None, f"Unexpected error: {data.get('error')}"
    assert "no_results" in data["signals"], "Missing no_results signal"
    if data["signals"].get("no_results"):
        assert data["signals"].get("low_confidence") is True
    print(f"  results: {len(data['results'])} items")
    print(f"  signals: {data['signals']}")
    print("  [OK] JSON structure valid")

    target_workflow_id = None
    if data["results"]:
        target_workflow_id = data["results"][0].get("workflow_id")

    # [1b] 워크플로우 검색 (MMR + 필터 파라미터 검증)
    print("\n[1b] search_workflows with MMR + filter params (Validation)")
    mmr_lambda = 0.3
    mmr_pool = 80
    result = search_workflows.invoke({
        "query": "video generation",
        "limit": 3,
        "diversify": True,
        "mmr_lambda": mmr_lambda,
        "mmr_pool": mmr_pool,
    })
    data = _parse_json(result)
    
    # 서버가 파라미터를 제대로 받았는지 meta 검증
    assert "total_raw" in data["meta"], "Missing total_raw in meta"
    filters = data["meta"].get("filters", {})
    assert filters.get("mmr_lambda") == mmr_lambda, f"mmr_lambda mismatch: expected {mmr_lambda}, got {filters.get('mmr_lambda')}"
    assert filters.get("mmr_pool") == mmr_pool, f"mmr_pool mismatch: expected {mmr_pool}, got {filters.get('mmr_pool')}"
    
    signals = data["signals"]
    assert "low_confidence" in signals
    assert "ambiguous" in signals
    print(f"  meta.filters: {filters}")
    print("  [OK] MMR Parameters reflected in Meta")

    # [1c] Guardrail Test: get_workflow_info & push_workflow(dry_run)
    if target_workflow_id:
        print(f"\n[1c] Guardrail Test with ID: {target_workflow_id}")
        
        # Detail Info
        print("  - get_workflow_info...")
        res_info = get_workflow_info.invoke({"workflow_id": target_workflow_id})
        data_info = _parse_json(res_info)
        assert "workflow" in data_info, "Missing 'workflow' in info response"
        assert data_info["workflow"]["workflow_id"] == target_workflow_id
        print("    [OK] Info retrieved")

        # Push Dry Run
        print("  - push_workflow(dry_run=True)...")
        res_push = push_workflow.invoke({"workflow_id": target_workflow_id, "dry_run": True})
        data_push = _parse_json(res_push)
        
        assert data_push.get("ok") is True, "Dry run failed"
        assert data_push.get("dry_run") is True, "dry_run flag mismatch in response"
        assert "preview" in data_push, "Missing preview in dry run"
        print("    [OK] Dry run validated (Safe)")
    else:
        print("\n[1c] SKIP: No workflows found to test detail/push")

    # [2] 지식 검색
    print("\n[2] search_knowledge('system architecture')")
    result = search_knowledge.invoke({"query": "system architecture", "limit": 3})
    data = _parse_json(result)
    assert "results" in data
    assert "signals" in data, "Missing signals in knowledge search"
    assert "no_results" in data["signals"], "Missing no_results in knowledge signals"
    print(f"  results: {len(data['results'])} items")
    print("  [OK] Knowledge search JSON valid")

    # [3] outbox 확인
    print("\n[3] check_outbox()")
    result = check_outbox.invoke({"limit": 5})
    data = _parse_json(result)
    assert "items" in data
    print(f"  items: {len(data['items'])} items")
    print("  [OK] Outbox JSON valid")

    print("\n--- Tool Direct Test PASSED ---\n")


def test_agent_scenario_1():
    """시나리오 1: 프롬프트 기반 워크플로우 검색"""
    print("=" * 60)
    print("TEST 2: Agent Scenario - Workflow Search")
    print("=" * 60)

    from src.agent.agent import run

    query = "사이버펑크 스타일 영상을 만들 수 있는 워크플로우를 찾아줘"
    print(f"\nQuery: {query}")
    print("-" * 40)

    response = run(query)
    print(f"\nAgent Response:\n{response}")
    _check_guardrails(query, response)
    
    # 응답 검증
    assert response is not None, "Agent response is None"
    assert len(response.strip()) > 10, "Agent response is too short (empty?)"
    print("\n  [OK] Response received")
    print("\n--- Scenario 1 PASSED ---\n")


def test_agent_scenario_2():
    """시나리오 2: 지식 + 워크플로우 복합 검색"""
    print("=" * 60)
    print("TEST 3: Agent Scenario - Knowledge + Workflow")
    print("=" * 60)

    from src.agent.agent import run

    query = "현재 시스템 아키텍처에 대해 알려주고, 비디오 생성에 적합한 워크플로우를 추천해줘"
    print(f"\nQuery: {query}")
    print("-" * 40)

    response = run(query)
    print(f"\nAgent Response:\n{response}")
    _check_guardrails(query, response)

    assert response is not None
    assert len(response.strip()) > 10
    print("\n  [OK] Response received")
    print("\n--- Scenario 2 PASSED ---\n")


def test_agent_scenario_3_mmr():
    """시나리오 3: MMR 파라미터를 명시한 다양성 검색"""
    print("=" * 60)
    print("TEST 4: Agent Scenario - MMR Diverse Search")
    print("=" * 60)

    from src.agent.agent import run

    query = (
        "검색 시 diversify=true, mmr_lambda=0.3, mmr_pool=80을 사용해서 "
        "사이버펑크 영상 워크플로우를 추천해줘"
    )
    print(f"\nQuery: {query}")
    print("-" * 40)

    response = run(query)
    print(f"\nAgent Response:\n{response}")
    _check_guardrails(query, response)
    
    assert response is not None
    assert len(response.strip()) > 10
    print("\n  [OK] Response received")
    print("\n--- Scenario 3 PASSED ---\n")


if __name__ == "__main__":
    # 테스트 선택
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "tools":
            test_tools_directly()
        elif test_name == "scenario1":
            test_agent_scenario_1()
        elif test_name == "scenario2":
            test_agent_scenario_2()
        elif test_name == "scenario3":
            test_agent_scenario_3_mmr()
        else:
            print(f"Unknown test: {test_name}")
            print("Usage: python scripts/test_agent.py [tools|scenario1|scenario2|scenario3]")
    else:
        # 전체 테스트 (순서대로)
        try:
            test_tools_directly()
            test_agent_scenario_1()
            test_agent_scenario_2()
            test_agent_scenario_3_mmr()  # 시나리오 3 추가됨
            
            print("=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
        except AssertionError as e:
            print("\n" + "=" * 60)
            print(f"TEST FAILED: {e}")
            print("=" * 60)
            sys.exit(1)
        except Exception as e:
            print("\n" + "=" * 60)
            print(f"ERROR OCCURRED: {e}")
            print("=" * 60)
            sys.exit(1)
