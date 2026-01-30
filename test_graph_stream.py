"""
실시간 멀티-에이전트 스트리밍 테스트

흐름: supervisor → planner → retriever → grade_results → architect → validator → human_review → executor → respond
"""
import time
from langchain_core.messages import HumanMessage
from src.agent.supervisor_graph import create_supervisor_graph

MAX_STEPS = 30  # 무한 루프 방지
TIMEOUT_SEC = 120  # LLM 호출 포함이므로 넉넉하게

def _short(obj, n=120):
    s = str(obj)
    return s if len(s) <= n else s[:n] + "..."

def test_multi_agent_stream():
    graph = create_supervisor_graph()
    initial_state = {
        "messages": [HumanMessage(content="비 오는 사이버펑크 도시, wan2.2 1080p 영상 만들어줘")]
    }

    print("[START] stream test\n")
    start = time.time()
    step = 0
    visited = []

    try:
        for event in graph.stream(initial_state):
            for node_name, node_out in event.items():
                step += 1
                elapsed = time.time() - start
                visited.append(node_name)
                print(f"[STEP {step:2d}] {node_name:<15s} ({elapsed:.1f}s)")

                if node_name == "supervisor":
                    print(f"  next_node: {node_out.get('next_node')}")
                    intent = node_out.get("intent")
                    if intent:
                        print(f"  intent: {intent}")

                elif node_name == "planner":
                    print(f"  creative_plan: {_short(node_out.get('creative_plan'))}")
                    print(f"  technical_plan: {_short(node_out.get('technical_plan'))}")
                    print(f"  retrieval_query: {_short(node_out.get('retrieval_query'))}")

                elif node_name == "retriever":
                    results = node_out.get("retrieval_results", {})
                    count = len(results.get("results", [])) if isinstance(results, dict) else 0
                    print(f"  results: {count}")

                elif node_name == "grade_results":
                    print(f"  grade: {node_out.get('retrieval_grade')}")
                    retries = node_out.get("retrieval_retries")
                    if retries:
                        print(f"  retries: {retries}")

                elif node_name == "architect":
                    changes = node_out.get("architect_changes", [])
                    print(f"  changes: {len(changes)}")
                    print(f"  architect_done: {node_out.get('architect_done')}")

                elif node_name == "validator":
                    print(f"  validation_passed: {node_out.get('validation_passed')}")
                    issues = node_out.get("validation_issues", [])
                    if issues:
                        print(f"  issues: {len(issues)}")

                elif node_name == "human_review":
                    print(f"  human_approved: {node_out.get('human_approved')}")

                elif node_name == "executor":
                    print(f"  executor_output: {node_out.get('executor_output')}")

                elif node_name == "respond":
                    print(f"  (final)")

                if step >= MAX_STEPS:
                    print("\n[STOP] max steps reached")
                    break

                if time.time() - start > TIMEOUT_SEC:
                    print("\n[STOP] timeout reached")
                    break

        elapsed = time.time() - start
        print(f"\n[END] stream finished in {elapsed:.1f}s, {step} steps")
        print(f"[FLOW] {' → '.join(visited)}")

        # 합격 기준 체크
        print("\n=== 합격 기준 ===")
        checks = [
            ("planner 노드 실행", "planner" in visited),
            ("retriever 노드 실행", "retriever" in visited),
            ("grade_results 노드 실행", "grade_results" in visited),
            ("architect 노드 실행", "architect" in visited),
            ("validator 노드 실행", "validator" in visited),
            ("human_review 노드 실행", "human_review" in visited),
            ("무한 루프 없이 종료", step < MAX_STEPS),
        ]
        for desc, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}: {desc}")

        pass_count = sum(1 for _, ok in checks if ok)
        print(f"\n  {pass_count}/{len(checks)} 합격")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_agent_stream()
