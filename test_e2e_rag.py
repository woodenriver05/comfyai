"""
E2E 테스트: 실제 RAG DB + Gemini LLM 연결

RAG API (localhost:7001) + GOOGLE_API_KEY 필요.
전체 멀티에이전트 파이프라인을 실행하고 각 노드 결과를 검증한다.
"""
import json
import os
import time

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import HumanMessage
from src.agent.supervisor_graph import create_supervisor_graph


def _short(obj, n=150):
    s = str(obj)
    return s if len(s) <= n else s[:n] + "..."


def run_e2e(query: str, label: str, max_steps: int = 30, timeout: int = 180):
    """하나의 E2E 시나리오를 실행하고 결과를 반환."""
    print(f"\n{'='*70}")
    print(f"[E2E] {label}")
    print(f"  Query: {query}")
    print(f"{'='*70}")

    graph = create_supervisor_graph(rag_url="http://localhost:7001")
    initial = {"messages": [HumanMessage(content=query)]}

    start = time.time()
    step = 0
    visited = []
    collected = {}  # 노드별 최근 출력 수집

    try:
        for event in graph.stream(initial, {"recursion_limit": max_steps}):
            for node_name, node_out in event.items():
                step += 1
                elapsed = time.time() - start
                visited.append(node_name)
                collected[node_name] = node_out

                print(f"  [{step:2d}] {node_name:<15s} ({elapsed:.1f}s)")

                if node_name == "supervisor":
                    nn = node_out.get("next_node")
                    intent = node_out.get("intent")
                    line = f"       next={nn}"
                    if intent:
                        line += f", intent={intent}"
                    print(line)

                elif node_name == "planner":
                    cp = node_out.get("creative_plan", {})
                    tp = node_out.get("technical_plan", {})
                    rq = node_out.get("retrieval_query", "")
                    print(f"       mood={cp.get('mood')}, style={cp.get('style')}")
                    print(f"       model={tp.get('model')}, res={tp.get('resolution')}, pipe={tp.get('pipeline')}")
                    print(f"       search_query: {_short(rq, 80)}")

                elif node_name == "retriever":
                    rr = node_out.get("retrieval_results", {})
                    results = rr.get("results", []) if isinstance(rr, dict) else []
                    sigs = rr.get("signals", {}) if isinstance(rr, dict) else {}
                    print(f"       results={len(results)}, top_score={sigs.get('top_score')}")
                    for r in results[:3]:
                        wid = str(r.get("workflow_id", "?"))[:12]
                        tags = r.get("tags", [])
                        score = r.get("score", 0)
                        print(f"         - {wid}... tags={tags} score={score:.3f}")

                elif node_name == "grade_results":
                    print(f"       grade={node_out.get('retrieval_grade')}, retries={node_out.get('retrieval_retries')}")

                elif node_name == "architect":
                    changes = node_out.get("architect_changes", [])
                    print(f"       changes={len(changes)}, done={node_out.get('architect_done')}")
                    for c in changes[:3]:
                        print(f"         {_short(c, 100)}")

                elif node_name == "validator":
                    print(f"       passed={node_out.get('validation_passed')}, issues={len(node_out.get('validation_issues', []))}")

                elif node_name == "human_review":
                    print(f"       approved={node_out.get('human_approved')}")

                elif node_name == "executor":
                    print(f"       output={node_out.get('executor_output')}")
                    msgs = node_out.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        content = last.content if hasattr(last, "content") else str(last)
                        print(f"       last_msg: {_short(content, 120)}")

                elif node_name == "respond":
                    print(f"       (final)")

                if time.time() - start > timeout:
                    print(f"\n  [TIMEOUT] {timeout}s")
                    break

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")

    elapsed = time.time() - start
    print(f"\n  FLOW: {' → '.join(visited)}")
    print(f"  {step} steps in {elapsed:.1f}s")

    return {
        "label": label,
        "visited": visited,
        "steps": step,
        "elapsed": elapsed,
        "collected": collected,
    }


def main():
    print("=" * 70)
    print("  ComfyUI Multi-Agent E2E Test (Live RAG DB + Gemini)")
    print("=" * 70)

    results = []

    # --- Scenario 1: generate (full pipeline) ---
    r1 = run_e2e(
        "비 오는 사이버펑크 도시의 네온 거리, wan2.2로 1080p 영상 만들어줘",
        "Scenario 1: Generate (Full Pipeline)",
    )
    results.append(r1)

    # --- Scenario 2: search (direct executor) ---
    r2 = run_e2e(
        "wan2.2 비디오 워크플로우 찾아줘",
        "Scenario 2: Search (Direct Executor)",
    )
    results.append(r2)

    # --- Scenario 3: knowledge ---
    r3 = run_e2e(
        "ComfyUI에서 GGUF 모델을 로드하는 방법 알려줘",
        "Scenario 3: Knowledge Query",
    )
    results.append(r3)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    checks = []
    for r in results:
        label = r["label"]
        visited = r["visited"]
        col = r["collected"]
        print(f"\n  {label}")
        print(f"    Steps: {r['steps']}, Time: {r['elapsed']:.1f}s")
        print(f"    Flow: {' → '.join(visited)}")

        if "Scenario 1" in label:
            checks.append(("S1: planner 실행", "planner" in visited))
            checks.append(("S1: retriever 실행", "retriever" in visited))
            checks.append(("S1: grade_results 실행", "grade_results" in visited))
            checks.append(("S1: architect 실행", "architect" in visited))
            checks.append(("S1: validator 실행", "validator" in visited))
            checks.append(("S1: human_review 실행", "human_review" in visited))
            checks.append(("S1: executor 실행", "executor" in visited))

            # planner 결과 검증
            planner_out = col.get("planner", {})
            cp = planner_out.get("creative_plan", {})
            tp = planner_out.get("technical_plan", {})
            checks.append(("S1: creative_plan.mood 존재", bool(cp.get("mood"))))
            checks.append(("S1: technical_plan.model 존재", bool(tp.get("model"))))
            checks.append(("S1: retrieval_query 존재", bool(planner_out.get("retrieval_query"))))

            # retriever 결과 검증
            ret_out = col.get("retriever", {})
            rr = ret_out.get("retrieval_results", {})
            if isinstance(rr, dict):
                res_list = rr.get("results", [])
                checks.append(("S1: 검색 결과 > 0", len(res_list) > 0))

            # grade 검증
            grade_out = col.get("grade_results", {})
            grade = grade_out.get("retrieval_grade")
            checks.append(("S1: grade 판정됨", grade in ("good", "fair", "poor")))

            # validator 검증
            val_out = col.get("validator", {})
            checks.append(("S1: validator_done", val_out.get("validator_done", False)))

            # human_review 검증
            hr_out = col.get("human_review", {})
            checks.append(("S1: human_approved", hr_out.get("human_approved") is not None))

            # 무한루프 방지
            checks.append(("S1: 30스텝 이내 종료", r["steps"] < 30))

        elif "Scenario 2" in label:
            checks.append(("S2: intent=search → executor 직행", "executor" in visited))
            checks.append(("S2: planner 건너뜀", "planner" not in visited))

        elif "Scenario 3" in label:
            checks.append(("S3: intent=knowledge → executor 직행", "executor" in visited))
            checks.append(("S3: planner 건너뜀", "planner" not in visited))

    print("\n" + "=" * 70)
    print("  CHECKS")
    print("=" * 70)
    pass_count = 0
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {desc}")
        if ok:
            pass_count += 1

    print(f"\n  {pass_count}/{len(checks)} PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
