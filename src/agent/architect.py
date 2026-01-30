"""
Architect 모듈 - 워크플로우 JSON 정규화

1. Path Normalization: 외부 경로(Windows/Linux/RunPod)를 파일명만 추출
2. Model Alignment: 외부 모델명을 로컬 보유 모델로 매핑
3. Hardware Optimization: VRAM 제약에 따른 최적화 노드 삽입 힌트
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 로컬 자산 매핑 테이블
# 외부 JSON의 모델명 → 로컬 보유 파일명
# 새 모델 추가 시 여기에만 추가하면 됨
# ---------------------------------------------------------------------------

LOCAL_ASSET_MAP: Dict[str, str] = {
    # Checkpoints / UNET
    "Wan2.1_14B_FP16.safetensors": "wan2.2-ti2v-5b-Q4_K_S.gguf",
    "Wan2.1_5B_FP16.safetensors": "wan2.2-ti2v-5b-Q4_K_S.gguf",
    "HunYuan_Video_720.ckpt": "wan2.2-ti2v-5b-Q4_K_S.gguf",
    # VAE
    "vae-ft-mse-840000.safetensors": "ae.safetensors",
    # LoRA
    "FilmGrain_Noise.safetensors": "FilmGrain_v2.gguf",
}

# 경로에서 파일명을 추출하는 패턴 키 목록
_PATH_KEYS = {
    "ckpt_name", "checkpoint", "model_name", "vae_name", "clip_name",
    "unet_name", "lora_name", "control_net_name", "image",
}


# ---------------------------------------------------------------------------
# 핵심 함수
# ---------------------------------------------------------------------------

def architect_normalize(
    workflow_json: Dict[str, Any],
    asset_map: Dict[str, str] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    워크플로우 JSON을 정규화합니다.

    Args:
        workflow_json: ComfyUI API 포맷 워크플로우 JSON
        asset_map: 외부 모델명 → 로컬 파일명 매핑 (기본: LOCAL_ASSET_MAP)

    Returns:
        (정규화된 JSON, 변경 이력 리스트)
    """
    if asset_map is None:
        asset_map = LOCAL_ASSET_MAP

    normalized = json.loads(json.dumps(workflow_json))  # deep copy
    changes: List[str] = []

    for node_id, node_data in normalized.items():
        if not isinstance(node_data, dict):
            continue

        inputs = node_data.get("inputs")
        if not isinstance(inputs, dict):
            continue

        class_type = node_data.get("class_type", "")

        for key, value in list(inputs.items()):
            if not isinstance(value, str):
                continue

            # 1) 경로 정규화: Windows(\) 또는 Linux(/) 경로에서 파일명 추출
            filename = re.split(r"[\\/]", value)[-1]

            # 2) 모델 매핑: asset_map에 있으면 로컬 파일로 교체
            if filename in asset_map:
                new_filename = asset_map[filename]
                if value != new_filename:
                    inputs[key] = new_filename
                    changes.append(
                        f"[{node_id}:{class_type}] {key}: '{_shorten(value)}' -> '{new_filename}' (MAPPED)"
                    )
            # 3) 경로만 지저분한 경우: 파일명만 남김
            elif value != filename and _is_path_key(key):
                inputs[key] = filename
                changes.append(
                    f"[{node_id}:{class_type}] {key}: '{_shorten(value)}' -> '{filename}' (CLEANED)"
                )

    return normalized, changes


def _is_path_key(key: str) -> bool:
    """파일 경로를 담는 키인지 판별."""
    key_lower = key.lower()
    return key_lower in _PATH_KEYS or "path" in key_lower or "name" in key_lower


def _shorten(s: str, max_len: int = 60) -> str:
    """긴 문자열을 줄임."""
    return s if len(s) <= max_len else "..." + s[-(max_len - 3):]
