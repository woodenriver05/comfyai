
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 현재 디렉토리(.scripts/)에서 상위 디렉토리로 이동하여 모듈 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.tools import (
    push_workflow,
    search_workflows,
    _compute_signals,
    _is_retryable,
    set_client
)
import json

class TestToolsUpdate(unittest.TestCase):
    def setUp(self):
        # Mock Client 설정
        self.mock_client = MagicMock()
        set_client(self.mock_client)

    def test_preview_diff_structure(self):
        """Diff 구조 확인: original, override, changed 필드가 포함되어야 함"""
        # Mock workflow info
        mock_info = MagicMock()
        mock_info.workflow_id = "test-id"
        mock_info.last_prompt_positive = "old_pos"
        mock_info.last_prompt_negative = "old_neg"
        mock_info.last_seed = 123
        
        self.mock_client.recall_info.return_value = mock_info

        # Run push_workflow with dry_run=True and some overrides
        result_json = push_workflow(
            workflow_id="test-id",
            prompt_positive="new_pos",
            seed=123, # Same seed
            dry_run=True
        )
        result = json.loads(result_json)
        
        preview = result["preview"]
        
        # Check prompt_positive (Changed)
        self.assertEqual(preview["prompt_positive"]["original"], "old_pos")
        self.assertEqual(preview["prompt_positive"]["override"], "new_pos")
        self.assertTrue(preview["prompt_positive"]["changed"])

        # Check prompt_negative (Not changed, no override provided)
        self.assertEqual(preview["prompt_negative"]["original"], "old_neg")
        self.assertIsNone(preview["prompt_negative"]["override"])
        self.assertFalse(preview["prompt_negative"]["changed"])

        # Check seed (Override provided but same value) - implementation detail: 
        # _field checks `override is not None and override != original`.
        # If override is passed, it is compared.
        self.assertEqual(preview["seed"]["original"], 123)
        self.assertEqual(preview["seed"]["override"], 123)
        self.assertFalse(preview["seed"]["changed"])

    def test_signals_logic(self):
        """Signals 로직 확인: low_confidence, ambiguous 등"""
        # Case 1: No results
        s1 = _compute_signals([])
        self.assertTrue(s1["no_results"])
        self.assertTrue(s1["low_confidence"])

        # Case 2: Low score
        s2 = _compute_signals([{"score": 0.1}])
        self.assertFalse(s2["no_results"])
        self.assertTrue(s2["low_confidence"]) # 0.1 < 0.25 (THRESHOLD)

        # Case 3: Ambiguous (small gap)
        s3 = _compute_signals([{"score": 0.82}, {"score": 0.80}])
        self.assertFalse(s3["low_confidence"])
        self.assertTrue(s3["ambiguous"]) # Gap 0.02 < 0.05 (THRESHOLD)
        self.assertEqual(s3["score_gap"], 0.02)

        # Case 4: Clear winner
        s4 = _compute_signals([{"score": 0.9}, {"score": 0.5}])
        self.assertFalse(s4["ambiguous"])
        self.assertEqual(s4["score_gap"], 0.4)

    def test_retryable_error(self):
        """Retryable 에러 확인"""
        self.assertTrue(_is_retryable(TimeoutError("Timeout")))
        self.assertTrue(_is_retryable(ConnectionError("Connection refused")))
        self.assertTrue(_is_retryable(OSError("Network unreachable")))
        self.assertFalse(_is_retryable(ValueError("Invalid input")))

if __name__ == "__main__":
    unittest.main()
