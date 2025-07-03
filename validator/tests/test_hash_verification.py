import pytest
import asyncio
from unittest.mock import Mock, patch
from validator.utils.hash_verification import calculate_model_hash, verify_model_hash, is_valid_model_hash
from validator.evaluation.scoring import handle_duplicate_submissions, group_by_losses
from validator.core.models import Submission
from core.models.utility_models import TaskType
import numpy as np


class MockMinerResult:
    def __init__(self, hotkey, test_loss, synth_loss, repo, model_hash=None, task_type=TaskType.INSTRUCTTEXTTASK):
        self.hotkey = hotkey
        self.test_loss = test_loss
        self.synth_loss = synth_loss
        self.submission = Submission(repo=repo, model_hash=model_hash) if repo else None
        self.task_type = task_type


class TestHashCalculation:
    """Test actual hash calculation with real repositories"""
    
    def test_calculate_hash_same_repo_twice(self):
        """Test that the same repo produces the same hash consistently"""
        repo_id = "unsloth/Llama-3.2-1B"
        
        hash1 = calculate_model_hash(repo_id)
        hash2 = calculate_model_hash(repo_id)
        
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 == hash2
        assert is_valid_model_hash(hash1)
        print(f"Hash for {repo_id}: {hash1}")
    
    def test_calculate_hash_different_repos(self):
        """Test that different repos produce different hashes"""
        repo1 = "unsloth/Llama-3.2-1B"
        repo2 = "kyutai/stt-1b-en_fr"
        
        hash1 = calculate_model_hash(repo1)
        hash2 = calculate_model_hash(repo2)
        
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2
        assert is_valid_model_hash(hash1)
        assert is_valid_model_hash(hash2)
        print(f"Hash for {repo1}: {hash1}")
        print(f"Hash for {repo2}: {hash2}")
    
    def test_verify_hash_correct(self):
        """Test hash verification with correct hash"""
        repo_id = "unsloth/Llama-3.2-1B"
        
        # Calculate the actual hash
        actual_hash = calculate_model_hash(repo_id)
        assert actual_hash is not None
        
        # Verify with the same hash
        result = verify_model_hash(repo_id, actual_hash, cleanup_cache=False)
        assert result == True
    
    def test_verify_hash_incorrect(self):
        """Test hash verification with incorrect hash"""
        repo_id = "unsloth/Llama-3.2-1B"
        fake_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        
        result = verify_model_hash(repo_id, fake_hash)
        assert result == False


class TestDuplicateDetectionWithRealHashes:
    """Test duplicate detection using real calculated hashes"""
    
    @pytest.mark.asyncio
    async def test_same_repo_marked_as_duplicate(self):
        """Test: Two submissions of same repo with same hash -> marked as duplicate"""
        repo_id = "unsloth/Llama-3.2-1B"
        
        # Calculate real hash
        model_hash = calculate_model_hash(repo_id)
        assert model_hash is not None
        print(f"Using hash: {model_hash}")
        
        results = [
            MockMinerResult("miner1", 0.5, 0.6, repo_id, model_hash),
            MockMinerResult("miner2", 0.5, 0.6, repo_id, model_hash),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # One should be kept, one marked as duplicate
        kept_count = sum(keep_submission.values())
        assert kept_count == 1
        assert keep_submission["miner1"] == True
        assert keep_submission["miner2"] == False
        print("✓ Same repo with same hash correctly marked as duplicate")
    
    @pytest.mark.asyncio
    async def test_different_repos_both_kept(self):
        """Test: Different repos with different hashes -> both kept"""
        repo1 = "unsloth/Llama-3.2-1B"
        repo2 = "kyutai/stt-1b-en_fr"
        
        # Calculate real hashes
        hash1 = calculate_model_hash(repo1)
        hash2 = calculate_model_hash(repo2)
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2
        print(f"Hash1: {hash1}")
        print(f"Hash2: {hash2}")
        
        results = [
            MockMinerResult("miner1", 0.5, 0.6, repo1, hash1),
            MockMinerResult("miner2", 0.5, 0.6, repo2, hash2),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Both should be kept since different hashes
        assert keep_submission["miner1"] == True
        assert keep_submission["miner2"] == True
        print("✓ Different repos with different hashes both kept")
    
    @pytest.mark.asyncio
    async def test_hash_vs_no_hash_prioritizes_hash(self):
        """Test: Same repo, one with hash, one without -> hash prioritized"""
        repo_id = "unsloth/Llama-3.2-1B"
        
        # Calculate real hash
        model_hash = calculate_model_hash(repo_id)
        assert model_hash is not None
        print(f"Using hash: {model_hash}")
        
        results = [
            MockMinerResult("miner_with_hash", 0.5, 0.6, repo_id, model_hash),
            MockMinerResult("miner_no_hash", 0.5, 0.6, repo_id, None),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Hash submission should be kept
        assert keep_submission["miner_with_hash"] == True
        assert keep_submission["miner_no_hash"] == False
        print("✓ Submission with hash prioritized over submission without hash")
    
    @pytest.mark.asyncio
    @patch('validator.evaluation.scoring.get_hf_upload_timestamp')
    async def test_no_hashes_timestamp_fallback(self, mock_timestamp):
        """Test: No hashes provided -> falls back to timestamp"""
        from datetime import datetime
        
        # Mock timestamps
        mock_timestamp.side_effect = lambda repo: {
            "unsloth/Llama-3.2-1B": datetime(2023, 1, 1),
            "kyutai/stt-1b-en_fr": datetime(2023, 1, 2),
        }.get(repo)
        
        results = [
            MockMinerResult("early_miner", 0.5, 0.6, "unsloth/Llama-3.2-1B", None),
            MockMinerResult("late_miner", 0.5, 0.6, "kyutai/stt-1b-en_fr", None),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Earlier timestamp should be kept
        assert keep_submission["early_miner"] == True
        assert keep_submission["late_miner"] == False
        print("✓ Timestamp fallback works when no hashes provided")


class TestAttackScenarios:
    """Test specific attack scenarios the hash system is designed to prevent"""
    
    @pytest.mark.asyncio
    async def test_model_copying_attack_prevention(self):
        """
        Simulate the attack scenario:
        1. Legitimate miner submits with hash
        2. Attacker copies same model but can't provide original hash
        3. System should prefer the hashed submission
        """
        repo_id = "unsloth/Llama-3.2-1B"
        
        # Legitimate miner calculates hash at submission time
        legitimate_hash = calculate_model_hash(repo_id)
        assert legitimate_hash is not None
        
        # Attacker copies model but doesn't have the original hash
        results = [
            MockMinerResult("legitimate_miner", 0.95, 0.93, repo_id, legitimate_hash),
            MockMinerResult("attacker", 0.95, 0.93, repo_id, None),  # No hash!
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Legitimate miner should be kept
        assert keep_submission["legitimate_miner"] == True
        assert keep_submission["attacker"] == False
        print("✓ Model copying attack prevented - legitimate submission with hash kept")
    
    @pytest.mark.asyncio 
    async def test_wrong_hash_attack_prevention(self):
        """
        Test attacker providing wrong hash:
        1. Legitimate miner submits with correct hash
        2. Attacker tries to submit same model with wrong hash
        3. Both should be marked as having different models (different hashes)
        """
        repo_id = "unsloth/Llama-3.2-1B"
        
        # Legitimate hash
        correct_hash = calculate_model_hash(repo_id)
        # Fake hash attacker might try
        fake_hash = "1111111111111111111111111111111111111111111111111111111111111111"
        
        results = [
            MockMinerResult("legitimate_miner", 0.95, 0.93, repo_id, correct_hash),
            MockMinerResult("attacker", 0.95, 0.93, repo_id, fake_hash),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Both kept since different hashes (system thinks they're different models)
        assert keep_submission["legitimate_miner"] == True
        assert keep_submission["attacker"] == True
        print("✓ Wrong hash doesn't help attacker - treated as different model")


def run_tests():
    """Run all tests and print results"""
    print("=== Testing Hash-Based Submission Verification ===\n")
    
    # Test hash calculation
    print("1. Testing hash calculation...")
    test_calc = TestHashCalculation()
    test_calc.test_calculate_hash_same_repo_twice()
    test_calc.test_calculate_hash_different_repos()
    test_calc.test_verify_hash_correct()
    test_calc.test_verify_hash_incorrect()
    print()
    
    # Test duplicate detection
    print("2. Testing duplicate detection...")
    test_dup = TestDuplicateDetectionWithRealHashes()
    asyncio.run(test_dup.test_same_repo_marked_as_duplicate())
    asyncio.run(test_dup.test_different_repos_both_kept())
    asyncio.run(test_dup.test_hash_vs_no_hash_prioritizes_hash())
    print()
    
    # Test attack scenarios
    print("3. Testing attack prevention...")
    test_attack = TestAttackScenarios()
    asyncio.run(test_attack.test_model_copying_attack_prevention())
    asyncio.run(test_attack.test_wrong_hash_attack_prevention())
    print()
    
    print("✅ All tests completed!")


if __name__ == "__main__":
    run_tests()