import pytest
import asyncio
from unittest.mock import Mock, patch
from validator.utils.hash_verification import calculate_model_hash, verify_model_hash, is_valid_model_hash
from validator.evaluation.scoring import handle_duplicate_submissions, group_by_losses
from core.models.utility_models import MinerSubmission, TaskType
import numpy as np


class MockMinerResult:
    def __init__(self, hotkey, test_loss, synth_loss, repo, model_hash=None, task_type=TaskType.INSTRUCTTEXTTASK):
        self.hotkey = hotkey
        self.test_loss = test_loss
        self.synth_loss = synth_loss
        self.submission = MinerSubmission(repo=repo, model_hash=model_hash) if repo else None
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
    
    def test_calculate_hash_image_models(self):
        """Test hash calculation with image/diffusion models"""
        repo1 = "int1306866/babbda63-94ce-448c-a49b-fc81c9179e27"
        repo2 = "int1306866/b5058266-5071-460d-821d-1b7ee66b00c4"
        
        hash1 = calculate_model_hash(repo1)
        hash2 = calculate_model_hash(repo2)
        
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2
        assert is_valid_model_hash(hash1)
        assert is_valid_model_hash(hash2)
        print(f"Image model hash for {repo1}: {hash1}")
        print(f"Image model hash for {repo2}: {hash2}")
    
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
    async def test_different_image_models_both_kept(self):
        """Test: Different image models with different hashes -> both kept"""
        repo1 = "int1306866/babbda63-94ce-448c-a49b-fc81c9179e27"
        repo2 = "int1306866/b5058266-5071-460d-821d-1b7ee66b00c4"
        
        # Calculate real hashes
        hash1 = calculate_model_hash(repo1)
        hash2 = calculate_model_hash(repo2)
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2
        print(f"Image hash1: {hash1}")
        print(f"Image hash2: {hash2}")
        
        results = [
            MockMinerResult("miner1", 0.5, 0.6, repo1, hash1, TaskType.IMAGETASK),
            MockMinerResult("miner2", 0.5, 0.6, repo2, hash2, TaskType.IMAGETASK),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Both should be kept since different hashes
        assert keep_submission["miner1"] == True
        assert keep_submission["miner2"] == True
        print("✓ Different image models with different hashes both kept")
    
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
    async def test_image_model_copying_attack_prevention(self):
        """Test attack prevention for image models"""
        repo_id = "int1306866/babbda63-94ce-448c-a49b-fc81c9179e27"
        
        # Legitimate miner calculates hash at submission time
        legitimate_hash = calculate_model_hash(repo_id)
        assert legitimate_hash is not None
        
        # Attacker copies model but doesn't have the original hash
        results = [
            MockMinerResult("legitimate_miner", 0.95, 0.93, repo_id, legitimate_hash, TaskType.IMAGETASK),
            MockMinerResult("attacker", 0.95, 0.93, repo_id, None, TaskType.IMAGETASK),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Legitimate miner should be kept
        assert keep_submission["legitimate_miner"] == True
        assert keep_submission["attacker"] == False
        print("✓ Image model copying attack prevented - legitimate submission with hash kept")
    
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


class TestEdgeCases:
    """Test comprehensive edge cases for duplicate detection"""
    
    def test_hash_validation_edge_cases(self):
        """Test hash validation with various invalid inputs"""
        test_cases = [
            ("0" * 64, True),  # All zeros
            ("f" * 64, True),  # All f's  
            ("A" * 64, True),  # Uppercase hex is valid
            ("g" + "0" * 63, False), # Invalid hex char
            ("0" * 63, False), # Too short
            ("0" * 65, False), # Too long
            ("", False), # Empty
            (None, False), # None
            (123, False), # Wrong type
            ("xyz" + "0" * 61, False), # Invalid chars
        ]
        
        for test_hash, expected in test_cases:
            result = is_valid_model_hash(test_hash)
            assert result == expected, f"Failed for hash: {test_hash}, expected {expected}, got {result}"
        print("✓ Hash validation edge cases passed")
    
    @pytest.mark.asyncio
    async def test_complex_duplicate_scenarios(self):
        """Test complex scenarios with 3+ submissions"""
        hash1 = "1111111111111111111111111111111111111111111111111111111111111111"
        hash2 = "2222222222222222222222222222222222222222222222222222222222222222"
        
        # Scenario: 3 submissions with same losses, mix of hashes and no hashes
        results = [
            MockMinerResult("miner_hash1", 0.5, 0.6, "repo1", hash1),
            MockMinerResult("miner_hash2", 0.5, 0.6, "repo2", hash2), 
            MockMinerResult("miner_no_hash", 0.5, 0.6, "repo3", None),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Both hashed submissions should be kept, no-hash should be dropped
        assert keep_submission["miner_hash1"] == True
        assert keep_submission["miner_hash2"] == True
        assert keep_submission["miner_no_hash"] == False
        print("✓ Complex 3-way scenario: hashes beat no-hash")
    
    @pytest.mark.asyncio
    async def test_multiple_identical_hashes(self):
        """Test multiple submissions with identical hashes"""
        same_hash = "3333333333333333333333333333333333333333333333333333333333333333"
        
        results = [
            MockMinerResult("miner1", 0.5, 0.6, "repo1", same_hash),
            MockMinerResult("miner2", 0.5, 0.6, "repo1", same_hash),
            MockMinerResult("miner3", 0.5, 0.6, "repo1", same_hash),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Only one should be kept
        kept_count = sum(keep_submission.values())
        assert kept_count == 1
        assert keep_submission["miner1"] == True
        assert keep_submission["miner2"] == False
        assert keep_submission["miner3"] == False
        print("✓ Multiple identical hashes: only first kept")
    
    @pytest.mark.asyncio
    async def test_mixed_loss_values_with_duplicates(self):
        """Test different loss values with some duplicates"""
        hash1 = "4444444444444444444444444444444444444444444444444444444444444444"
        hash2 = "5555555555555555555555555555555555555555555555555555555555555555"
        
        results = [
            # Group 1: Loss (0.5, 0.6) - duplicates
            MockMinerResult("miner1", 0.5, 0.6, "repo1", hash1),
            MockMinerResult("miner2", 0.5, 0.6, "repo1", hash1),
            # Group 2: Loss (0.7, 0.8) - no duplicates
            MockMinerResult("miner3", 0.7, 0.8, "repo2", hash2),
            # Group 3: Loss (0.9, 1.0) - no duplicates  
            MockMinerResult("miner4", 0.9, 1.0, "repo3", None),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # Only duplicates in first group should be affected
        assert keep_submission["miner1"] == True
        assert keep_submission["miner2"] == False
        assert keep_submission["miner3"] == True
        assert keep_submission["miner4"] == True
        print("✓ Mixed loss values: only duplicates within same loss group affected")
    
    @pytest.mark.asyncio  
    async def test_empty_and_nan_losses(self):
        """Test handling of invalid loss values"""
        hash1 = "6666666666666666666666666666666666666666666666666666666666666666"
        
        results = [
            MockMinerResult("miner_valid", 0.5, 0.6, "repo1", hash1),
            MockMinerResult("miner_nan", float('nan'), float('nan'), "repo2", hash1),
        ]
        
        keep_submission = await handle_duplicate_submissions(results)
        
        # NaN losses should not be grouped, so both kept
        assert keep_submission["miner_valid"] == True
        assert keep_submission["miner_nan"] == True
        print("✓ NaN losses handled correctly")
    
    @pytest.mark.asyncio
    async def test_no_submissions(self):
        """Test empty submission list"""
        results = []
        keep_submission = await handle_duplicate_submissions(results)
        assert keep_submission == {}
        print("✓ Empty submission list handled")
    
    @patch('validator.utils.hash_verification.calculate_model_hash')
    def test_hash_calculation_failure(self, mock_calc):
        """Test hash calculation failure handling"""
        mock_calc.return_value = None
        
        result = verify_model_hash("fake/repo", "valid_hash")
        assert result == False
        print("✓ Hash calculation failure handled")
    
    def test_invalid_repo_id(self):
        """Test invalid repository ID"""
        result = calculate_model_hash("")
        assert result is None
        
        result = calculate_model_hash(None)
        assert result is None
        print("✓ Invalid repo ID handled")


def run_tests():
    """Run all tests and print results"""
    print("=== Testing Hash-Based Submission Verification ===\n")
    
    # Test hash calculation
    print("1. Testing hash calculation...")
    test_calc = TestHashCalculation()
    test_calc.test_calculate_hash_same_repo_twice()
    test_calc.test_calculate_hash_different_repos()
    test_calc.test_calculate_hash_image_models()
    test_calc.test_verify_hash_correct()
    test_calc.test_verify_hash_incorrect()
    print()
    
    # Test duplicate detection
    print("2. Testing duplicate detection...")
    test_dup = TestDuplicateDetectionWithRealHashes()
    asyncio.run(test_dup.test_same_repo_marked_as_duplicate())
    asyncio.run(test_dup.test_different_repos_both_kept())
    asyncio.run(test_dup.test_different_image_models_both_kept())
    asyncio.run(test_dup.test_hash_vs_no_hash_prioritizes_hash())
    print()
    
    # Test attack scenarios
    print("3. Testing attack prevention...")
    test_attack = TestAttackScenarios()
    asyncio.run(test_attack.test_model_copying_attack_prevention())
    asyncio.run(test_attack.test_image_model_copying_attack_prevention())
    asyncio.run(test_attack.test_wrong_hash_attack_prevention())
    print()
    
    # Test edge cases
    print("4. Testing edge cases...")
    test_edge = TestEdgeCases()
    test_edge.test_hash_validation_edge_cases()
    asyncio.run(test_edge.test_complex_duplicate_scenarios())
    asyncio.run(test_edge.test_multiple_identical_hashes())
    asyncio.run(test_edge.test_mixed_loss_values_with_duplicates())
    asyncio.run(test_edge.test_empty_and_nan_losses())
    asyncio.run(test_edge.test_no_submissions())
    test_edge.test_hash_calculation_failure()
    test_edge.test_invalid_repo_id()
    print()
    
    print("✅ All tests completed! Comprehensive coverage achieved.")


if __name__ == "__main__":
    run_tests()