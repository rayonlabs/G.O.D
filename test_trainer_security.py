#!/usr/bin/env python3
"""
End-to-end test script to verify trainer security logging by making actual HTTP requests
"""

import requests
import json
import time

TRAINER_URL = "http://localhost:8001"

def test_unauthorized_access():
    """Test that unauthorized IPs get blocked"""
    print("🔒 Testing unauthorized access (should be blocked)...")
    
    try:
        response = requests.get(f"{TRAINER_URL}/v1/trainer/get_gpu_availability", timeout=5)
        if response.status_code == 403:
            print("✅ Unauthorized request blocked (403 Forbidden)")
            return True
        else:
            print(f"❌ Expected 403 but got {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_malicious_repo_detection():
    """Test malicious repo detection with actual HTTP request"""
    print("🚨 Testing malicious repo detection...")
    
    # Create a fake malicious training request
    malicious_payload = {
        "training_data": {
            "model": "test-model",
            "task_id": "test-attack-123",
            "hours_to_complete": 1.0,
            "expected_repo_name": "test-repo"
        },
        "github_repo": "https://github.com/haihp02/sn56-tournament-repo",
        "gpu_ids": [0],
        "hotkey": "TEST_ATTACK_HOTKEY_12345",
        "github_branch": "main",
        "github_commit_hash": "abc123"
    }
    
    try:
        # This should trigger our malicious repo detection
        response = requests.post(
            f"{TRAINER_URL}/v1/trainer/start_training", 
            json=malicious_payload,
            timeout=10,
            headers={"User-Agent": "security-test-script/1.0"}
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 403:
            print("✅ Request blocked by IP whitelist (expected)")
        else:
            print(f"Response: {response.text[:200]}...")
            
        print("✅ Malicious repo test request sent (check trainer logs for alerts)")
        return True
        
    except Exception as e:
        print(f"❌ Malicious repo test failed: {e}")
        return False

def test_legitimate_request():
    """Test a legitimate repo request"""
    print("✅ Testing legitimate repo request...")
    
    legitimate_payload = {
        "training_data": {
            "model": "test-model",
            "task_id": "test-legit-123", 
            "hours_to_complete": 1.0,
            "expected_repo_name": "test-repo"
        },
        "github_repo": "https://github.com/legitimate/test-repo",
        "gpu_ids": [0],
        "hotkey": "TEST_LEGIT_HOTKEY_12345",
        "github_branch": "main", 
        "github_commit_hash": "def456"
    }
    
    try:
        response = requests.post(
            f"{TRAINER_URL}/v1/trainer/start_training",
            json=legitimate_payload, 
            timeout=10,
            headers={"User-Agent": "security-test-script/1.0"}
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 403:
            print("✅ Request blocked by IP whitelist (expected)")
        else:
            print(f"Response: {response.text[:200]}...")
            
        print("✅ Legitimate repo test request sent")
        return True
        
    except Exception as e:
        print(f"❌ Legitimate repo test failed: {e}")
        return False

def check_logs_instructions():
    """Print instructions for checking logs"""
    print("\n" + "="*60)
    print("📋 TO VERIFY THE LOGGING WORKED:")
    print("="*60)
    print("Run this command to check the trainer logs:")
    print('pm2 logs trainer | grep -E "\\[SECURITY|MALICIOUS|TEST_ATTACK|TEST_LEGIT"')
    print("\n🔍 Look for:")
    print("• [SECURITY] logs showing IP, User-Agent, timestamps")
    print("• [SECURITY ALERT] MALICIOUS REPO DETECTED for haihp02 repo")
    print("• Process IDs and request headers") 
    print("• No alerts for the legitimate repo")
    print("="*60)

if __name__ == "__main__":
    print("🧪 Trainer Security End-to-End Test")
    print("="*50)
    
    print(f"Testing trainer at: {TRAINER_URL}")
    print("Note: These requests will likely be blocked by IP whitelist")
    print("But they will test the logging system!\n")
    
    # Run tests
    test_unauthorized_access()
    print()
    test_malicious_repo_detection() 
    print()
    test_legitimate_request()
    
    check_logs_instructions()
    
    print(f"\n✨ Tests completed! Check trainer logs to verify security logging works.")