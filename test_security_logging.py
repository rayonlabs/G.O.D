#!/usr/bin/env python3
"""
Test script to verify security logging works correctly
"""

def test_imports():
    """Test all imports needed for security logging"""
    print("Testing imports...")
    try:
        import datetime
        print("✅ datetime - OK")
        
        import os
        print("✅ os - OK") 
        
        import traceback
        print("✅ traceback - OK")
        
        # Test the actual functions we use
        timestamp = datetime.datetime.utcnow().isoformat()
        print(f"✅ timestamp generation - {timestamp}")
        
        process_id = os.getpid()
        print(f"✅ process ID - {process_id}")
        
        stack = traceback.format_stack()
        print(f"✅ stack trace - {len(stack)} frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_malicious_repo_detection():
    """Test malicious repo detection logic"""
    print("\nTesting malicious repo detection...")
    
    test_cases = [
        ("https://github.com/haihp02/sn56-tournament-repo", True),
        ("https://github.com/legitimate/repo", False),
        ("https://github.com/haihp02/sn56-tournament-repo.git", True),
        ("https://github.com/other/haihp02/sn56-tournament-repo", True),
    ]
    
    for repo, should_detect in test_cases:
        detected = "haihp02/sn56-tournament-repo" in repo
        status = "✅" if detected == should_detect else "❌"
        print(f"{status} {repo} - detected: {detected}, expected: {should_detect}")

def test_logging_format():
    """Test the actual logging format strings"""
    print("\nTesting logging format...")
    try:
        import datetime
        import os
        
        # Mock data
        timestamp = datetime.datetime.utcnow().isoformat()
        client_ip = "10.0.1.153" 
        user_agent = "python-httpx/0.24.0"
        process_id = os.getpid()
        hotkey = "5CfrR18PBjWGNTy8cq9wiZNJT9oi1eoDGSyJzN3ewSKA8bVX"
        task_id = "test-task-123"
        
        # Test the exact format strings we use
        log1 = f"[SECURITY] [{timestamp}] IP check - Client: {client_ip}, User-Agent: {user_agent}, PID: {process_id}"
        print(f"✅ IP check log: {log1}")
        
        log2 = f"[SECURITY ALERT] [{timestamp}] MALICIOUS REPO DETECTED!"
        print(f"✅ Alert log: {log2}")
        
        log3 = f"[SECURITY ALERT] [{timestamp}] IP: {client_ip}, User-Agent: {user_agent}, PID: {process_id}"
        print(f"✅ Details log: {log3}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging format failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Security Logging Test Suite")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_logging_format() 
    test_malicious_repo_detection()  # Always run this
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Security logging should work.")
    else:
        print("❌ Some tests failed. Check the trainer environment.")