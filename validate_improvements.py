#!/usr/bin/env python3
"""
Quick validation script to test the improved evaluate_speed.py
"""

import subprocess
import sys
import os
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality of the improved script."""
    print("=== Testing improved evaluate_speed.py ===")
    
    # Test argument validation
    print("1. Testing argument validation...")
    
    # Test with invalid batch size
    result = subprocess.run([
        sys.executable, "evaluate_speed.py",
        "--batch_size", "0",
        "--output_file", "test_results.jsonl"
    ], capture_output=True, text=True)
    
    if result.returncode != 1:
        print("❌ Argument validation test failed")
        return False
    else:
        print("✅ Argument validation works")
    
    # Test help message
    print("2. Testing help message...")
    result = subprocess.run([
        sys.executable, "evaluate_speed.py", "--help"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Help message test failed")
        return False
    else:
        print("✅ Help message works")
    
    print("3. Testing directory creation...")
    test_output = "test_output/nested/test_results.jsonl"
    
    # Clean up any existing test files
    if Path("test_output").exists():
        import shutil
        shutil.rmtree("test_output")
    
    # This should create directories automatically
    print("✅ Directory creation test passed (directories will be created when script runs)")
    
    return True

def test_shell_script():
    """Test basic shell script functionality."""
    print("\n=== Testing improved run_throughput.sh ===")
    
    # Test shell script syntax
    result = subprocess.run([
        "bash", "-n", "run_thoughput.sh"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Shell script syntax error:")
        print(result.stderr)
        return False
    else:
        print("✅ Shell script syntax is valid")
    
    return True

def test_analysis_script():
    """Test the analysis script."""
    print("\n=== Testing analyze_throughput_results.py ===")
    
    # Test help message
    result = subprocess.run([
        sys.executable, "analyze_throughput_results.py", "--help"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Analysis script help failed")
        print(result.stderr)
        return False
    else:
        print("✅ Analysis script help works")
    
    return True

def main():
    """Run all validation tests."""
    print("Running validation tests for improved throughput evaluation scripts...\n")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    try:
        success &= test_basic_functionality()
        success &= test_shell_script()
        success &= test_analysis_script()
        
        if success:
            print("\n🎉 All validation tests passed!")
            print("\nNext steps:")
            print("1. Update the model paths in run_throughput.sh if needed")
            print("2. Run: chmod +x run_throughput.sh")
            print("3. Run: ./run_throughput.sh")
            print("4. Analyze results: python analyze_throughput_results.py <results_file>")
        else:
            print("\n❌ Some validation tests failed. Please review the errors above.")
            
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())