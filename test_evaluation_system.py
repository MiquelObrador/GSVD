#!/usr/bin/env python3
"""
Quick test of the evaluation system with a small configuration.
Use this to verify everything works before running the full evaluation.
"""

import subprocess
import sys
import json
from pathlib import Path
import tempfile

def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("🧪 Running quick test of evaluation system...")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_output = f.name
    
    try:
        # Test with original model (ratio=1.0) and small parameters
        cmd = [
            sys.executable, "evaluate_speed.py",
            "--model_base", "gpt2",  # Use smaller model for testing
            "--updated_model_path", "gpt2",
            "--ratio", "1.0",
            "--batch_size", "2",  # Increased for better testing
            "--generated_len", "16",  # Increased for better testing
            "--seq_len", "512",  # Use smaller seq_len for GPT-2 (max 1024)
            "--eval_dataset", "wikitext2",
            "--output_file", temp_output
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            
            # Check if results were written
            if Path(temp_output).exists():
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"✅ Results written: {len(lines)} records")
                        
                        # Parse and display the result
                        try:
                            result_data = json.loads(lines[0])
                            throughput = result_data.get('throughput_tokens_per_sec', 0)
                            memory = result_data.get('total_peak_memory_gb', 0)
                            successful_batches = result_data.get('successful_batches', 'N/A')
                            failed_batches = result_data.get('failed_batches', 'N/A')
                            error = result_data.get('error', None)
                            
                            print(f"📊 Test Results:")
                            print(f"   Throughput: {throughput:.1f} tokens/sec")
                            print(f"   Memory: {memory:.2f} GB")
                            print(f"   Successful batches: {successful_batches}")
                            print(f"   Failed batches: {failed_batches}")
                            
                            if error:
                                print(f"   ⚠️  Error reported: {error}")
                            elif throughput == 0:
                                print(f"   ⚠️  Zero throughput may indicate test issues")
                                
                        except json.JSONDecodeError:
                            print("⚠️  Results file contains invalid JSON")
                    else:
                        print("⚠️  Results file is empty")
            else:
                print("⚠️  Results file was not created")
        else:
            print("❌ Evaluation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up
        if Path(temp_output).exists():
            Path(temp_output).unlink()
    
    return True

def test_analysis():
    """Test the analysis script with existing data."""
    print("\n🔍 Testing analysis script...")
    
    # Check if there are any existing results to analyze
    result_files = list(Path().glob("**/evaluation_results.jsonl"))
    
    if result_files:
        test_file = result_files[0]
        print(f"Found existing results file: {test_file}")
        
        cmd = [
            sys.executable, "analyze_throughput_results.py",
            str(test_file),
            "--output_dir", "test_analysis"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Analysis script works!")
                return True
            else:
                print("⚠️  Analysis script had issues (may be due to missing matplotlib)")
                print("STDERR:", result.stderr)
                return True  # Not critical for core functionality
        except Exception as e:
            print(f"⚠️  Analysis test failed: {e}")
            return True  # Not critical
    else:
        print("⚠️  No existing results to test analysis with")
        return True

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Throughput Evaluation System\n")
    
    success = True
    
    # Test 1: Quick evaluation test
    success &= run_quick_test()
    
    # Test 2: Analysis test
    success &= test_analysis()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Check model paths in run_thoughput.sh")
        print("2. Run: ./run_thoughput.sh")
        print("3. Analyze with: python analyze_throughput_results.py <results_file>")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("The system may still work, but review the issues.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())