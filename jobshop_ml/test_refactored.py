"""
Quick test script to verify the refactored system is ready to use.

Run this to check if everything is set up correctly.
"""

import sys
import os

def test_imports():
    """Test if all core modules can be imported."""
    print("Testing imports...")
    try:
        from core import (
            SchedulingInstance,
            ScheduleSolution,
            DataLoader,
            create_instance,
            validate_instance,
            compute_statistics,
            ModelBuilder,
            Solver,
            Reporter,
            ExcelWriter
        )
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install pandas numpy openpyxl gurobipy tabulate")
        return False

def test_basic_functionality():
    """Test basic functionality without solving."""
    print("\nTesting basic functionality...")
    try:
        from core import SchedulingInstance
        
        # Create a minimal instance
        instance = SchedulingInstance(['JOB1', 'JOB2'])
        instance.add_operation('JOB1', 'OP1', 1, 10.0, 'MACHINE1')
        instance.add_operation('JOB2', 'OP2', 1, 15.0, 'MACHINE1')
        
        print(f"✓ Created instance: {instance}")
        
        # Test instance methods
        ops = instance.get_job_operations('JOB1')
        print(f"✓ Job operations: {len(ops)} operations")
        
        from core import validate_instance
        is_valid, errors = validate_instance(instance)
        print(f"✓ Instance validation: {'Valid' if is_valid else 'Invalid'}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_solution_class():
    """Test ScheduleSolution class."""
    print("\nTesting ScheduleSolution class...")
    try:
        from core import SchedulingInstance, ScheduleSolution
        
        instance = SchedulingInstance(['JOB1'])
        instance.add_operation('JOB1', 'OP1', 1, 10.0, 'MACHINE1')
        
        solution = ScheduleSolution(instance)
        solution.makespan = 100.0
        solution.objective_value = 150.0
        solution.start_times[('JOB1', 'OP1')] = 0.0
        solution.completion_times[('JOB1', 'OP1')] = 10.0
        
        print(f"✓ Created solution: {solution}")
        
        sequence = solution.get_schedule_sequence()
        print(f"✓ Schedule sequence: {len(sequence)} operations")
        
        return True
    except Exception as e:
        print(f"✗ Solution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'openpyxl': 'openpyxl',
        'gurobipy': 'gurobipy',
        'tabulate': 'tabulate'
    }
    
    missing = []
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("REFACTORED SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Test 2: Imports
    results.append(("Imports", test_imports()))
    
    # Test 3: Basic functionality
    if results[-1][1]:  # Only if imports succeeded
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("Solution Class", test_solution_class()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("  1. Place your Excel files in this directory")
        print("  2. Run: python main_scheduling.py --solve --export-excel schedule.xlsx")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before using the system.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



