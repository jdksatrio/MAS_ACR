#!/usr/bin/env python3

from evaluate_system_precision_recall import MedicalSystemEvaluator

def main():
    """Check the current evaluation status"""
    print("🔍 Checking evaluation status...")
    print("="*50)
    
    evaluator = MedicalSystemEvaluator()
    completed, next_index = evaluator.check_evaluation_status()
    
    if completed > 0:
        total_cases = 200  # Assuming 200 total test cases
        progress = (completed / total_cases) * 100
        remaining = total_cases - completed
        
        print(f"📈 Progress: {progress:.1f}% ({completed}/{total_cases})")
        print(f"⏳ Remaining: {remaining} cases")
        
        if completed == total_cases:
            print("🎉 Evaluation completed!")
        else:
            print(f"🔄 To resume: Run the evaluation script again")
    else:
        print("🆕 No evaluation in progress")

if __name__ == "__main__":
    main() 