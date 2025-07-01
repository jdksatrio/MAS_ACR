#!/usr/bin/env python3

import asyncio
import pandas as pd
import os
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import time
from datetime import datetime

from langgraph_integration.enhanced_medical_workflow import create_enhanced_medical_workflow
from langchain_ollama import ChatOllama

class MedicalSystemEvaluator:
    """
    Evaluates precision, recall, and other metrics for the medical AI system
    without modifying the core workflow components.
    """
    
    def __init__(self, csv_file: str = "synthetic_patient_cases.csv"):
        self.csv_file = csv_file
        self.results = []
        self.metrics = {}
        
    def load_test_cases(self) -> pd.DataFrame:
        """Load test cases from CSV file"""
        print(f"📊 Loading test cases from {self.csv_file}")
        df = pd.read_csv(self.csv_file, delimiter='|')
        print(f"✅ Loaded {len(df)} test cases")
        return df
    
    def extract_procedures_from_text(self, text: str) -> Set[str]:
        """Extract procedure names from text (both ground truth and predictions)"""
        if pd.isna(text) or not text:
            return set()
        
        # Split by semicolon and clean up
        procedures = [proc.strip() for proc in text.split(';')]
        procedures = [proc for proc in procedures if proc]  # Remove empty strings
        return set(procedures)
    
    def extract_predicted_procedures(self, final_answer: str) -> Set[str]:
        """Extract procedure names from the system's final answer"""
        if not final_answer:
            return set()
        
        predicted_procedures = set()
        
        # Look for **Imaging:** patterns in the final answer
        lines = final_answer.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('**Imaging:**'):
                # Extract procedure name between **Imaging:** and : Usually Appropriate
                procedure_part = line.replace('**Imaging:**', '').strip()
                if ': Usually Appropriate' in procedure_part:
                    procedure_name = procedure_part.split(': Usually Appropriate')[0].strip()
                    predicted_procedures.add(procedure_name)
                elif ':' in procedure_part:
                    # Handle other appropriateness levels
                    procedure_name = procedure_part.split(':')[0].strip()
                    predicted_procedures.add(procedure_name)
        
        return predicted_procedures
    
    def calculate_metrics(self, true_procedures: Set[str], predicted_procedures: Set[str]) -> Dict[str, float]:
        """Calculate precision, recall, F1, and other metrics"""
        if not true_procedures and not predicted_procedures:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
        
        if not predicted_procedures:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
        
        if not true_procedures:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
        
        # Calculate intersection
        intersection = true_procedures.intersection(predicted_procedures)
        
        # Precision: What fraction of predicted procedures are correct?
        precision = len(intersection) / len(predicted_procedures) if predicted_procedures else 0.0
        
        # Recall: What fraction of true procedures were predicted?
        recall = len(intersection) / len(true_procedures) if true_procedures else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact Match: Are the sets exactly the same?
        exact_match = 1.0 if true_procedures == predicted_procedures else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "true_count": len(true_procedures),
            "predicted_count": len(predicted_procedures),
            "intersection_count": len(intersection)
        }
    
    async def evaluate_single_case(self, workflow, case: Dict) -> Dict:
        """Evaluate a single test case"""
        try:
            # Extract test case data
            original_variant = case['original_variant']
            patient_condition = case['synthetic_patient_condition']
            true_procedures_str = case['appropriate_imaging']
            
            print(f"\n🔄 Testing case: {original_variant[:80]}...")
            
            # Run the system
            start_time = time.time()
            result = await workflow.ainvoke({
                'user_query': patient_condition,
                'messages': []
            })
            execution_time = time.time() - start_time
            
            # Extract results
            final_answer = result.get('final_answer', '')
            acr_recommendations = result.get('acr_recommendations', {})
            
            # Extract procedures
            true_procedures = self.extract_procedures_from_text(true_procedures_str)
            predicted_procedures = self.extract_predicted_procedures(final_answer)
            
            # Also extract from ACR recommendations as backup
            if not predicted_procedures and 'usually_appropriate_procedures' in acr_recommendations:
                acr_procedures = acr_recommendations['usually_appropriate_procedures']
                predicted_procedures = {proc['procedure_id'] for proc in acr_procedures}
            
            # Calculate metrics
            metrics = self.calculate_metrics(true_procedures, predicted_procedures)
            
            # Compile results
            case_result = {
                'original_variant': original_variant,
                'patient_condition': patient_condition,
                'true_procedures': list(true_procedures),
                'predicted_procedures': list(predicted_procedures),
                'execution_time': execution_time,
                **metrics,
                'acr_similarity': acr_recommendations.get('best_variant', {}).get('variant_similarity', 0.0),
                'selected_variant': acr_recommendations.get('best_variant', {}).get('variant_id', 'Unknown')
            }
            
            print(f"✅ Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            
            return case_result
            
        except Exception as e:
            print(f"❌ Error evaluating case: {str(e)}")
            return {
                'original_variant': case.get('original_variant', 'Unknown'),
                'patient_condition': case.get('synthetic_patient_condition', 'Unknown'),
                'error': str(e),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'exact_match': 0.0
            }
    
    async def evaluate_system(self, max_cases: int = None, start_from: int = 0):
        """Evaluate the entire system"""
        print("🚀 Starting Medical AI System Evaluation")
        print("="*60)
        
        # Load test cases
        df = self.load_test_cases()
        
        # Limit cases if specified
        if max_cases:
            df = df.iloc[start_from:start_from + max_cases]
            print(f"📋 Evaluating {len(df)} cases (from index {start_from})")
        
        # Initialize local LLM
        print("🔧 Initializing system...")
        llm = ChatOllama(
            model="alibayram/medgemma:latest",
            temperature=0.1
        )
        print("Using Local alibayram/medgemma:latest")
        
        workflow = create_enhanced_medical_workflow(llm, neo4j_password=os.getenv("NEO4J_PASSWORD"))
        
        # Evaluate each case
        print(f"\n🎯 Starting evaluation of {len(df)} cases...")
        start_time = time.time()
        
        for idx, (_, case) in enumerate(df.iterrows()):
            print(f"\n[{idx+1}/{len(df)}] ", end="")
            case_result = await self.evaluate_single_case(workflow, case)
            self.results.append(case_result)
            
            # Save intermediate results every 10 cases
            if (idx + 1) % 10 == 0:
                self.save_intermediate_results(f"evaluation_results_checkpoint_{idx+1}.json")
        
        total_time = time.time() - start_time
        
        # Calculate overall metrics
        self.calculate_overall_metrics()
        
        # Display results
        self.display_results(total_time)
        
        # Save final results
        self.save_results()
    
    def calculate_overall_metrics(self):
        """Calculate overall system metrics"""
        if not self.results:
            return
        
        # Filter out error cases
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            print("❌ No valid results to calculate metrics")
            return
        
        # Calculate averages
        total_precision = sum(r['precision'] for r in valid_results)
        total_recall = sum(r['recall'] for r in valid_results)
        total_f1 = sum(r['f1'] for r in valid_results)
        total_exact_match = sum(r['exact_match'] for r in valid_results)
        
        n_valid = len(valid_results)
        
        self.metrics = {
            'total_cases': len(self.results),
            'valid_cases': n_valid,
            'error_cases': len(self.results) - n_valid,
            'avg_precision': total_precision / n_valid,
            'avg_recall': total_recall / n_valid,
            'avg_f1': total_f1 / n_valid,
            'exact_match_rate': total_exact_match / n_valid,
            'avg_execution_time': sum(r.get('execution_time', 0) for r in valid_results) / n_valid,
            'avg_acr_similarity': sum(r.get('acr_similarity', 0) for r in valid_results) / n_valid
        }
    
    def display_results(self, total_time: float):
        """Display evaluation results"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        if not self.metrics:
            print("❌ No metrics calculated")
            return
        
        print(f"📋 Total Cases: {self.metrics['total_cases']}")
        print(f"✅ Valid Cases: {self.metrics['valid_cases']}")
        print(f"❌ Error Cases: {self.metrics['error_cases']}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"⚡ Avg Time/Case: {self.metrics['avg_execution_time']:.2f}s")
        
        print("\n🎯 PERFORMANCE METRICS:")
        print(f"   Precision: {self.metrics['avg_precision']:.3f}")
        print(f"   Recall:    {self.metrics['avg_recall']:.3f}")
        print(f"   F1 Score:  {self.metrics['avg_f1']:.3f}")
        print(f"   Exact Match: {self.metrics['exact_match_rate']:.3f}")
        
        print(f"\n🔍 ACR PERFORMANCE:")
        print(f"   Avg Similarity: {self.metrics['avg_acr_similarity']:.3f}")
        
        # Show some examples
        valid_results = [r for r in self.results if 'error' not in r]
        if valid_results:
            print(f"\n📝 SAMPLE RESULTS:")
            for i, result in enumerate(valid_results[:3]):
                print(f"\n   Example {i+1}:")
                print(f"   Original: {result['original_variant'][:60]}...")
                print(f"   True: {result['true_procedures']}")
                print(f"   Predicted: {result['predicted_procedures']}")
                print(f"   Metrics: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}")
    
    def save_results(self):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'total_cases': len(self.results),
                    'csv_file': self.csv_file
                },
                'metrics': self.metrics,
                'detailed_results': self.results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save summary CSV
        summary_file = f"evaluation_summary_{timestamp}.csv"
        if self.results:
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(summary_file, index=False)
            print(f"💾 Summary CSV saved to: {summary_file}")
    
    def save_intermediate_results(self, filename: str):
        """Save intermediate results during evaluation"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

async def main():
    """Main evaluation function"""
    evaluator = MedicalSystemEvaluator()
    
    # You can customize these parameters:
    # - max_cases: Limit number of test cases (None for all)
    # - start_from: Starting index (useful for resuming)
    
    await evaluator.evaluate_system(
        max_cases=None,  # Run all 200 cases with enhanced prompt
        start_from=0
    )

if __name__ == "__main__":
    asyncio.run(main()) 