#!/usr/bin/env python3

import asyncio
import pandas as pd
import os
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import time
from datetime import datetime
import psutil

from langgraph_integration.enhanced_medical_workflow import create_enhanced_medical_workflow
from langchain_ollama import ChatOllama

class MedicalSystemEvaluator:
    """
    Evaluates precision, recall, and other metrics for the medical AI system
    without modifying the core workflow components.
    """
    
    def __init__(self, csv_file: str = "patient_cases_openai.csv"):
        self.csv_file = csv_file
        self.results = []  # Keep for backward compatibility, but limit size
        self.results_file = None  # Stream results to disk
        self.case_count = 0
        self.metrics = {}
        self.checkpoint_file = "evaluation_checkpoint.json"
        
    def _open_results_stream(self):
        """Open a file stream for writing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file_path = f"evaluation_results_stream_{timestamp}.jsonl"
        self.results_file = open(self.results_file_path, 'w')
        print(f"📝 Streaming results to: {self.results_file_path}")
        
    def _write_result_to_stream(self, case_result: Dict):
        """Write a single case result to the stream"""
        if not self.results_file:
            self._open_results_stream()
        
        # Write to stream
        self.results_file.write(json.dumps(case_result, default=str) + '\n')
        self.results_file.flush()  # Ensure it's written immediately
        
        # Keep only last 10 results in memory for display purposes
        self.results.append(case_result)
        if len(self.results) > 10:
            self.results.pop(0)  # Remove oldest
            
        self.case_count += 1
        
    def _close_results_stream(self):
        """Close the results stream"""
        if self.results_file:
            self.results_file.close()
            self.results_file = None
        
    def _force_ollama_cleanup(self):
        """Force Ollama to unload models to free memory"""
        try:
            import subprocess
            # Try to stop the model to free memory
            result = subprocess.run(
                ['ollama', 'stop', 'alibayram/medgemma:latest'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print("🧹 Ollama model unloaded")
            else:
                print(f"⚠️  Ollama stop warning: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not stop Ollama model: {e}")
        
    def load_checkpoint(self) -> int:
        """Load previous results and return the index to continue from"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.results = checkpoint_data.get('results', [])
                    last_index = checkpoint_data.get('last_completed_index', -1)
                    print(f"📂 Loaded checkpoint: {len(self.results)} completed cases")
                    print(f"🔄 Resuming from case {last_index + 2}")
                    return last_index + 1
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print("🔄 Starting fresh evaluation")
        return 0
    
    def save_checkpoint(self, last_completed_index: int):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'results': self.results[-10:],  # Only save last 10 for resuming
            'total_completed': self.case_count,
            'results_file_path': getattr(self, 'results_file_path', None),
            'last_completed_index': last_completed_index,
            'timestamp': datetime.now().isoformat(),
            'case_count': self.case_count
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        print(f"💾 Checkpoint saved: {self.case_count} cases completed")
        
    def load_test_cases(self) -> pd.DataFrame:
        """Load test cases from CSV file and normalize column names"""
        print(f"📊 Loading test cases from {self.csv_file}")
        df = pd.read_csv(self.csv_file, delimiter='|')

        # If dataset is the new OpenAI set, adapt columns
        if 'desc_1' in df.columns and 'procedure' in df.columns:
            # Use the first description as patient condition
            df['synthetic_patient_condition'] = df['desc_1']

            # Parse procedure JSON string into semicolon-separated list (matching old format)
            def _json_to_semicolon(text: str) -> str:
                """Return semicolon list of procedures rated *Usually appropriate* only (exclude `Usually not appropriate`)."""
                try:
                    proc_dict = json.loads(text)
                    ua = [
                        name
                        for name, rating in proc_dict.items()
                        if isinstance(rating, str) and rating.lower().startswith("usually appropriate")
                    ]
                    return ';'.join(ua)
                except Exception:
                    return ""

            df['appropriate_imaging'] = df['procedure'].apply(_json_to_semicolon)

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
            
            # Add timeout to prevent hanging
            import asyncio
            
            # Monitor memory before processing
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run the system
            start_time = time.time()
            try:
                # 5 minute timeout per case (complex workflow with GraphRAG + ColBERT + Neo4j)
                result = await asyncio.wait_for(
                    workflow.ainvoke({
                        'user_query': patient_condition,
                        'messages': []
                    }), 
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                print(f"\n⏰ Case timed out after 5 minutes")
                return {
                    'original_variant': original_variant,
                    'patient_condition': patient_condition,
                    'error': 'Timeout after 5 minutes',
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'exact_match': 0.0
                }
            
            execution_time = time.time() - start_time
            
            # Monitor memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
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
                'memory_used_mb': memory_used,
                'memory_after_mb': memory_after,
                **metrics,
                'acr_similarity': acr_recommendations.get('best_variant', {}).get('variant_similarity', 0.0),
                'selected_variant': acr_recommendations.get('best_variant', {}).get('variant_id', 'Unknown')
            }
            
            print(f"✅ P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, Mem: +{memory_used:.1f}MB (Total: {memory_after:.1f}MB)")
            
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
        
        # Check for existing checkpoint first
        if start_from == 0:  # Only load checkpoint if not manually specifying start
            checkpoint_start = self.load_checkpoint()
            if checkpoint_start > 0:
                start_from = checkpoint_start
        
        # Load test cases
        df = self.load_test_cases()
        
        # Limit cases if specified
        if max_cases:
            df = df.iloc[start_from:start_from + max_cases]
            print(f"📋 Evaluating {len(df)} cases (from index {start_from})")
        else:
            # If resuming, only process remaining cases
            df = df.iloc[start_from:]
            print(f"📋 Evaluating {len(df)} remaining cases (from index {start_from})")
        
        # Process in smaller batches to manage memory
        batch_size = 5  # Very small batches to avoid memory issues
        total_cases = len(df)
        
        print(f"\n🎯 Processing {total_cases} cases in batches of {batch_size}...")
        start_time = time.time()
        
        for batch_start in range(0, total_cases, batch_size):
            batch_end = min(batch_start + batch_size, total_cases)
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"\n🔄 Batch {batch_start//batch_size + 1}: Processing cases {start_from + batch_start + 1} to {start_from + batch_end}")
            
            # Initialize fresh LLM and workflow for each batch to prevent memory leaks
            print("\n🔧 Initializing fresh system for batch...")
            llm = ChatOllama(
                model="alibayram/medgemma:latest",
                temperature=0.1
            )
            workflow = create_enhanced_medical_workflow(llm, neo4j_password=os.getenv("NEO4J_PASSWORD"))
            
            # Process batch
            for idx, (original_idx, case) in enumerate(batch_df.iterrows()):
                current_case_num = batch_start + idx + 1
                total_case_num = start_from + batch_start + idx + 1
                print(f"\n[{current_case_num}/{total_cases}] (Global: {total_case_num}) ", end="")
                
                case_result = await self.evaluate_single_case(workflow, case)
                self._write_result_to_stream(case_result)
            
                # Save checkpoint every case in small batches
                self.save_checkpoint(start_from + batch_start + idx)
            
            # Cleanup after each batch
            print(f"\n🧹 Cleaning up batch {batch_start//batch_size + 1}...")
            
            # Step 1: Close ACR agent resources (ColBERT + Neo4j)
            try:
                # Get the ACR agent from workflow and close its resources
                if hasattr(workflow, 'nodes') and 'acr_retrieval' in workflow.nodes:
                    acr_agent = workflow.nodes['acr_retrieval']
                    if hasattr(acr_agent, 'close'):
                        acr_agent.close()
                        print("✅ ACR agent resources closed")
            except Exception as e:
                print(f"⚠️  Warning: Could not close ACR agent: {e}")
            
            # Step 2: Close LLM HTTP client
            try:
                if hasattr(llm, 'client') and hasattr(llm.client, 'aclose'):
                    await llm.client.aclose()
                    print("✅ LLM client closed")
                elif hasattr(llm, 'client') and hasattr(llm.client, 'close'):
                    llm.client.close()
                    print("✅ LLM client closed")
            except Exception as e:
                print(f"⚠️  Warning: Could not close LLM client: {e}")
            
            # Step 3: Close any Neo4j drivers in workflow
            try:
                if hasattr(workflow, 'nodes'):
                    for node_name, node in workflow.nodes.items():
                        if hasattr(node, 'driver') and hasattr(node.driver, 'close'):
                            node.driver.close()
                            print(f"✅ {node_name} Neo4j driver closed")
                        elif hasattr(node, 'rationale_retriever') and hasattr(node.rationale_retriever, 'close'):
                            node.rationale_retriever.close()
                            print(f"✅ {node_name} rationale retriever closed")
            except Exception as e:
                print(f"⚠️  Warning: Could not close workflow resources: {e}")
                
            # Step 4: Delete objects and force garbage collection
            del workflow
            del llm
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            print(f"🗑️  Garbage collected: {collected} objects")
            
            # Step 5: Force Ollama to unload models
            self._force_ollama_cleanup()
            
            # Save intermediate results after each batch
            self.save_intermediate_results(f"evaluation_results_batch_{batch_start//batch_size + 1}.json")
            
            # Brief pause to let system recover
            print(f"💤 Pausing 5 seconds for system recovery...")
            time.sleep(5)
            
            print(f"✅ Batch {batch_start//batch_size + 1} completed. Memory cleaned up.")
        
        total_time = time.time() - start_time
        
        # Close the results stream
        self._close_results_stream()
        
        # Calculate overall metrics
        self.calculate_overall_metrics()
        
        # Display results
        self.display_results(total_time)
        
        # Save final results
        self.save_results()
    
    def calculate_overall_metrics(self):
        """Calculate overall system metrics"""
        if self.case_count == 0:
            return
        
        # Read all results from streamed file
        all_results = []
        if hasattr(self, 'results_file_path') and os.path.exists(self.results_file_path):
            with open(self.results_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line.strip())
                            all_results.append(result)
                        except json.JSONDecodeError:
                            continue
        else:
            # Fallback to memory results
            all_results = self.results
        
        # Filter out error cases
        valid_results = [r for r in all_results if 'error' not in r]
        
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
            'total_cases': self.case_count,
            'valid_cases': n_valid,
            'error_cases': self.case_count - n_valid,
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
        
        # If we have a streamed file, copy it to the final results file
        if hasattr(self, 'results_file_path') and os.path.exists(self.results_file_path):
            # Read all results from stream
            all_results = []
            with open(self.results_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line.strip())
                            all_results.append(result)
                        except json.JSONDecodeError:
                            continue
            
            # Save as structured JSON
            with open(results_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'timestamp': timestamp,
                        'total_cases': self.case_count,
                        'csv_file': self.csv_file,
                        'streamed_from': self.results_file_path
                    },
                    'metrics': self.metrics,
                    'detailed_results': all_results
                }, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file} ({self.case_count} cases)")
            print(f"📄 Original stream: {self.results_file_path}")
        else:
            # Fallback for memory-based results
            with open(results_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'timestamp': timestamp,
                        'total_cases': self.case_count,
                        'csv_file': self.csv_file
                    },
                    'metrics': self.metrics,
                    'detailed_results': self.results
                }, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        # Save summary CSV
        summary_file = f"evaluation_summary_{timestamp}.csv"
        if self.case_count > 0:
            # Read results from stream to avoid large DataFrame in memory
            if hasattr(self, 'results_file_path') and os.path.exists(self.results_file_path):
                # Use pandas read_json with lines=True for JSONL format
                try:
                    df_results = pd.read_json(self.results_file_path, lines=True)
                    df_results.to_csv(summary_file, index=False)
                    print(f"\n💾 Summary CSV saved to: {summary_file}")
                except Exception as e:
                    print(f"\n⚠️  Could not create CSV: {e}")
            else:
                # Fallback to memory results
                df_results = pd.DataFrame(self.results)
                df_results.to_csv(summary_file, index=False)
                print(f"\n💾 Summary CSV saved to: {summary_file}")
    
    def save_intermediate_results(self, filename: str):
        """Save intermediate results during evaluation"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def check_evaluation_status(self):
        """Check current evaluation progress"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed = checkpoint_data.get('total_completed', 0)
                last_index = checkpoint_data.get('last_completed_index', -1)
                timestamp = checkpoint_data.get('timestamp', 'Unknown')
                
                print(f"📊 Current Status:")
                print(f"   Completed cases: {completed}")
                print(f"   Last completed index: {last_index}")
                print(f"   Next case to process: {last_index + 2}")
                print(f"   Last update: {timestamp}")
                return completed, last_index + 1
        else:
            print("📊 No previous evaluation found - starting fresh")
            return 0, 0

async def main():
    """Main evaluation function"""
    evaluator = MedicalSystemEvaluator()
    
    print("🔧 Enhanced evaluation with auto-resume capability")
    print("💡 The system will automatically resume from where it left off if interrupted")
    print("💡 Checkpoints are saved every 5 cases")
    
    try:
        # Run the full evaluation - it will auto-resume if needed
        await evaluator.evaluate_system(
            max_cases=5,  # Run all cases
            start_from=175     # Auto-resume will override this if checkpoint exists
        )
        
        print("\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        print("💾 Progress saved in checkpoint - run again to resume")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        print("💾 Progress saved in checkpoint - run again to resume")
        evaluator.save_checkpoint(len(evaluator.results) - 1)

if __name__ == "__main__":
    asyncio.run(main()) 