import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from neo4j import GraphDatabase
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import traceback
import signal
import sys
from pathlib import Path

@dataclass
class EnrichmentResult:
    """Data structure for enrichment results."""
    index: int
    condition: str
    variant: str
    procedure: str
    appropriateness: str
    clinical_rationale: str
    evidence_quality: str
    references: str
    success: bool
    error_message: str = None
    processing_time: float = 0.0
    timestamp: str = None

class StreamingPerplexityEnricher:
    """
    Streaming Perplexity enrichment system with real-time output and safe cancellation.
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password",
                 model: str = "llama-3.1-sonar-large-128k-online",
                 delay_between_calls: float = 2.0,
                 max_retries: int = 3,
                 save_every: int = 5):
        """Initialize the streaming enrichment system."""
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.model = model
        self.delay_between_calls = delay_between_calls
        self.max_retries = max_retries
        self.save_every = save_every
        
        # Progress tracking
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.start_time = None
        self.is_cancelled = False
        
        # File management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f'enrichment_{self.session_id}.log'
        self.progress_filename = f'enrichment_progress_{self.session_id}.csv'
        self.state_filename = f'enrichment_state_{self.session_id}.json'
        
        # Setup logging with both file and console output
        self.setup_logging()
        
        # Get API key
        self.perplexity_api_key = self._get_api_key()
        if not self.perplexity_api_key:
            raise ValueError("❌ Perplexity API key not found. Please export PERPLEXITY_API_KEY environment variable.")
        
        self.perplexity_headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Test connections
        self._test_connections()
        
        # Setup signal handlers for safe cancellation
        self.setup_signal_handlers()
        
        # Results storage
        self.results = []
        
        self.print_header()
    
    def setup_logging(self):
        """Setup logging with both file and console output."""
        
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful cancellation."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received cancellation signal (Ctrl+C)")
            print("💾 Saving progress and shutting down gracefully...")
            self.is_cancelled = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def print_header(self):
        """Print the enrichment system header."""
        print("🚀 STREAMING PERPLEXITY ENRICHMENT SYSTEM")
        print("=" * 60)
        print(f"📅 Session ID: {self.session_id}")
        print(f"🤖 Model: {self.model}")
        print(f"⏱️ API Delay: {self.delay_between_calls}s")
        print(f"🔄 Max Retries: {self.max_retries}")
        print(f"💾 Save Every: {self.save_every} procedures")
        print(f"📋 Log File: {self.log_filename}")
        print(f"📊 Progress File: {self.progress_filename}")
        print("=" * 60)
    
    def _get_api_key(self) -> Optional[str]:
        """Get Perplexity API key from environment or file."""
        # Try environment first
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if api_key:
            return api_key.strip()
        
        # Try api_key.txt file
        try:
            with open("api_key.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("PERPLEXITY_API_KEY"):
                        return line.split("=", 1)[1].strip()
        except FileNotFoundError:
            pass
        
        return None
    
    def _test_connections(self):
        """Test Neo4j and Perplexity API connections."""
        # Test Neo4j
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    print("✅ Neo4j connection successful")
                else:
                    raise Exception("Neo4j test query failed")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            raise
        
        # Test Perplexity API
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.perplexity_headers,
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Perplexity API connection successful")
            else:
                raise Exception(f"API test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Perplexity API connection failed: {e}")
            raise
    
    def close(self):
        """Close connections and save final state."""
        if self.driver:
            self.driver.close()
        
        # Save final state
        if self.results:
            self.save_progress()
            self.save_state()
    
    def get_enrichment_queue(self) -> List[Tuple[str, str, str, str]]:
        """Get all combinations that need enrichment."""
        query = """
        MATCH (c:Condition)-[:HAS_VARIANT]->(v:Variant)-[r]->(p:Procedure)
        WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        AND type(r) IN ["USUALLY_APPROPRIATE", "MAY_BE_APPROPRIATE", "USUALLY_NOT_APPROPRIATE"]
        AND (p.clinical_rationale IS NULL OR p.evidence_quality IS NULL OR p.references IS NULL)
        RETURN c.id as condition, v.id as variant, p.id as procedure, type(r) as appropriateness
        ORDER BY c.id, v.id, p.id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            combinations = [(record["condition"], record["variant"], record["procedure"], record["appropriateness"]) 
                          for record in result]
        
        print(f"📋 Found {len(combinations)} combinations needing enrichment")
        return combinations
    
    def create_enrichment_prompt(self, condition: str, variant: str, procedure: str, appropriateness: str) -> str:
        """Create enrichment prompt for Perplexity API."""
        return f"""Based on the American College of Radiology (ACR) Appropriateness Criteria, provide detailed information about:

**Medical Condition:** {condition}
**Clinical Variant:** {variant}
**Imaging Procedure:** {procedure}
**ACR Appropriateness Rating:** {appropriateness.replace('_', ' ').title()}

Please provide comprehensive information in the following structured format:

1. **CLINICAL_RATIONALE:** Explain the clinical reasoning behind why this imaging procedure is rated as "{appropriateness.replace('_', ' ').lower()}" for this specific condition and variant. Include relevant pathophysiology, diagnostic considerations, clinical decision-making factors, risk-benefit analysis, and alternative imaging options.

2. **EVIDENCE_QUALITY:** Describe the quality and strength of scientific evidence supporting this recommendation. Include level of evidence, study types, sample sizes, strength of recommendation according to ACR criteria, any limitations or gaps in the evidence, and recent updates.

3. **REFERENCES:** List the most relevant and authoritative sources supporting this recommendation including ACR Appropriateness Criteria documents, recent peer-reviewed medical literature, clinical practice guidelines, systematic reviews, and key studies.

Please focus specifically on ACR Appropriateness Criteria and ensure all information is accurate, current, and evidence-based. Format your response exactly as:

CLINICAL_RATIONALE: [detailed clinical explanation]

EVIDENCE_QUALITY: [evidence assessment and quality description]

REFERENCES: [list of relevant sources with proper citations]"""
    
    def query_perplexity_with_retry(self, prompt: str, procedure_name: str) -> Dict[str, str]:
        """Query Perplexity API with retry logic and real-time feedback."""
        
        for attempt in range(self.max_retries):
            if self.is_cancelled:
                return self._get_cancelled_response()
            
            try:
                print(f"   🔄 API call attempt {attempt + 1}/{self.max_retries}...")
                start_time = time.time()
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert radiologist and medical researcher specializing in ACR Appropriateness Criteria. Provide accurate, evidence-based information about medical imaging recommendations."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "return_citations": True,
                    "search_domain_filter": ["acr.org"],
                    "return_images": False,
                    "return_related_questions": False
                }
                
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=self.perplexity_headers,
                    json=payload,
                    timeout=60
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"   ✅ API response received ({processing_time:.1f}s)")
                    
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    citations = data.get('citations', [])
                    
                    parsed = self._parse_perplexity_response(content, citations)
                    parsed['processing_time'] = processing_time
                    
                    if parsed['success']:
                        print(f"   ✅ Response parsed successfully")
                    else:
                        print(f"   ⚠️ Response parsing had issues")
                    
                    return parsed
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * 5
                    print(f"   ⏳ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_msg = f"API error: {response.status_code} - {response.text[:200]}"
                    if attempt == self.max_retries - 1:
                        print(f"   ❌ {error_msg}")
                        return self._get_error_response(error_msg, processing_time)
                    else:
                        print(f"   ⚠️ {error_msg} (retrying...)")
                        time.sleep(2 ** attempt)
                        continue
                        
            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                if attempt == self.max_retries - 1:
                    print(f"   ❌ {error_msg}")
                    return self._get_error_response(error_msg, 0.0)
                else:
                    print(f"   ⚠️ {error_msg} (retrying...)")
                    time.sleep(2 ** attempt)
                    continue
        
        return self._get_error_response("All retry attempts failed", 0.0)
    
    def _get_cancelled_response(self) -> Dict[str, str]:
        """Return response for cancelled operation."""
        return {
            "clinical_rationale": "",
            "evidence_quality": "",
            "references": "",
            "success": False,
            "error": "Cancelled by user",
            "processing_time": 0.0
        }
    
    def _get_error_response(self, error_msg: str, processing_time: float) -> Dict[str, str]:
        """Return error response."""
        return {
            "clinical_rationale": "",
            "evidence_quality": "",
            "references": "",
            "success": False,
            "error": error_msg,
            "processing_time": processing_time
        }
    
    def _parse_perplexity_response(self, content: str, citations: List[str]) -> Dict[str, str]:
        """Parse the structured response from Perplexity API."""
        try:
            clinical_rationale = ""
            evidence_quality = ""
            references = ""
            
            # Split content into lines and process
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Check for section headers
                if line.upper().startswith('CLINICAL_RATIONALE:'):
                    current_section = 'clinical_rationale'
                    clinical_rationale = line[len('CLINICAL_RATIONALE:'):].strip()
                elif line.upper().startswith('EVIDENCE_QUALITY:'):
                    current_section = 'evidence_quality'
                    evidence_quality = line[len('EVIDENCE_QUALITY:'):].strip()
                elif line.upper().startswith('REFERENCES:'):
                    current_section = 'references'
                    references = line[len('REFERENCES:'):].strip()
                elif line and current_section:
                    # Continue adding to the current section
                    if current_section == 'clinical_rationale':
                        clinical_rationale += " " + line
                    elif current_section == 'evidence_quality':
                        evidence_quality += " " + line
                    elif current_section == 'references':
                        references += " " + line
            
            # Add citations if available
            if citations:
                citation_text = "\n\nCitations:\n" + "\n".join([f"- {cite}" for cite in citations])
                references += citation_text
            
            return {
                "clinical_rationale": clinical_rationale.strip(),
                "evidence_quality": evidence_quality.strip(),
                "references": references.strip(),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "clinical_rationale": content,  # Fallback to full content
                "evidence_quality": "",
                "references": "\n".join(citations) if citations else "",
                "success": False,
                "error": f"Parse error: {str(e)}"
            }
    
    def update_neo4j_procedure(self, procedure_id: str, enrichment_data: Dict[str, str]) -> bool:
        """Update a procedure in Neo4j with enrichment data."""
        try:
            query = """
            MATCH (p:Procedure {id: $procedure_id, gid: "ACR_BATCH_1"})
            SET p.clinical_rationale = $clinical_rationale,
                p.evidence_quality = $evidence_quality,
                p.references = $references,
                p.enriched_timestamp = datetime(),
                p.enrichment_source = "perplexity_api",
                p.enrichment_model = $model,
                p.enrichment_session = $session_id
            RETURN p.id as updated_id
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "procedure_id": procedure_id,
                    "clinical_rationale": enrichment_data["clinical_rationale"],
                    "evidence_quality": enrichment_data["evidence_quality"], 
                    "references": enrichment_data["references"],
                    "model": self.model,
                    "session_id": self.session_id
                })
                
                record = result.single()
                return record is not None
                    
        except Exception as e:
            print(f"   ❌ Neo4j update failed: {str(e)}")
            return False
    
    def save_progress(self):
        """Save progress to CSV file."""
        try:
            if not self.results:
                return
            
            df = pd.DataFrame([asdict(r) for r in self.results])
            df.to_csv(self.progress_filename, index=False)
            
        except Exception as e:
            print(f"❌ Error saving progress: {str(e)}")
    
    def save_state(self):
        """Save current state for resuming."""
        try:
            state = {
                "session_id": self.session_id,
                "processed_count": self.processed_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "is_cancelled": self.is_cancelled,
                "model": self.model,
                "last_processed_index": len(self.results) - 1 if self.results else -1
            }
            
            with open(self.state_filename, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving state: {str(e)}")
    
    def print_progress_summary(self, current_index: int, total_count: int):
        """Print progress summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress_pct = (current_index / total_count) * 100 if total_count > 0 else 0
        
        avg_time_per_item = elapsed / current_index if current_index > 0 else 0
        remaining_items = total_count - current_index
        eta_seconds = remaining_items * avg_time_per_item
        
        success_rate = (self.success_count / current_index * 100) if current_index > 0 else 0
        
        print(f"\n📊 PROGRESS SUMMARY")
        print(f"   Progress: {current_index}/{total_count} ({progress_pct:.1f}%)")
        print(f"   Success: {self.success_count} ({success_rate:.1f}%)")
        print(f"   Failures: {self.failure_count}")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   ETA: {eta_seconds/60:.1f} minutes")
        print(f"   Avg time/item: {avg_time_per_item:.1f}s")
    
    def enrich_procedures_streaming(self, 
                                  max_procedures: Optional[int] = None, 
                                  start_from: int = 0) -> List[EnrichmentResult]:
        """Main streaming enrichment method."""
        
        print(f"\n🚀 Starting streaming enrichment...")
        self.start_time = time.time()
        
        # Get enrichment queue
        queue = self.get_enrichment_queue()
        
        if start_from > 0:
            queue = queue[start_from:]
            print(f"🔄 Resuming from index {start_from}")
        
        if max_procedures:
            queue = queue[:max_procedures]
            print(f"📊 Processing {len(queue)} procedures (limited)")
        
        total_count = len(queue)
        print(f"📋 Total procedures to process: {total_count}")
        print("⏸️ Press Ctrl+C to gracefully cancel and save progress")
        print("=" * 60)
        
        for i, (condition, variant, procedure, appropriateness) in enumerate(queue):
            if self.is_cancelled:
                print(f"\n🛑 Enrichment cancelled by user at item {i}")
                break
            
            current_index = i + start_from + 1
            
            print(f"\n🔄 [{current_index}/{total_count + start_from}] Processing: {procedure}")
            print(f"   📋 Condition: {condition}")
            print(f"   🔍 Variant: {variant}")
            print(f"   📊 Rating: {appropriateness.replace('_', ' ').title()}")
            
            # Create prompt and query Perplexity
            prompt = self.create_enrichment_prompt(condition, variant, procedure, appropriateness)
            perplexity_response = self.query_perplexity_with_retry(prompt, procedure)
            
            # Create enrichment result
            result = EnrichmentResult(
                index=current_index - 1,
                condition=condition,
                variant=variant,
                procedure=procedure,
                appropriateness=appropriateness,
                clinical_rationale=perplexity_response["clinical_rationale"],
                evidence_quality=perplexity_response["evidence_quality"],
                references=perplexity_response["references"],
                success=perplexity_response["success"],
                error_message=perplexity_response.get("error"),
                processing_time=perplexity_response.get("processing_time", 0.0),
                timestamp=datetime.now().isoformat()
            )
            
            # Update Neo4j if successful
            if result.success and not self.is_cancelled:
                print(f"   💾 Updating Neo4j...")
                neo4j_success = self.update_neo4j_procedure(procedure, perplexity_response)
                result.success = neo4j_success
                
                if neo4j_success:
                    self.success_count += 1
                    print(f"   ✅ Successfully enriched and saved to Neo4j")
                else:
                    self.failure_count += 1
                    result.error_message = "Neo4j update failed"
                    print(f"   ❌ API successful but Neo4j update failed")
            else:
                self.failure_count += 1
                print(f"   ❌ Enrichment failed: {result.error_message}")
            
            self.processed_count += 1
            self.results.append(result)
            
            # Save progress periodically
            if self.processed_count % self.save_every == 0:
                print(f"   💾 Saving progress...")
                self.save_progress()
                self.save_state()
                self.print_progress_summary(current_index, total_count + start_from)
            
            # Respect rate limits (unless cancelled)
            if not self.is_cancelled and i < len(queue) - 1:
                print(f"   ⏱️ Waiting {self.delay_between_calls}s...")
                time.sleep(self.delay_between_calls)
        
        # Final save
        print(f"\n💾 Saving final results...")
        self.save_progress()
        self.save_state()
        
        # Final summary
        total_time = time.time() - self.start_time
        print(f"\n🎉 ENRICHMENT SESSION COMPLETE")
        print(f"=" * 50)
        print(f"📊 Total processed: {self.processed_count}")
        print(f"✅ Successful: {self.success_count}")
        print(f"❌ Failed: {self.failure_count}")
        print(f"📈 Success rate: {self.success_count/self.processed_count*100:.1f}%" if self.processed_count > 0 else "N/A")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📋 Log file: {self.log_filename}")
        print(f"📊 Progress file: {self.progress_filename}")
        print(f"💾 State file: {self.state_filename}")
        
        if self.is_cancelled:
            print(f"⏸️ Session was cancelled - you can resume later")
        
        return self.results

def main():
    """Main function to run the streaming enrichment."""
    
    # Initialize enricher
    try:
        enricher = StreamingPerplexityEnricher(
            model="llama-3.1-sonar-large-128k-online",
            delay_between_calls=2.0,
            max_retries=3,
            save_every=5
        )
    except Exception as e:
        print(f"❌ Failed to initialize enricher: {e}")
        return
    
    try:
        # Get current statistics
        print(f"\n📊 Checking current enrichment status...")
        stats = enricher.get_enrichment_statistics()
        
        # Ask for confirmation
        print(f"\n🎯 Ready to start streaming enrichment!")
        print(f"💰 Estimated cost: ~$45 for complete enrichment")
        print(f"⏱️ Estimated time: ~6 hours")
        print(f"💾 Progress saved every 5 procedures")
        print(f"⏸️ Safe cancellation with Ctrl+C")
        
        response = input(f"\nProceed with enrichment? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            # Start streaming enrichment
            results = enricher.enrich_procedures_streaming()
            
        else:
            print("👋 Enrichment cancelled by user")
            
    except KeyboardInterrupt:
        print(f"\n⏸️ Enrichment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📋 Check log file: {enricher.log_filename}")
        
    finally:
        enricher.close()

def get_enrichment_statistics():
    """Get current enrichment statistics from Neo4j."""
    try:
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
        
        query = """
        MATCH (c:Condition)-[:HAS_VARIANT]->(v:Variant)-[r]->(p:Procedure)
        WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        AND type(r) IN ["USUALLY_APPROPRIATE", "MAY_BE_APPROPRIATE", "USUALLY_NOT_APPROPRIATE"]
        WITH count(*) as total_combinations
        MATCH (p:Procedure)
        WHERE p.gid = "ACR_BATCH_1" AND p.clinical_rationale IS NOT NULL
        RETURN total_combinations, count(p) as enriched_procedures
        """
        
        with driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if record:
                total = record["total_combinations"]
                enriched = record["enriched_procedures"]
                
                print(f"📊 Current enrichment status:")
                print(f"   Total combinations: {total:,}")
                print(f"   Already enriched: {enriched:,}")
                print(f"   Remaining: {total - enriched:,}")
                print(f"   Completion: {enriched/total*100:.1f}%")
                
                return {
                    "total": total,
                    "enriched": enriched,
                    "remaining": total - enriched
                }
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return None

if __name__ == "__main__":
    # Show current stats first
    get_enrichment_statistics()
    
    # Run main enrichment
    main() 