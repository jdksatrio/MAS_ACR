import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from neo4j import GraphDatabase
from dataclasses import dataclass
import logging
from datetime import datetime
import traceback

@dataclass
class EnrichmentResult:
    """Data structure for enrichment results."""
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

class PerplexityEnricher:
    """
    Enriches Neo4j ACR appropriateness criteria with clinical rationale, 
    evidence quality, and references using Perplexity API.
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password",
                 perplexity_api_key: str = None,
                 model: str = "llama-3.1-sonar-large-128k-online",
                 batch_size: int = 10,
                 delay_between_calls: float = 2.0,
                 max_retries: int = 3):
        """
        Initialize the enrichment system.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username  
            neo4j_password: Neo4j password
            perplexity_api_key: Perplexity API key
            model: Perplexity model to use
            batch_size: Number of procedures to process in each batch
            delay_between_calls: Delay between API calls to respect rate limits
            max_retries: Maximum number of retries for failed requests
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.model = model
        self.batch_size = batch_size
        self.delay_between_calls = delay_between_calls
        self.max_retries = max_retries
        
        # Setup logging
        self.log_filename = f'enrichment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Perplexity API
        self.perplexity_api_key = perplexity_api_key or self._get_api_key()
        if not self.perplexity_api_key:
            raise ValueError("Perplexity API key not provided. Please add PERPLEXITY_API_KEY to api_key.txt or set environment variable.")
        
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
        
        self.logger.info(f"🚀 Perplexity Enricher initialized")
        self.logger.info(f"📊 Model: {model}, Batch size: {batch_size}, API delay: {delay_between_calls}s")
    
    def _get_api_key(self) -> Optional[str]:
        """Get Perplexity API key from file or environment."""
        # Try reading from api_key.txt
        try:
            with open("api_key.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("PERPLEXITY_API_KEY"):
                        return line.split("=", 1)[1].strip()
        except FileNotFoundError:
            pass
        
        # Try environment variable
        return os.getenv("PERPLEXITY_API_KEY")
    
    def _test_connections(self):
        """Test Neo4j and Perplexity API connections."""
        # Test Neo4j
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    self.logger.info("✅ Neo4j connection successful")
                else:
                    raise Exception("Neo4j test query failed")
        except Exception as e:
            self.logger.error(f"❌ Neo4j connection failed: {e}")
            raise
        
        # Test Perplexity API
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Test connection"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.perplexity_headers,
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("✅ Perplexity API connection successful")
            else:
                raise Exception(f"API test failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"❌ Perplexity API connection failed: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def get_enrichment_queue(self) -> List[Tuple[str, str, str, str]]:
        """
        Get all condition-variant-procedure combinations that need enrichment.
        
        Returns:
            List of tuples: (condition_id, variant_id, procedure_id, appropriateness)
        """
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
        
        self.logger.info(f"📋 Found {len(combinations)} combinations needing enrichment")
        return combinations
    
    def create_enrichment_prompt(self, condition: str, variant: str, procedure: str, appropriateness: str) -> str:
        """Create a detailed prompt for Perplexity API to get ACR-specific information."""
        
        prompt = f"""Based on the American College of Radiology (ACR) Appropriateness Criteria, provide detailed information about the following medical imaging recommendation:

**Medical Condition:** {condition}
**Clinical Variant:** {variant}
**Imaging Procedure:** {procedure}
**ACR Appropriateness Rating:** {appropriateness.replace('_', ' ').title()}

Please provide comprehensive information in the following structured format:

1. **CLINICAL_RATIONALE:** Explain the clinical reasoning behind why this imaging procedure is rated as "{appropriateness.replace('_', ' ').lower()}" for this specific condition and variant. Include:
   - Relevant pathophysiology and clinical presentation
   - Diagnostic considerations and differential diagnosis
   - How this imaging contributes to patient management
   - Risk-benefit considerations
   - Alternative imaging options and their limitations

2. **EVIDENCE_QUALITY:** Describe the quality and strength of scientific evidence supporting this recommendation. Include:
   - Level of evidence (systematic reviews, RCTs, observational studies, expert opinion)
   - Sample sizes and study populations
   - Strength of recommendation according to ACR criteria
   - Any limitations or gaps in the evidence
   - Recent updates or emerging evidence

3. **REFERENCES:** List the most relevant and authoritative sources supporting this recommendation:
   - ACR Appropriateness Criteria documents
   - Recent peer-reviewed medical literature
   - Clinical practice guidelines from professional societies
   - Systematic reviews and meta-analyses
   - Key studies that established the evidence base

Please focus specifically on ACR Appropriateness Criteria and ensure all information is accurate, current, and evidence-based. Format your response exactly as:

CLINICAL_RATIONALE: [detailed clinical explanation]

EVIDENCE_QUALITY: [evidence assessment and quality description]

REFERENCES: [list of relevant sources with proper citations]"""

        return prompt
    
    def query_perplexity_with_retry(self, prompt: str) -> Dict[str, str]:
        """Query Perplexity API with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert radiologist and medical researcher specializing in ACR Appropriateness Criteria. Provide accurate, evidence-based information about medical imaging recommendations with proper clinical reasoning and current evidence assessment."
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
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    citations = data.get('citations', [])
                    
                    # Parse the structured response
                    parsed = self._parse_perplexity_response(content, citations)
                    parsed['processing_time'] = processing_time
                    return parsed
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_msg = f"Perplexity API error: {response.status_code} - {response.text}"
                    if attempt == self.max_retries - 1:  # Last attempt
                        self.logger.error(error_msg)
                        return {
                            "clinical_rationale": "",
                            "evidence_quality": "",
                            "references": "",
                            "success": False,
                            "error": error_msg,
                            "processing_time": processing_time
                        }
                    else:
                        self.logger.warning(f"API error on attempt {attempt + 1}, retrying: {error_msg}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                        
            except Exception as e:
                error_msg = f"Exception querying Perplexity (attempt {attempt + 1}): {str(e)}"
                if attempt == self.max_retries - 1:  # Last attempt
                    self.logger.error(error_msg)
                    return {
                        "clinical_rationale": "",
                        "evidence_quality": "",
                        "references": "",
                        "success": False,
                        "error": error_msg,
                        "processing_time": 0.0
                    }
                else:
                    self.logger.warning(f"Exception on attempt {attempt + 1}, retrying: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        # Should never reach here, but just in case
        return {
            "clinical_rationale": "",
            "evidence_quality": "",
            "references": "",
            "success": False,
            "error": "All retry attempts failed",
            "processing_time": 0.0
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
            self.logger.error(f"Error parsing Perplexity response: {str(e)}")
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
                p.enrichment_model = $model
            RETURN p.id as updated_id
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "procedure_id": procedure_id,
                    "clinical_rationale": enrichment_data["clinical_rationale"],
                    "evidence_quality": enrichment_data["evidence_quality"], 
                    "references": enrichment_data["references"],
                    "model": self.model
                })
                
                record = result.single()
                if record:
                    return True
                else:
                    self.logger.error(f"❌ Failed to update procedure: {procedure_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating Neo4j for procedure {procedure_id}: {str(e)}")
            return False
    
    def save_progress(self, results: List[EnrichmentResult], filename: str = None):
        """Save enrichment progress to CSV file."""
        try:
            if not filename:
                filename = f"enrichment_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            df = pd.DataFrame([{
                'condition': r.condition,
                'variant': r.variant,
                'procedure': r.procedure,
                'appropriateness': r.appropriateness,
                'clinical_rationale': r.clinical_rationale,
                'evidence_quality': r.evidence_quality,
                'references': r.references,
                'success': r.success,
                'error_message': r.error_message,
                'processing_time': r.processing_time
            } for r in results])
            
            df.to_csv(filename, index=False)
            self.logger.info(f"💾 Progress saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Error saving progress: {str(e)}")
            return None
    
    def enrich_procedures(self, 
                         max_procedures: Optional[int] = None, 
                         start_from: int = 0,
                         save_every: int = 10) -> List[EnrichmentResult]:
        """
        Main method to enrich procedures with Perplexity API data.
        
        Args:
            max_procedures: Maximum number of procedures to process (None for all)
            start_from: Index to start processing from (for resuming)
            save_every: Save progress every N procedures
            
        Returns:
            List of EnrichmentResult objects
        """
        self.logger.info("🚀 Starting enrichment process...")
        
        # Get the enrichment queue
        queue = self.get_enrichment_queue()
        
        if start_from > 0:
            queue = queue[start_from:]
            self.logger.info(f"🔄 Resuming from index {start_from}")
        
        if max_procedures:
            queue = queue[:max_procedures]
            self.logger.info(f"📊 Processing {len(queue)} procedures (limited by max_procedures)")
        
        results = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, (condition, variant, procedure, appropriateness) in enumerate(queue):
            actual_index = i + start_from
            
            self.logger.info(f"🔄 Processing {actual_index + 1}/{len(queue) + start_from}: {procedure}")
            
            # Create prompt and query Perplexity
            prompt = self.create_enrichment_prompt(condition, variant, procedure, appropriateness)
            perplexity_response = self.query_perplexity_with_retry(prompt)
            
            # Create enrichment result object
            result = EnrichmentResult(
                condition=condition,
                variant=variant, 
                procedure=procedure,
                appropriateness=appropriateness,
                clinical_rationale=perplexity_response["clinical_rationale"],
                evidence_quality=perplexity_response["evidence_quality"],
                references=perplexity_response["references"],
                success=perplexity_response["success"],
                error_message=perplexity_response.get("error"),
                processing_time=perplexity_response.get("processing_time", 0.0)
            )
            
            # Update Neo4j if successful
            if result.success:
                neo4j_success = self.update_neo4j_procedure(procedure, perplexity_response)
                result.success = neo4j_success
                if neo4j_success:
                    successful += 1
                    self.logger.info(f"✅ Successfully enriched: {procedure}")
                else:
                    failed += 1
                    result.error_message = "Neo4j update failed"
            else:
                failed += 1
                self.logger.warning(f"❌ Failed to enrich: {procedure} - {result.error_message}")
            
            results.append(result)
            
            # Save progress periodically
            if (i + 1) % save_every == 0:
                self.save_progress(results)
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (i + 1)
                remaining_time = avg_time * (len(queue) - i - 1)
                
                self.logger.info(f"📊 Progress: {i + 1}/{len(queue)} ({(i+1)/len(queue)*100:.1f}%)")
                self.logger.info(f"⏱️ Elapsed: {elapsed_time/60:.1f}m, Remaining: {remaining_time/60:.1f}m")
                self.logger.info(f"📈 Success rate: {successful}/{successful+failed} ({successful/(successful+failed)*100:.1f}%)")
            
            # Respect rate limits
            if i < len(queue) - 1:  # Don't delay after the last item
                time.sleep(self.delay_between_calls)
        
        # Final save
        final_filename = self.save_progress(results)
        
        total_time = time.time() - start_time
        self.logger.info(f"🎉 Enrichment complete!")
        self.logger.info(f"📊 Summary: {successful} successful, {failed} failed out of {len(results)} total")
        self.logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"💾 Results saved to: {final_filename}")
        
        return results
    
    def get_enrichment_statistics(self) -> Dict[str, int]:
        """Get current enrichment statistics."""
        query = """
        MATCH (c:Condition)-[:HAS_VARIANT]->(v:Variant)-[r]->(p:Procedure)
        WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        AND type(r) IN ["USUALLY_APPROPRIATE", "MAY_BE_APPROPRIATE", "USUALLY_NOT_APPROPRIATE"]
        WITH count(*) as total_combinations
        MATCH (p:Procedure)
        WHERE p.gid = "ACR_BATCH_1" AND p.clinical_rationale IS NOT NULL
        RETURN total_combinations, count(p) as enriched_procedures
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if record:
                total = record["total_combinations"]
                enriched = record["enriched_procedures"]
                
                stats = {
                    "total_combinations": total,
                    "enriched_procedures": enriched,
                    "remaining_procedures": total - enriched,
                    "completion_percentage": round((enriched / total) * 100, 2) if total > 0 else 0
                }
                
                self.logger.info(f"📊 Enrichment Statistics:")
                self.logger.info(f"   Total combinations: {stats['total_combinations']:,}")
                self.logger.info(f"   Enriched: {stats['enriched_procedures']:,}")
                self.logger.info(f"   Remaining: {stats['remaining_procedures']:,}")
                self.logger.info(f"   Completion: {stats['completion_percentage']}%")
                
                return stats
            
        return {}

def main():
    """Main function to run the enrichment process."""
    
    print("🚀 ACR Appropriateness Criteria Enrichment System")
    print("=" * 60)
    
    # Initialize enricher
    try:
        enricher = PerplexityEnricher(
            model="llama-3.1-sonar-large-128k-online",  # Best balance of quality/cost
            batch_size=10,
            delay_between_calls=2.0,  # 2 second delay for safety
            max_retries=3
        )
    except Exception as e:
        print(f"❌ Failed to initialize enricher: {e}")
        print("\n💡 To get started:")
        print("1. Sign up at https://www.perplexity.ai/")
        print("2. Go to https://www.perplexity.ai/settings/api")
        print("3. Generate an API key")
        print("4. Add 'PERPLEXITY_API_KEY=your_key_here' to api_key.txt")
        return
    
    try:
        # Get current statistics
        enricher.get_enrichment_statistics()
        
        # Ask user for confirmation
        print("\n🎯 Ready to start enrichment!")
        print("💰 Estimated cost: ~$45 for complete enrichment")
        print("⏱️ Estimated time: ~6 hours")
        
        response = input("\nProceed with enrichment? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\n🚀 Starting enrichment process...")
            
            # Start enrichment
            results = enricher.enrich_procedures(
                max_procedures=None,  # Process all
                start_from=0,
                save_every=5  # Save every 5 procedures
            )
            
            # Final statistics
            enricher.get_enrichment_statistics()
            
        else:
            print("👋 Enrichment cancelled by user")
            
    except KeyboardInterrupt:
        print("\n⏸️ Enrichment interrupted by user")
        print("💾 Progress has been saved and can be resumed")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📋 Full traceback logged to: {enricher.log_filename}")
        
    finally:
        enricher.close()

if __name__ == "__main__":
    main() 