from typing import Dict, Any, List, Optional
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import sys
import json

# Add the retrieve_acr path to import the core functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieve_acr'))
from medical_procedure_recommender_vectorized import MedicalProcedureRecommenderVectorized
from colbert_acr_retriever import ColBERTACRRetriever


class EnhancedACRAgent:
    """
    Enhanced ACR Agent with LLM-powered query optimization for better ACR condition matching.
    
    This agent uses Chain of Thought reasoning to:
    1. Analyze the medical query for ACR-relevant components  
    2. Understand ACR condition naming patterns and structure
    3. Formulate optimal search queries for ACR database
    4. Execute strategic retrieval with multiple query approaches
    
    Supports both Neo4j vectorized search and high-performance ColBERT retrieval.
    """
    
    def __init__(
        self,
        llm,
        colbert_index_path: str = None,
        neo4j_password: str = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j"
    ):
        self.llm = llm
        self.colbert_index_path = colbert_index_path
        self.neo4j_password = neo4j_password
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.colbert_retriever = None
        self.neo4j_recommender = None
        self.initialized = False
    
    async def initialize(self, neo4j_password: str = None):
        """Initialize ColBERT for variant search and Neo4j for procedure fetching"""
        if self.initialized:
            return
            
        try:
            # Initialize ColBERT for variant search
            self.colbert_retriever = ColBERTACRRetriever(
                index_path=self.colbert_index_path,
                debug=True
            )
            
            # Initialize Neo4j for procedure fetching
            password = neo4j_password or self.neo4j_password or os.environ.get("NEO4J_PASSWORD", "medgraphrag")
            self.neo4j_recommender = MedicalProcedureRecommenderVectorized(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=password,
                embedding_provider="pubmedbert",
                embedding_model="NeuML/pubmedbert-base-embeddings"
            )
            
            self.initialized = True
            print(f"Enhanced ACR Agent initialized with ColBERT + Neo4j")
            
        except Exception as e:
            print(f"Enhanced ACR Agent initialization failed: {e}")
            raise
    
    def close(self):
        """Close connections"""
        if self.colbert_retriever:
            self.colbert_retriever.close()
        if self.neo4j_recommender:
            self.neo4j_recommender.close()
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent execution with LLM-powered query optimization
        """
        if not self.initialized:
            await self.initialize(state.get("neo4j_password"))
        
        user_query = state.get("user_query", "")
        if not user_query:
            # Extract query from messages if not in state
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    user_query = message.content
                    break
        
        if not user_query:
            return self._create_error_response("No user query found")
        
        try:
            # Step 1: LLM analyzes query and formulates optimal ACR search queries
            query_analysis = await self._analyze_query_for_acr(user_query)
            
            # Step 2: Execute strategic ACR retrieval with optimized queries
            acr_results = await self._execute_acr_retrieval(user_query, query_analysis)
            
            # Step 3: LLM evaluates and selects best results
            final_recommendations = await self._evaluate_acr_results(user_query, query_analysis, acr_results)
            
            # Create response message
            response_message = AIMessage(
                content=f"Enhanced ACR Agent optimized query and retrieved procedure recommendations",
                additional_kwargs={
                    "query_analysis": query_analysis,
                    "acr_results": acr_results,
                    "final_recommendations": final_recommendations,
                    "source": "enhanced_acr_agent"
                }
            )
            
            return {
                "messages": [response_message],
                "acr_recommendations": final_recommendations,
                "acr_analysis": query_analysis,
                "next_step": "enhanced_supervisor"
            }
            
        except Exception as e:
            import traceback
            print(f"Enhanced ACR Agent Exception:")
            print(f"   Error: {str(e)}")
            print(f"   Type: {type(e)}")
            print(f"   Traceback:")
            traceback.print_exc()
            return self._create_error_response(f"Enhanced ACR Agent failed: {str(e)}")
    

    
    async def _analyze_query_for_acr(self, user_query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze the medical query and formulate optimal ACR search queries
        """
        analysis_prompt = f"""
        You are an expert ACR (American College of Radiology) search strategist with deep medical knowledge. Analyze this medical query and formulate the most effective search queries for the ACR Appropriateness Criteria database.

        **Medical Query:** {user_query}

        **CRITICAL SUCCESS FACTORS:**
        1. **PRESERVE KEY TERMS**: Keep important medical terms from the original query - they often match ACR terminology exactly
        2. **MULTIPLE QUERY STRATEGIES**: Generate several different query formulations, not just one
        3. **ACR TERMINOLOGY MAPPING**: Translate to specific ACR language patterns (e.g., "follow-up" → "surveillance", "after surgery" → "post-surgical")

        **MEDICAL REASONING FRAMEWORK:**
        
        1. **SYMPTOM ANALYSIS**: What does each symptom tell you anatomically and pathologically?
        2. **ANATOMICAL DEDUCTION**: Which body system/organ is most likely involved?
        3. **CONDITION TRANSLATION**: What is the underlying medical condition or diagnosis?
        4. **CLINICAL WORKFLOW**: Is this initial evaluation, follow-up, or surveillance?
        5. **ACR MAPPING**: How would ACR categorize this condition?

        **QUERY TRANSFORMATION EXAMPLES:**
        
        Example 1: "routine monitoring following surgical excision of a pituitary adenoma"
        → Keep Key Terms: "pituitary", "surgical excision", "monitoring"
        → ACR Translation: "postpituitary surveillance", "sellar mass resection follow-up", "pituitary surgery surveillance"
        → Primary Search: "postpituitary surveillance"
        
        Example 2: "Blood in stool, bright red"
        → Keep Key Terms: "blood", "stool", "bright red"
        → ACR Translation: "lower gastrointestinal bleeding", "hematochezia", "rectal bleeding"
        → Primary Search: "lower gastrointestinal tract bleeding"
        
        Example 3: "Right upper quadrant pain + fever + elevated WBC"
        → Keep Key Terms: "right upper quadrant", "fever", "elevated WBC"
        → ACR Translation: "acute biliary disease", "cholangitis suspected", "biliary infection"
        → Primary Search: "right upper quadrant pain fever elevated WBC"
        
        Example 4: "diagnosed with proximal deep vein thrombosis, suitable candidate for anticoagulation"
        → Keep Key Terms: "diagnosed", "proximal", "deep vein thrombosis", "anticoagulation"
        → ACR Translation: "acute venous thromboembolism anticoagulation", "confirmed DVT treatment", "proximal DVT no contraindication"
        → Primary Search: "acute venous thromboembolism anticoagulation"
        
        Example 5: "first seizure episode, having no history of recent trauma or injury"
        → Keep Key Terms: "first seizure", "no history", "trauma", "injury"
        → Negation Detection: "no history of trauma" → "unrelated to trauma"
        → ACR Translation: "new onset seizure unrelated trauma", "seizure no trauma", "first seizure non-traumatic"
        → Primary Search: "new onset seizure unrelated trauma"

        **ACR DATABASE PATTERNS:**
        - ACR organizes by: "[Condition] [Context]" or "[Anatomy] [Symptom]-[Qualifier]"
        - Examples: "Acute Hip Pain-Suspected Fracture", "Lower GI Bleeding-Active", "Chest Pain-Low CAD Risk"
        - Variants specify: "Initial imaging", "Follow-up imaging", "Next step", "Surveillance"
        - Post-surgical terminology: "post-", "surveillance", "follow-up", specific anatomical terms

        **ENHANCED SEARCH STRATEGY:**
        1. **Primary Query**: Most direct translation preserving key original terms
        2. **ACR-Specific Query**: Using exact ACR terminology and patterns  
        3. **Anatomical Query**: Focused on anatomical region + key symptoms
        4. **Fallback Query**: Original query as backup

        **CRITICAL INSTRUCTIONS:**
        1. **PRESERVE ORIGINAL TERMS**: If the query mentions "pituitary adenoma", "surgical excision", "monitoring" - keep these terms!
        2. **DISTINGUISH CLINICAL PHASE**: 
           - "suspected/raising suspicion" → use "suspected" + "initial imaging"
           - "diagnosed with/confirmed" → use "acute" + condition name + treatment context
           - "candidate for anticoagulation" → include "anticoagulation" + "no contraindication"
        3. **DETECT NEGATIONS**: 
           - "no history of trauma" → use "unrelated to trauma" or "non-traumatic"
           - "no prior surgery" → use "without surgery" or "non-surgical"
           - "denies/negative for" → translate to "without" or "unrelated to"
        4. **GENERATE MULTIPLE QUERIES**: Create 3-4 different query formulations targeting different aspects
        5. **USE ACR TERMINOLOGY**: Map common terms to ACR language (follow-up→surveillance, after surgery→post-surgical, diagnosed→acute/confirmed)
        6. **THINK ANATOMICALLY**: Every symptom points to specific body regions - identify them
        7. **CONSIDER CLINICAL CONTEXT**: Determine if this is screening, diagnostic, or treatment/management

        Please provide analysis in this EXACT JSON format:

        {{
            "clinical_analysis": {{
                "primary_symptom": "main presenting symptom",
                "anatomical_region": "specific anatomical region using medical terminology",
                "clinical_urgency": "acute/chronic/routine",
                "patient_demographics": "age/gender if relevant",
                "clinical_context": "emergency/trauma/follow-up/screening",
                "underlying_condition": "most likely medical condition/diagnosis",
                "medical_reasoning": "step-by-step medical analysis"
            }},
            "acr_search_strategy": {{
                "primary_search_query": "most direct query preserving key original terms",
                "acr_specific_query": "query using exact ACR terminology patterns",
                "anatomical_query": "anatomical region + key symptoms query",
                "alternative_queries": ["fallback query using original text", "broader condition category query"],
                "anatomical_keywords": ["precise anatomical terms"],
                "clinical_keywords": ["condition and context terms"],
                "acr_condition_predictions": ["predicted ACR condition 1", "predicted ACR condition 2"]
            }},
            "imaging_context": {{
                "imaging_urgency": "initial/follow-up/surveillance",
                "prior_imaging": "mentioned/not mentioned",
                "clinical_decision_point": "diagnostic/screening/monitoring"
            }},
            "search_confidence": "high/medium/low",
            "rationale": "why this medical reasoning and search strategy should work"
        }}

        **QUERY PRIORITY ORDER:**
        1. Try ACR-specific query first (best chance of exact match)
        2. Try primary query (preserves original terms)
        3. Try anatomical query (broader scope)
        4. Fall back to original query if needed
        
        **REMEMBER:** The goal is to find the exact ACR variant that matches the clinical scenario. Preserve key medical terms from the original query and generate multiple targeted search approaches.
        """
        
        try:
            analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            # Enhanced JSON parsing with multiple cleaning strategies
            response_content = analysis_response.content.strip()
            analysis_data = self._parse_json_response(response_content)
            
            print(f"ACR Agent Query Analysis:")
            print(f"  Primary symptom: {analysis_data['clinical_analysis'].get('primary_symptom', 'Unknown')}")
            print(f"  Anatomical region: {analysis_data['clinical_analysis'].get('anatomical_region', 'Unknown')}")
            print(f"  Underlying condition: {analysis_data['clinical_analysis'].get('underlying_condition', 'Unknown')}")
            print(f"  Medical reasoning: {analysis_data['clinical_analysis'].get('medical_reasoning', 'Unknown')[:100]}...")
            print(f"  Primary search: {analysis_data['acr_search_strategy'].get('primary_search_query', 'Unknown')}")
            print(f"  Search confidence: {analysis_data.get('search_confidence', 'Unknown')}")
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            # Fallback: Create basic search strategy if JSON parsing fails
            print(f"JSON parsing failed: {e}")
            print(f"Raw LLM response: {response_content[:500]}...")  # Show first 500 chars
            return {
                "clinical_analysis": {
                    "primary_symptom": self._extract_primary_symptom(user_query),
                    "anatomical_region": self._extract_anatomy(user_query),
                    "clinical_urgency": "acute" if "acute" in user_query.lower() else "routine",
                    "patient_demographics": "not specified",
                    "clinical_context": "diagnostic",
                    "underlying_condition": "unspecified",
                    "medical_reasoning": "fallback analysis"
                },
                "acr_search_strategy": {
                    "primary_search_query": user_query,
                    "alternative_queries": [user_query],
                    "anatomical_keywords": [self._extract_anatomy(user_query)],
                    "clinical_keywords": [self._extract_primary_symptom(user_query)],
                    "acr_condition_predictions": []
                },
                "imaging_context": {
                    "imaging_urgency": "initial",
                    "prior_imaging": "not mentioned",
                    "clinical_decision_point": "diagnostic"
                },
                "search_confidence": "low",
                "rationale": "Fallback analysis due to JSON parsing failure"
            }
        
        except Exception as e:
            print(f"ACR query analysis failed: {e}")
            return {"error": str(e)}
    
    async def _execute_acr_retrieval(self, user_query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategic ACR retrieval using direct variant search + condition context
        """
        if "error" in analysis:
            # Fallback to basic search
            return {"primary_result": self._direct_variant_search(user_query)}

        # =============================================================
        # TEMP CHANGE: use ONLY the original user query for retrieval
        # =============================================================
        direct_result = await self._direct_variant_search_with_context(user_query)
        return {
            "primary_result": direct_result,
            "alternative_results": [],
            "search_metadata": {
                "queries_tried": [f"Original: {user_query}"],
                "best_similarity": direct_result.get("best_variant", {}).get("variant_similarity", 0.0),
                "total_procedures": len(direct_result.get("usually_appropriate_procedures", [])),
                "search_method": "original_only",
                "best_query": user_query
            }
        }

        # -------- ORIGINAL MULTI-QUERY LOGIC BELOW (currently skipped) --------
    
    async def _evaluate_acr_results(self, user_query: str, analysis: Dict[str, Any], acr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate ACR results and select the best recommendations with smart variant selection
        """
        primary_result = acr_results.get("primary_result", {})
        
        if "error" in primary_result or not primary_result:
            return primary_result
        
        # Get all top variants for evaluation
        all_variants = primary_result.get("all_variants", [])
        
        # Ensure all_variants is a list
        if not isinstance(all_variants, list):
            print(f"Warning: all_variants is {type(all_variants)}, expected list")
            return self._add_evaluation_metadata(primary_result, analysis, acr_results, "invalid_variants")
        
        # Add rank to each variant
        for i, variant in enumerate(all_variants):
            variant["rank"] = i + 1
        
        if len(all_variants) <= 1:
            # Only one variant, no need for LLM evaluation
            return self._add_evaluation_metadata(primary_result, analysis, acr_results, "single_variant")

        # ------------------------------------------------------------------
        #  NEW: Always ask the LLM to pick the best match among top variants
        # ------------------------------------------------------------------
        variants_for_prompt = all_variants[:10]  # limit to 10 variants for prompt brevity
        formatted_variants = self._format_variants_for_llm_selection(variants_for_prompt)

        expected_phase = analysis.get('imaging_context', {}).get('clinical_decision_point', 'unspecified').lower()
        evaluation_prompt = f"""
TASK: Compare the original clinical query directly with these 10 ranked variants and select the BEST match.

ORIGINAL QUERY:
"{user_query}"

RANKED VARIANTS (from ColBERT semantic search):
{formatted_variants}

INSTRUCTIONS:
1. **IGNORE the ColBERT similarity scores** - focus on clinical accuracy
2. **Compare each variant directly** to the original query for:
   - **PRIMARY CLINICAL INDICATION** - What is the main reason for imaging? (e.g., "ICU admission" vs "pneumonia")
   - Patient age appropriateness (adult vs pediatric vs neonatal) - **CRITICAL**
   - **Clinical signs matching** (fever vs no fever, elevated WBC vs normal WBC) - **HIGHEST PRIORITY**
   - Clinical scenario match (first seizure vs known seizure disorder)
   - Trauma context ("no history of trauma" = "unrelated to trauma", NOT "history of trauma")
   - **CLINICAL SEQUENCE/TIMING** (initial vs follow-up vs next study vs surveillance)
   - Anatomical region and symptom specificity

**CLINICAL INDICATION PRIORITY**: Focus on the PRIMARY reason for the imaging study. Secondary findings or specific subtypes should only be selected if they are the main clinical concern.

**PRIORITY RULES**: 
1. Clinical signs (fever/WBC) must match EXACTLY. If a variant contradicts the patient's clinical signs, it should be immediately rejected regardless of how well it matches other criteria.
2. **AVOID OVER-SPECIFICITY**: Prefer general appropriate variants over overly specific variants unless the patient clearly has the specific condition.
3. **AGE MATCHING**: Adult patients (18+ years) should NEVER get pediatric/neonatal variants. Pediatric patients should NEVER get adult variants.

3. **Pay special attention to negations**:
   - "no history of trauma" should match "unrelated to trauma"
   - "no recent trauma or injury" means trauma-related variants are WRONG
   - "having no history of" means absence, not presence

4. **Age considerations**:
   - "24-year-old" is an ADULT, not pediatric (children/neonatal variants inappropriate)
   - Adult variants should be strongly preferred for adult patients

5. **CRITICAL: Clinical sequence recognition**:
   - **"inconclusive/negative/equivocal ultrasound"** = prior imaging done → need "Next imaging study"
   - **"after initial imaging"** = follow-up needed → NOT "Initial imaging"
   - **"further evaluation"** = additional workup → NOT first-line imaging
   - **"HIDA scan ordered"** = advanced imaging → NOT basic ultrasound
   - **"Initial imaging"** variants are WRONG if prior imaging mentioned
   - **"Next imaging study"** variants are CORRECT if prior imaging was done

EXAMPLES OF CLINICAL SEQUENCE ERRORS TO AVOID:
- Query mentions "inconclusive ultrasound" + "HIDA scan ordered" → MUST select "Next imaging study" NOT "Initial imaging"
- Query mentions "negative CT" + "MRI recommended" → MUST select follow-up variant NOT initial variant
- Query mentions "after surgery" → MUST select surveillance/follow-up NOT initial imaging

EXAMPLES OF CLINICAL SIGN MATCHING ERRORS TO AVOID:
- Query mentions "fever and leukocytosis" → MUST select "Fever, elevated WBC" NOT "No fever and no high WBC"
- Query mentions "no fever" → MUST select "No fever" NOT "Fever" variants
- Query mentions "elevated white blood cell count" → MUST select "elevated WBC" NOT "normal WBC" variants

**CRITICAL ERROR PATTERN TO AVOID:**
If the original query states "fever, and leukocytosis" then variants saying "No fever and no high white blood cell (WBC) count" are COMPLETELY WRONG and should be REJECTED immediately, even if they match other criteria like clinical sequence.

EXAMPLES OF OVER-SPECIFICITY ERRORS TO AVOID:
- Patient with "hip prosthesis pain" → select "Symptomatic hip prosthesis" NOT "metal-on-metal trunnionosis" unless explicitly mentioned
- Patient "admitted to ICU" → select "ICU imaging" NOT "pneumonia with effusion" unless pneumonia is the primary concern
- Patient with "diaphragm dysfunction" → select "diaphragm dysfunction" NOT "diaphragmatic hernia" unless hernia is specified
- General condition → prefer general variant over rare/specific subtype

EXAMPLES OF AGE MATCHING ERRORS TO AVOID:
- "72-year-old patient" → NEVER select "Child" or "Younger than 5 years" variants
- "4-year-old patient" → NEVER select "Adult" variants, MUST select pediatric-appropriate variants
- Age-specific variants should only be selected when age matches exactly

SELECT the variant that most accurately represents the clinical scenario described in the original query.

Reply **ONLY** with valid JSON:
{{
  "selected_variant_rank": <1-10>,
  "confidence": "low" | "medium" | "high", 
  "reasoning": "brief explanation of why this variant best matches the original query, specifically addressing clinical sequence if relevant"
}}
"""

        print("\n📝 LLM Variant-selection prompt:\n" + "-"*60)
        print(evaluation_prompt)

        try:
            evaluation_response = await self.llm.ainvoke([HumanMessage(content=evaluation_prompt)])
            response_content = evaluation_response.content.strip()
            print("\n🤖 LLM response:\n" + "-"*60)
            print(response_content)

            decision = self._parse_json_response(response_content)
            selected_rank = int(decision.get("selected_variant_rank", 1))
            confidence = decision.get("confidence", "medium").lower()
            reasoning = decision.get("reasoning", "LLM decision")

            # Accept override if the LLM did not pick rank-1 and is confident
            should_override = (
                1 <= selected_rank <= len(all_variants) and
                selected_rank != 1 and
                confidence == "high"
            )

            if should_override:
                print(f"🔄 LLM overrides to rank {selected_rank} (confidence high).")
                chosen_variant = all_variants[selected_rank-1]
                result_with_override = await self._rebuild_result_with_variant(primary_result, chosen_variant, user_query)
                return self._add_evaluation_metadata(result_with_override, analysis, acr_results, "llm_override", llm_reasoning=reasoning, llm_confidence=confidence, original_rank=selected_rank)
            else:
                print("✅ Keeping ColBERT rank-1 variant but enriching procedures from Neo4j.")

                # Enrich procedures using Neo4j to ensure full list, not just ColBERT stub
                try:
                    rank1_variant = all_variants[0]
                    enriched_result = await self._rebuild_result_with_variant(primary_result, rank1_variant, user_query)
                except Exception as enrich_err:
                    print(f"⚠️  Procedure enrichment failed: {enrich_err}")
                    enriched_result = primary_result

                return self._add_evaluation_metadata(enriched_result, analysis, acr_results, "llm_confirmed", llm_reasoning=reasoning, llm_confidence=confidence, original_rank=1)

        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return self._add_evaluation_metadata(primary_result, analysis, acr_results, "llm_error")
    
    def _has_potential_negation_issue(self, query: str, variants: List[Dict[str, Any]]) -> bool:
        """
        Enhanced negation detection to identify potential semantic mismatches.
        Triggers LLM validation when negation context suggests possible errors.
        """
        # Safety check
        if not isinstance(variants, list):
            print(f"Warning: variants is {type(variants)}, expected list")
            return False
        
        query_lower = query.lower()
        
        # Comprehensive negation patterns
        negation_patterns = [
            "no ", "not ", "without ", "negative ", "absent ", "never ", "none ",
            "no history", "having no", "unrelated to", "free of", "lacking",
            "denies", "negative for", "absence of", "no significant", "no prior"
        ]
        
        positive_patterns = [
            "with ", "positive ", "history of ", "known ", "previous ", "with history",
            "presence of", "positive for", "associated with", "related to", "due to"
        ]
        
        # Medical context patterns for better detection
        medical_contexts = [
            "trauma", "injury", "history", "complications", "symptoms", "findings",
            "disease", "condition", "abnormality", "pathology", "lesion"
        ]
        
        # Check query negation context
        query_negations = [pattern for pattern in negation_patterns if pattern in query_lower]
        query_positives = [pattern for pattern in positive_patterns if pattern in query_lower]
        
        if not (query_negations or query_positives):
            return False  # No clear negation context
        
        # Analyze variants for conflicting patterns
        for variant in variants:
            variant_text = variant.get("variant_id", "").lower()
            
            # Check for direct contradictions
            variant_negations = [pattern for pattern in negation_patterns if pattern in variant_text]
            variant_positives = [pattern for pattern in positive_patterns if pattern in variant_text]
            
            # Specific medical context checks
            for context in medical_contexts:
                if context in query_lower and context in variant_text:
                    # Check for negation mismatch in same medical context
                    query_context_negative = any(f"{neg}{context}" in query_lower or f"{neg} {context}" in query_lower 
                                               for neg in ["no", "without", "negative", "absent"])
                    variant_context_positive = any([
                        f"history of {context}" in variant_text,
                        f"with {context}" in variant_text,
                        f"positive {context}" in variant_text
                    ])
                    
                    if query_context_negative and variant_context_positive:
                        return True
                    
                    # Specific trauma pattern (most common issue)
                    if context == "trauma":
                        query_no_trauma = any(pattern in query_lower for pattern in 
                                            ["no history of trauma", "no trauma", "no recent trauma", "having no history"])
                        variant_has_trauma = "history of" in variant_text and "trauma" in variant_text
                        
                        if query_no_trauma and variant_has_trauma:
                            return True
            
            # General semantic contradiction check
            if query_negations and variant_positives:
                return True
            if query_positives and variant_negations:
                return True
        
        return False
    
    def _format_variants_for_llm_selection(self, variants: List[Dict[str, Any]]) -> str:
        """Format variants for LLM selection evaluation with inferred clinical phase"""
        def _infer_phase(text: str) -> str:
            text_l = text.lower()
            if any(k in text_l for k in ["treatment", "therapy", "management"]):
                return "treatment"
            if any(k in text_l for k in ["initial", "diagnosis", "diagnostic", "evaluation"]):
                return "diagnosis"
            if "follow" in text_l or "surveillance" in text_l:
                return "follow-up"
            return "unspecified"

        formatted = []
        for variant in variants:
            rank = variant.get("rank", "Unknown")
            content = variant.get("content", variant.get("variant_id", "Unknown"))
            similarity = variant.get("similarity", 0.0)
            condition = variant.get("condition_id", "Unknown")
            phase = _infer_phase(content)

            formatted.append(f"**Rank {rank}:** {content}")
            formatted.append(f"  - Similarity: {similarity:.4f}")
            formatted.append(f"  - Condition: {condition}")
            formatted.append(f"  - Inferred Phase: {phase}")
            formatted.append("")

        return "\n".join(formatted)
    
    async def _rebuild_result_with_variant(self, original_result: Dict[str, Any], selected_variant: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Rebuild ACR result with LLM-selected variant"""
        variant_id = selected_variant["variant_id"]
        
        try:
            # Resolve variant ID if placeholder (colbert_variant_X)
            if variant_id.startswith("colbert_variant"):
                resolved_id = self.neo4j_recommender.resolve_variant_id_by_text(selected_variant.get("content", ""))
                if resolved_id:
                    print(f"🔗 Resolved ColBERT placeholder to Neo4j variant id: {resolved_id}")
                    variant_id = resolved_id

            # Get procedures for the (resolved) variant
            procedures = self.neo4j_recommender.get_procedures_for_variant(variant_id, "USUALLY_APPROPRIATE")
            
            # Ensure procedures is a list
            if not isinstance(procedures, list):
                print(f"Warning: get_procedures_for_variant returned {type(procedures)}, expected list")
                procedures = []
            
            # Rebuild result structure
            new_result = original_result.copy()
            new_result["best_variant"] = {
                "variant_id": variant_id,
                "variant_similarity": selected_variant["similarity"],
                "variant_description": variant_id  # Could be enhanced with actual description
            }
            new_result["usually_appropriate_procedures"] = procedures
            new_result["context_for_output"] = {
                "clinical_category": selected_variant.get("condition_id", "Unknown"),
                "specific_scenario": variant_id,
                "search_confidence": "llm_override"
            }
            
            print(f"🔄 Rebuilt result with variant: {variant_id}")
            print(f"Found {len(procedures)} procedures for selected variant")
            
            return new_result
            
        except Exception as e:
            print(f"Error in _rebuild_result_with_variant: {e}")
            print(f"   Variant ID: {variant_id}")
            print(f"   Procedures type: {type(procedures) if 'procedures' in locals() else 'undefined'}")
            
            # Return original result as fallback
            return original_result
    
    def _add_evaluation_metadata(self, result: Dict[str, Any], analysis: Dict[str, Any], 
                                acr_results: Dict[str, Any], evaluation_type: str,
                                llm_reasoning: str = "", llm_confidence: str = "",
                                negation_detected: bool = False, original_rank: int = 1) -> Dict[str, Any]:
        """Add evaluation metadata to result"""
        if "evaluation" not in result:
            result["evaluation"] = {}
        
        result["evaluation"].update({
            "search_strategy": analysis.get("acr_search_strategy", {}),
            "queries_tried": acr_results.get("search_metadata", {}).get("queries_tried", []),
            "best_similarity": acr_results.get("search_metadata", {}).get("best_similarity", 0.0),
            "evaluation_type": evaluation_type,
            "llm_reasoning": llm_reasoning,
            "llm_confidence": llm_confidence,
            "negation_issue_detected": negation_detected,
            "selected_rank": original_rank
        })
        
        if evaluation_type == "llm_override":
            print(f"🎉 LLM Override Applied: {llm_reasoning}")
        
        return result
    
    def _format_acr_result_for_evaluation(self, result: Dict[str, Any]) -> str:
        """Format ACR result for LLM evaluation"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "best_variant" in result:
            variant_id = result["best_variant"].get("variant_id", "Unknown")
            similarity = result["best_variant"].get("variant_similarity", 0.0)
            procedures = result.get("usually_appropriate_procedures", [])
            
            return f"Variant: {variant_id} (similarity: {similarity:.3f}), Procedures: {len(procedures)}"
        
        return "No valid result"
    
    def _format_alternative_results(self, alternatives: List[Dict[str, Any]]) -> str:
        """Format alternative results for LLM evaluation"""
        if not alternatives:
            return "None"
        
        formatted = []
        for i, alt in enumerate(alternatives):
            query = alt.get("query", "Unknown")
            result = alt.get("result", {})
            if "best_variant" in result:
                similarity = result["best_variant"].get("variant_similarity", 0.0)
                formatted.append(f"{i+1}. Query: '{query}' (similarity: {similarity:.3f})")
        
        return "\n".join(formatted) if formatted else "None valid"
    
    def _extract_primary_symptom(self, query: str) -> str:
        """Extract primary symptom from query (fallback method)"""
        query_lower = query.lower()
        symptoms = ["pain", "bleeding", "trauma", "fever", "headache", "chest pain", "abdominal pain"]
        
        for symptom in symptoms:
            if symptom in query_lower:
                return symptom
        
        return "unspecified symptom"
    
    def _extract_anatomy(self, query: str) -> str:
        """Extract anatomical region from query (fallback method)"""
        query_lower = query.lower()
        anatomy = ["head", "chest", "abdomen", "pelvis", "hip", "knee", "ankle", "hand", "wrist", "spine"]
        
        for region in anatomy:
            if region in query_lower:
                return region
        
        return "unspecified region"
    
    async def _direct_variant_search_with_context(self, query_text: str) -> Dict[str, Any]:
        """
        Hybrid search: ColBERT for variant finding + Neo4j for procedure fetching
        """
        try:
            print(f"Hybrid Search: '{query_text}'")
            
            # Step 1: Use ColBERT to find best matching ACR variant
            colbert_results = self.colbert_retriever.search_acr_variants(query_text, k=10)
            
            if not colbert_results:
                return {"error": "No ACR variants found"}
            
            # Format all results with proper structure
            formatted_results = []
            for i, result in enumerate(colbert_results):
                formatted_result = {
                    "rank": i + 1,
                    "variant_id": result["content"],
                    "content": result["content"],
                    "similarity": result["score"],
                    "condition_id": "colbert_matched_condition",
                    "relevance_score": result["score"]
                }
                formatted_results.append(formatted_result)
            
            best_result = formatted_results[0]
            variant_content = best_result["content"]
            
            print(f"ColBERT found variant: {variant_content}")
            print(f"Similarity: {best_result['similarity']:.4f}")
            
            # Step 2: Extract actual variant ID from the content and use Neo4j to get procedures
            # The variant content should contain the actual variant ID
            # Need to find this variant in Neo4j and get its procedures
            procedures = self.neo4j_recommender.get_procedures_for_variant(variant_content, "USUALLY_APPROPRIATE")
            
            # Build proper result structure
            result = {
                "query": query_text,
                "retrieval_method": "hybrid_colbert_neo4j",
                "best_variant": {
                    "variant_id": variant_content,
                    "content": variant_content,
                    "variant_similarity": best_result["similarity"],
                    "relevance_score": best_result["similarity"]
                },
                "usually_appropriate_procedures": procedures if isinstance(procedures, list) else [],
                "all_variants": formatted_results,  # Use formatted results here
                "total_results": len(procedures) if isinstance(procedures, list) else 0
            }
            
            print(f"Neo4j returned {len(result['usually_appropriate_procedures'])} procedures")
            return result
                
        except Exception as e:
            print(f"Hybrid variant search failed: {e}")
            return {"error": f"Hybrid variant search failed: {str(e)}"}
    
    def _direct_variant_search(self, query_text: str) -> Dict[str, Any]:
        """
        Synchronous fallback for direct variant search
        """
        try:
            # Simple fallback using the original recommender
            return self.neo4j_recommender.recommend_procedures(query_text)
        except Exception as e:
            return {"error": f"Fallback search failed: {str(e)}"}
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """
        Robust JSON parsing with multiple cleaning strategies
        """
        import re
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code blocks
        cleaned = response_content
        
        # Remove ```json ... ``` blocks
        if '```json' in cleaned:
            cleaned = re.sub(r'```json\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        
        # Remove generic ``` blocks
        elif '```' in cleaned:
            cleaned = re.sub(r'```\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Find JSON within text
        # Look for { ... } pattern
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Remove common prefixes/suffixes
        lines = cleaned.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if line.strip().startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
                if line.strip().endswith('}') and line.count('}') >= line.count('{'):
                    break
        
        if json_lines:
            try:
                return json.loads('\n'.join(json_lines))
            except json.JSONDecodeError:
                pass
        
        # Strategy 5: Last resort - extract between first { and last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                json_content = cleaned[first_brace:last_brace + 1]
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # If all strategies fail, raise the original error
        print(f"All JSON parsing strategies failed")
        print(f"Raw response (first 200 chars): {response_content[:200]}")
        print(f"Raw response (last 200 chars): {response_content[-200:]}")
        raise json.JSONDecodeError("Could not parse JSON from LLM response", response_content, 0)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        error_msg = AIMessage(
            content=f"Enhanced ACR Agent error: {error_message}",
            additional_kwargs={"error": True, "source": "enhanced_acr_agent"}
        )
        
        return {
            "messages": [error_msg],
            "acr_recommendations": {"error": error_message},
            "next_step": "error_handling"
        }