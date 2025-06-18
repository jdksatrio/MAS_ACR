from typing import Dict, Any, TypedDict, Annotated, Optional
import sys
import os
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# Add the retrieve_acr directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieve_acr'))

from medical_procedure_recommender_vectorized import MedicalProcedureRecommenderVectorized

# Import AgentState from main module
try:
    from .graphrag_node import AgentState
except ImportError:
    # Fallback for direct execution
    from graphrag_node import AgentState

class ACRRetrievalNode:
    """
    LangGraph node for ACR (American College of Radiology) procedure recommendations.
    Uses VECTOR INDEXES for fast similarity search to find appropriate medical procedures.
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j", 
        neo4j_password: str = None,
        embedding_provider: str = "pubmedbert",
        embedding_model: str = "NeuML/pubmedbert-base-embeddings",
        openai_api_key: str = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.neo4j_password = neo4j_password
        self.recommender_params = {
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "openai_api_key": openai_api_key,
            "ollama_base_url": ollama_base_url
        }
        self.recommender = None
        self.initialized = False
    
    async def initialize(self, neo4j_password: str = None):
        """Initialize the ACR recommender"""
        if self.initialized and self.recommender:
            return
            
        password = neo4j_password or self.neo4j_password
        if not password:
            raise ValueError("Neo4j password is required for ACR retrieval")
        
        try:
            self.recommender = MedicalProcedureRecommenderVectorized(
                neo4j_password=password,
                **self.recommender_params
            )
            self.initialized = True
            print("VECTORIZED ACR Retrieval node initialized (using vector indexes)")
            
        except Exception as e:
            print(f"ACR Retrieval initialization failed: {e}")
            raise
    
    def close(self):
        """Close the Neo4j connection"""
        if self.recommender:
            self.recommender.close()
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node function - processes medical queries and returns ACR procedure recommendations.
        """
        if not self.initialized:
            # Try to get password from state or environment
            neo4j_password = state.get("neo4j_password") or os.getenv("NEO4J_PASSWORD")
            await self.initialize(neo4j_password)
        
        user_query = state.get("user_query", "")
        if not user_query:
            # Extract query from messages if not in state
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    user_query = message.content
                    break
        
        if not user_query:
            error_message = AIMessage(
                content="No medical query found for ACR procedure recommendation",
                additional_kwargs={"error": True, "source": "acr_retrieval_node"}
            )
            return {
                "messages": [error_message],
                "acr_recommendations": {},
                "next_step": "error_handling"
            }
        
        try:
            # Get ACR procedure recommendations
            recommendations = self.recommender.recommend_procedures(user_query)
            
            # Format the recommendations for the response
            if "error" in recommendations:
                response_content = f"ACR Retrieval Error: {recommendations['error']}"
                response_message = AIMessage(
                    content=response_content,
                    additional_kwargs={
                        "error": True,
                        "source": "acr_retrieval_node",
                        "acr_recommendations": recommendations
                    }
                )
                next_step = "error_handling"
            else:
                response_content = self._format_recommendations_summary(recommendations)
                response_message = AIMessage(
                    content=response_content,
                    additional_kwargs={
                        "source": "acr_retrieval_node",
                        "acr_recommendations": recommendations
                    }
                )
                next_step = "acr_analysis"
            
            return {
                "messages": [response_message],
                "acr_recommendations": recommendations,
                "next_step": next_step
            }
            
        except Exception as e:
            error_message = AIMessage(
                content=f"ACR procedure recommendation failed: {str(e)}",
                additional_kwargs={"error": True, "source": "acr_retrieval_node"}
            )
            
            return {
                "messages": [error_message],
                "acr_recommendations": {"error": str(e)},
                "next_step": "error_handling"
            }
    
    def _format_recommendations_summary(self, recommendations: Dict) -> str:
        """Format ACR recommendations into a readable summary"""
        if "error" in recommendations:
            return f"Error: {recommendations['error']}"
        
        summary = f"ACR Procedure Recommendations for: {recommendations['query']}\n\n"
        
        if "best_variant" in recommendations:
            # New format with best variant
            summary += f"**Top Matching Condition:**\n"
            summary += f"• {recommendations['top_condition']['condition_id']}\n"
            summary += f"• Similarity: {recommendations['top_condition']['condition_similarity']:.3f}\n\n"
            
            summary += f"**Best Matching Variant:**\n"
            summary += f"• {recommendations['best_variant']['variant_id']}\n"
            summary += f"• Similarity: {recommendations['best_variant']['variant_similarity']:.3f}\n\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                summary += f"**Usually Appropriate Procedures ({len(procedures)}):**\n"
                for i, procedure in enumerate(procedures[:5], 1):  # Limit to top 5
                    summary += f"{i}. {procedure['procedure_id']}\n"
                    if procedure.get('dosage'):
                        summary += f"   • Radiation Dosage: {procedure['dosage']}\n"
                
                if len(procedures) > 5:
                    summary += f"... and {len(procedures) - 5} more procedures\n"
            else:
                summary += "**No usually appropriate procedures found for this variant.**\n"
        
        else:
            # Old format with multiple conditions
            summary += f"**Similar Conditions Found:**\n"
            for i, condition in enumerate(recommendations['similar_conditions'][:3], 1):
                summary += f"{i}. {condition['condition_id']} (similarity: {condition['similarity_score']:.3f})\n"
            
            # Count total procedures
            total_appropriate = len(recommendations['aggregated_procedures']['USUALLY_APPROPRIATE'])
            total_maybe = len(recommendations['aggregated_procedures']['MAYBE_APPROPRIATE'])
            
            summary += f"\n**Procedure Summary:**\n"
            summary += f"• Usually Appropriate: {total_appropriate} procedures\n"
            summary += f"• Maybe Appropriate: {total_maybe} procedures\n"
        
        return summary

class ACRAnalysisNode:
    """
    Node that analyzes ACR recommendations and provides clinical insights.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Analyze ACR recommendations and provide clinical context"""
        
        user_query = state.get("user_query", "")
        acr_recommendations = state.get("acr_recommendations", {})
        graphrag_context = state.get("graphrag_context", "")
        
        if not acr_recommendations or "error" in acr_recommendations:
            return {
                "messages": [AIMessage(content="No valid ACR recommendations available for analysis")],
                "acr_analysis": "No ACR data available",
                "next_step": "final_response"
            }
        
        # Create analysis prompt combining GraphRAG context and ACR recommendations
        analysis_prompt = self._create_analysis_prompt(user_query, acr_recommendations, graphrag_context)
        
        try:
            # Get analysis from LLM
            analysis_result = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            return {
                "messages": [analysis_result],
                "acr_analysis": analysis_result.content,
                "next_step": "final_response"
            }
            
        except Exception as e:
            error_message = AIMessage(content=f"ACR analysis failed: {str(e)}")
            return {
                "messages": [error_message],
                "acr_analysis": f"Analysis error: {str(e)}",
                "next_step": "error_handling"
            }
    
    def _create_analysis_prompt(self, user_query: str, acr_recommendations: Dict, graphrag_context: str) -> str:
        """Create a comprehensive analysis prompt with enhanced medical context"""
        
        prompt = f"""
        You are an expert medical AI assistant providing comprehensive clinical analysis by integrating:
        1. Patient clinical presentation
        2. ACR Appropriateness Criteria recommendations  
        3. Medical knowledge graph insights
        4. Evidence-based clinical rationales
        
        **Patient Clinical Presentation:** {user_query}
        
        **ACR Procedure Recommendations:**
        {self._format_acr_for_prompt(acr_recommendations)}
        """
        
        if graphrag_context:
            prompt += f"""
        
        **Medical Knowledge Graph Context:**
        {graphrag_context}
        
        **Enhanced Context Integration Instructions:**
        Use the medical knowledge graph context to provide deeper clinical insights including:
        - Disease pathophysiology and progression patterns
        - Risk factors and predisposing conditions  
        - Differential diagnosis considerations
        - Complications and associated conditions
        - Evidence from medical literature and clinical studies
        """
        
        prompt += """
        
        **Please provide a comprehensive medical analysis using this ENHANCED format:**
        
        ## Clinical Assessment and Context
        Based on the condition [extract key condition from query], provide a detailed medical assessment incorporating:
        - Pathophysiology and clinical significance of the presenting condition
        - Risk factors and predisposing conditions from the knowledge graph
        - Potential complications if left undiagnosed or untreated
        - Differential diagnosis considerations
        - Clinical urgency and severity indicators
        
        [Continue with 2-3 paragraphs of comprehensive clinical context using the GraphRAG knowledge]
        
        ## Imaging Recommendations
        The appropriate imaging options for the patient include:
        
        **Imaging:** [First imaging modality] : Usually Appropriate
        **Rationale:** [Comprehensive clinical justification for this specific modality, including why it's the most appropriate choice, diagnostic yield, and clinical utility]
        **References:** 
        1. ACR Appropriateness Criteria®: [specific ACR guideline]
        2. [Additional clinical evidence or guideline if relevant]
        
        ------
        
        **Imaging:** [Second imaging modality] : Usually Appropriate  
        **Rationale:** [Clinical justification explaining when this alternative would be preferred, its specific advantages, limitations, and clinical scenarios where it's optimal]
        **References:**
        1. ACR Appropriateness Criteria®: [specific ACR guideline]
        2. [Additional supporting evidence]
        
        ------
        
        **Imaging:** [Third imaging modality] : Usually Appropriate
        **Rationale:** [Clinical justification for this option, including comparative advantages, specific clinical indications, and when it might be considered]
        **References:**
        1. ACR Appropriateness Criteria®: [specific ACR guideline]
        2. [Supporting clinical evidence]
        
        ------
        
        ## Clinical Workflow and Priority
        - Recommended imaging sequence and timing
        - Clinical urgency and priority level
        - Patient preparation requirements
        - Expected diagnostic timeline and follow-up
        
        **CRITICAL REQUIREMENTS - MUST FOLLOW EXACTLY:**
        1. ALWAYS show ALL 3 imaging procedures if 3 are available (or all available if fewer than 3)
        2. Use the EXACT format above with "Imaging:", "Rationale:", "References:" for EACH procedure
        3. Include ------ separators between each procedure
        4. Each procedure must have its appropriateness category (Usually Appropriate, May Be Appropriate, etc.)
        5. Do NOT combine procedures - show each one separately with its own rationale and references
        6. If only 1-2 procedures are available, show all of them using the same format
        7. Each rationale should be distinct and procedure-specific, explaining when/why that particular modality is preferred
        """
        
        return prompt
    
    def _format_acr_for_prompt(self, recommendations: Dict) -> str:
        """Format ACR recommendations for the LLM prompt"""
        if "error" in recommendations:
            return f"Error: {recommendations['error']}"
        
        formatted = f"Query: {recommendations['query']}\n\n"
        
        if "best_variant" in recommendations:
            # New format
            formatted += f"Top Condition: {recommendations['top_condition']['condition_id']}\n"
            formatted += f"Condition Similarity: {recommendations['top_condition']['condition_similarity']:.3f}\n\n"
            
            formatted += f"Best Variant: {recommendations['best_variant']['variant_id']}\n"
            formatted += f"Variant Similarity: {recommendations['best_variant']['variant_similarity']:.3f}\n\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                formatted += "Available Procedures with Appropriateness Categories:\n"
                for i, procedure in enumerate(procedures[:3], 1):  # Limit to top 3 for multiple display
                    formatted += f"{i}. {procedure['procedure_id']} : Usually Appropriate\n"
                    if procedure.get('dosage'):
                        formatted += f"   Radiation Dosage: {procedure['dosage']}\n"
                
                # Check if there are May Be Appropriate procedures in enriched data
                enriched = recommendations.get('enriched_rationales', {})
                if enriched and 'procedures' in enriched:
                    maybe_appropriate = []
                    for proc in enriched['procedures']:
                        if proc.get('appropriateness') == 'MAY_BE_APPROPRIATE':
                            maybe_appropriate.append(proc)
                    
                    if maybe_appropriate:
                        formatted += "\nMay Be Appropriate Procedures:\n"
                        for i, procedure in enumerate(maybe_appropriate[:2], 1):  # Add up to 2 more
                            formatted += f"{len(procedures) + i}. {procedure['procedure_name']} : May Be Appropriate\n"
                            if procedure.get('dosage'):
                                formatted += f"   Radiation Dosage: {procedure['dosage']}\n"
            else:
                formatted += "No usually appropriate procedures found.\n"
        
        else:
            # Old format
            formatted += "Similar Conditions:\n"
            for condition in recommendations['similar_conditions']:
                formatted += f"- {condition['condition_id']} (similarity: {condition['similarity_score']:.3f})\n"
            
            formatted += "\nAvailable Procedures by Appropriateness Category:\n"
            
            # Usually Appropriate procedures
            usually_appropriate = recommendations['aggregated_procedures']['USUALLY_APPROPRIATE']
            if usually_appropriate:
                formatted += f"\nUsually Appropriate:\n"
                for i, procedure in enumerate(usually_appropriate[:3], 1):  # Limit to top 3
                    formatted += f"{i}. {procedure['procedure_id']} : Usually Appropriate"
                    if procedure.get('dosage'):
                        formatted += f" (Dosage: {procedure['dosage']})"
                    formatted += f" [from {procedure['source_condition']}]\n"
            
            # May Be Appropriate procedures  
            maybe_appropriate = recommendations['aggregated_procedures']['MAYBE_APPROPRIATE']
            if maybe_appropriate and len(usually_appropriate) < 3:
                remaining_slots = 3 - len(usually_appropriate)
                formatted += f"\nMay Be Appropriate:\n"
                for i, procedure in enumerate(maybe_appropriate[:remaining_slots], len(usually_appropriate) + 1):
                    formatted += f"{i}. {procedure['procedure_id']} : May Be Appropriate"
                    if procedure.get('dosage'):
                        formatted += f" (Dosage: {procedure['dosage']})"
                    formatted += f" [from {procedure['source_condition']}]\n"
        
        return formatted 