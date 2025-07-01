from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# nodes
from .graphrag_node import AgentState, MedicalGraphRAGNode, MedicalAnalysisNode
from .acr_retrieval_node import ACRRetrievalNode, ACRAnalysisNode


class EnhancedMedicalAnalysisNode:
    """
    Enhanced analysis node that combines GraphRAG context and ACR recommendations
    for comprehensive medical insights.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Provide comprehensive analysis combining all available medical data"""
        
        user_query = state.get("user_query", "")
        graphrag_context = state.get("graphrag_context", "")
        acr_recommendations = state.get("acr_recommendations", {})
        
        # Create comprehensive analysis prompt
        analysis_prompt = self._create_comprehensive_prompt(
            user_query, graphrag_context, acr_recommendations
        )
        
        try:
            # Get comprehensive analysis from LLM
            analysis_result = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            return {
                "messages": [analysis_result],
                "analysis_result": analysis_result.content,
                "final_answer": analysis_result.content,
                "next_step": "complete"
            }
            
        except Exception as e:
            error_message = AIMessage(content=f"Enhanced analysis failed: {str(e)}")
            return {
                "messages": [error_message],
                "analysis_result": f"Analysis error: {str(e)}",
                "final_answer": f"Analysis error: {str(e)}",
                "next_step": "error_handling"
            }
    
    def _create_comprehensive_prompt(self, user_query: str, graphrag_context: str, acr_recommendations: Dict) -> str:
        """Create a comprehensive analysis prompt combining all data sources"""
        
        prompt = f"""
        You are an advanced medical AI assistant providing comprehensive clinical analysis.
        
        **Patient Query:** {user_query}
        """
        
        # Add GraphRAG context if available
        if graphrag_context:
            prompt += f"""
        
        **Medical Knowledge Graph Context:**
        {graphrag_context}
        """
        
        # Add ACR recommendations if available
        if acr_recommendations and "error" not in acr_recommendations:
            prompt += f"""
        
        **ACR Procedure Recommendations:**
        {self._format_acr_summary(acr_recommendations)}
        """
        
        prompt += """
        
        **Please provide a comprehensive medical analysis including:**
        
        ## 1. Clinical Assessment
        - Primary diagnosis considerations based on the query
        - Key symptoms, risk factors, and clinical indicators
        - Differential diagnosis possibilities
        
        ## 2. Knowledge Graph Insights
        - Relevant medical relationships and connections found
        - Supporting evidence from medical literature/guidelines
        - Disease associations and comorbidities
        
        ## 3. Diagnostic Procedure Analysis
        - Appropriateness of recommended procedures for this case
        - Clinical rationale for each suggested procedure
        - Expected diagnostic yield and clinical utility
        
        ## 4. Risk-Benefit Assessment
        - Radiation exposure considerations (if applicable)
        - Patient safety factors
        - Cost-effectiveness of diagnostic approach
        
        ## 5. Clinical Workflow Recommendations
        - Optimal sequence of diagnostic procedures
        - Priority levels (urgent vs. routine)
        - Prerequisites or contraindications to consider
        
        ## 6. Treatment Implications
        - How diagnostic results might influence treatment decisions
        - Potential therapeutic pathways based on findings
        - Follow-up recommendations
        
        ## 7. Patient Communication
        - Key points to discuss with the patient
        - Expected timeline and process explanation
        - Preparation instructions if needed
        
        Provide evidence-based recommendations using appropriate medical terminology while maintaining clarity for clinical decision-making.
        """
        
        return prompt
    
    def _format_acr_summary(self, recommendations: Dict) -> str:
        """Format ACR recommendations for the comprehensive prompt"""
        if "error" in recommendations:
            return f"ACR Retrieval Error: {recommendations['error']}"
        
        summary = f"Query: {recommendations['query']}\n"
        
        if "best_variant" in recommendations:
            # New format
            summary += f"\nMatched Condition: {recommendations['top_condition']['condition_id']}\n"
            summary += f"Condition Similarity: {recommendations['top_condition']['condition_similarity']:.3f}\n"
            summary += f"Best Variant: {recommendations['best_variant']['variant_id']}\n"
            summary += f"Variant Similarity: {recommendations['best_variant']['variant_similarity']:.3f}\n"
            
            procedures = recommendations['usually_appropriate_procedures']
            if procedures:
                summary += "\nUsually Appropriate Procedures:\n"
                for procedure in procedures:
                    summary += f"• {procedure['procedure_id']}\n"
                    if procedure.get('dosage'):
                        summary += f"  Radiation Dosage: {procedure['dosage']}\n"
        else:
            # Old format
            summary += "\nSimilar Conditions:\n"
            for condition in recommendations['similar_conditions'][:3]:
                summary += f"• {condition['condition_id']} (similarity: {condition['similarity_score']:.3f})\n"
            
            # Show procedure counts
            for level in ['USUALLY_APPROPRIATE', 'MAYBE_APPROPRIATE']:
                procedures = recommendations['aggregated_procedures'][level]
                if procedures:
                    summary += f"\n{level.replace('_', ' ')} ({len(procedures)} procedures):\n"
                    for procedure in procedures[:3]:  # Show top 3
                        summary += f"• {procedure['procedure_id']}\n"
        
        return summary


def create_medical_workflow(llm, neo4j_password: str = None) -> StateGraph:
    """
    Create a medical workflow that combines GraphRAG and ACR retrieval.
    
    Args:
        llm: Your LangChain LLM instance
        neo4j_password: Neo4j password for ACR retrieval
        
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    
    # Initialize nodes
    graphrag_node = MedicalGraphRAGNode()
    acr_node = ACRRetrievalNode(
        retrieval_method="colbert",
        neo4j_password=neo4j_password,
        debug=True
    )
    analysis_node = EnhancedMedicalAnalysisNode(llm)
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("graphrag_retrieval", graphrag_node)
    workflow.add_node("acr_retrieval", acr_node)
    workflow.add_node("medical_analysis", analysis_node)
    workflow.add_node("final_response", lambda state: {
        "final_answer": state.get("analysis_result", "No analysis available"),
        "next_step": "complete"
    })
    
    # Define workflow edges - Sequential processing
    workflow.set_entry_point("graphrag_retrieval")
    workflow.add_edge("graphrag_retrieval", "acr_retrieval")
    workflow.add_edge("acr_retrieval", "medical_analysis")
    workflow.add_edge("medical_analysis", "final_response")
    workflow.add_edge("final_response", END)
    
    # Compile the workflow
    return workflow.compile()


async def run_medical_workflow(
    user_query: str, 
    llm, 
    neo4j_password: str = None
) -> Dict[str, Any]:
    """
    Run the medical workflow.
    
    Args:
        user_query: Medical question or case description
        llm: LangChain LLM instance
        neo4j_password: Neo4j password for ACR retrieval
        
    Returns:
        Complete workflow result with medical analysis
    """
    
    # Create workflow
    workflow = create_medical_workflow(llm, neo4j_password)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "neo4j_password": neo4j_password
    }
    
    try:
        # Run the workflow
        result = await workflow.ainvoke(initial_state)
        
        return {
            "success": True,
            "query": user_query,
            "final_answer": result.get("final_answer", ""),
            "graphrag_context": result.get("graphrag_context", ""),
            "acr_recommendations": result.get("acr_recommendations", {}),
            "analysis_result": result.get("analysis_result", ""),
            "messages": result.get("messages", [])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": user_query
        } 