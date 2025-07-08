from typing import Dict, Any, TypedDict, Annotated
import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_graphrag'))

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import compute_mdhash_id

# Define the state schema for your multi-agent system
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str
    graphrag_context: str
    graphrag_entities: list[dict]
    acr_recommendations: dict
    enriched_rationales: dict
    acr_analysis: str
    analysis_result: str
    final_answer: str
    next_step: str
    neo4j_password: str

class MedicalGraphRAGNode:
    """
    LangGraph node for Medical GraphRAG integration.
    Directly queries Neo4j database containing MIMIC data instead of empty GraphRAG cache.
    """
    
    def __init__(
        self,
        working_dir: str = "./nano_graphrag_cache_langgraph",
        enable_llm_cache: bool = True,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = None
    ):
        self.working_dir = working_dir
        self.enable_llm_cache = enable_llm_cache
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graph_rag = None
        self.neo4j_driver = None
        self.initialized = False
        
        # Medical entity types
        self.medical_entity_types = [
            "DISEASE", "SYMPTOM", "MEDICATION", "TREATMENT", 
            "DIAGNOSTIC_TEST", "ANATOMY", "PERSON", "ORGANIZATION",
            "PROCEDURE", "CONDITION"
        ]
    
    async def initialize(self):
        """Initialize direct Neo4j connection to query MIMIC data"""
        if self.initialized:
            return
            
        try:
            # Import Neo4j driver
            from neo4j import GraphDatabase
            
            # Get password from environment or parameter
            password = self.neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")
            
            # Initialize direct Neo4j connection for MIMIC data
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, password)
            )
            
            self.initialized = True
            print(f"Medical Neo4j connection initialized at {self.neo4j_uri}")
            
        except Exception as e:
            print(f"Neo4j initialization failed: {e}")
            raise
    
    def _query_mimic_context(self, user_query: str) -> str:
        """Query MIMIC data from Neo4j using semantic similarity for clinical decision support context"""
        if not self.neo4j_driver:
            return "Neo4j connection not initialized"
        
        try:
            return self._semantic_search_entities(user_query)
        except Exception as e:
            return f"Clinical context retrieval failed: {str(e)}"
    
    def _semantic_search_entities(self, user_query: str) -> str:
        """Use embedding similarity to find relevant medical entities"""
        try:
            with self.neo4j_driver.session() as session:
                # Get query embedding using PubMedBERT model
                query_embedding = self._get_query_embedding(user_query)
                
                if query_embedding is None:
                    return "**Clinical Context:** Could not generate query embedding"
                
                # Use vector similarity search on GraphRAG entities
                similarity_query = """
                CALL db.index.vector.queryNodes('graphrag_embeddings_index', $k, $query_embedding)
                YIELD node, score
                WHERE score > 0.3
                
                // Get related entities for top matches
                OPTIONAL MATCH (node)-[r]-(related:chunk_entity_relation_entity)
                WHERE related.entity_type IN ['DISEASE', 'SYMPTOM', 'DIAGNOSTIC_TEST', 'TREATMENT', 'ANATOMY']
                  AND related.id IS NOT NULL
                
                RETURN node.id as primary_entity,
                       node.entity_type as primary_type,
                       node.description as primary_description,
                       score as similarity_score,
                       collect(DISTINCT {
                           entity: related.id,
                           type: related.entity_type,
                           description: related.description
                       })[0..3] as related_entities
                ORDER BY similarity_score DESC
                LIMIT 8
                """
                
                result = session.run(similarity_query, {
                    "query_embedding": query_embedding,
                    "k": 10
                })
                return self._build_rich_context(list(result), user_query)
                
        except Exception as e:
            print(f"Embedding search failed: {e}")
            return f"**Clinical Context:** {user_query} (embedding search error)"
    
    def _get_query_embedding(self, text: str):
        """Get embedding for query using PubMedBERT model"""
        try:
            # Import PubMedBERT model (same as used for GraphRAG entities)
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
            
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            return None
    
    def _build_rich_context(self, records, user_query):
        """Build comprehensive clinical context from semantic search results"""
        if not records:
            return f"**Clinical Context:** {user_query} (no semantic matches found)"
        
        # Extract information from top semantic matches
        primary_entities = []
        related_conditions = []
        diagnostic_tests = []
        clinical_descriptions = []
        
        for record in records[:5]:  # Top 5 semantic matches
            entity_name = record["primary_entity"]
            entity_type = record["primary_type"]
            description = record.get("primary_description", "")
            similarity = record["similarity_score"]
            
            # Collect primary entities with high similarity
            if similarity > 0.5 and entity_name:
                entity_display = entity_name.replace('_', ' ').title()
                primary_entities.append(f"{entity_display} ({similarity:.2f})")
            
            # Extract meaningful descriptions
            if description and len(description) > 30:
                # Clean description
                clean_desc = description.split('<SEP>')[0] if '<SEP>' in description else description
                clean_desc = clean_desc.strip('"').strip()
                clinical_descriptions.append(clean_desc)
            
            # Process related entities
            related_entities = record.get("related_entities", [])
            for item in related_entities:
                entity = item.get("entity")
                entity_type = item.get("type")
                
                if entity and entity_type == "DIAGNOSTIC_TEST":
                    diagnostic_tests.append(entity.replace('_', ' ').title())
                elif entity and entity_type in ["DISEASE", "CONDITION", "SYMPTOM"]:
                    related_conditions.append(entity.replace('_', ' ').title())
        
        # Build comprehensive context
        context_parts = []
        
        # Primary matches with similarity scores
        if primary_entities:
            context_parts.append(f"**Primary Medical Entities:** {', '.join(primary_entities[:3])}")
        
        # Related conditions
        if related_conditions:
            unique_conditions = list(set(related_conditions))[:3]
            context_parts.append(f"**Related Conditions:** {', '.join(unique_conditions)}")
        
        # Diagnostic considerations
        if diagnostic_tests:
            unique_tests = list(set(diagnostic_tests))[:3]
            context_parts.append(f"**Diagnostic Considerations:** {', '.join(unique_tests)}")
        
        # Clinical insights from descriptions
        if clinical_descriptions:
            best_description = max(clinical_descriptions, key=len)
            if len(best_description) > 200:
                best_description = best_description[:200] + "..."
            context_parts.append(f"**Clinical Insights:** {best_description}")
        
        # Urgency assessment
        urgency_keywords = ["acute", "emergency", "urgent", "severe", "critical"]
        is_urgent = any(keyword in user_query.lower() or 
                       any(keyword in desc.lower() for desc in clinical_descriptions)
                       for keyword in urgency_keywords)
        
        if is_urgent:
            context_parts.append("**Clinical Urgency:** Acute presentation requiring timely evaluation")
        else:
            context_parts.append("**Clinical Urgency:** Chronic condition allowing routine evaluation")
        
        return "\n".join(context_parts) if context_parts else f"**Clinical Context:** {user_query}"
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node function - processes the state and returns updates.
        This is the main entry point for the node.
        """
        if not self.initialized:
            await self.initialize()
        
        user_query = state.get("user_query", "")
        if not user_query:
            # Extract query from messages if not in state
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    user_query = message.content
                    break
        
        try:
            # Query MIMIC data directly from Neo4j
            medical_context = self._query_mimic_context(user_query)
            
            # Create response message
            response_message = AIMessage(
                content=f"Retrieved medical context from MIMIC knowledge graph",
                additional_kwargs={
                    "graphrag_result": medical_context,
                    "source": "medical_graphrag_node"
                }
            )
            
            # Return state updates
            return {
                "messages": [response_message],
                "graphrag_context": medical_context,
                "next_step": "analysis"  # Signal next node to process
            }
            
        except Exception as e:
            error_message = AIMessage(
                content=f"Medical context query failed: {str(e)}",
                additional_kwargs={"error": True, "source": "medical_graphrag_node"}
            )
            
            return {
                "messages": [error_message],
                "graphrag_context": "",
                "next_step": "error_handling"
            }

class MedicalAnalysisNode:
    """
    Example analysis node that processes GraphRAG context.
    This shows how to chain nodes in your multi-agent system.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the GraphRAG context and user query"""
        
        user_query = state.get("user_query", "")
        graphrag_context = state.get("graphrag_context", "")
        
        if not graphrag_context:
            return {
                "messages": [AIMessage(content="No GraphRAG context available for analysis")],
                "analysis_result": "No context available",
                "next_step": "final_response"
            }
        
        # Create prompt for final imaging recommendation format
        analysis_prompt = f"""
        Based on the clinical context, provide imaging recommendations in this EXACT format:

        Based on the condition {graphrag_context}, [insert brief medical assessment here]. The appropriate imaging for the patient is/are:

        Imaging: [specific imaging modality]
        Rationale: [brief clinical justification]

        References:
        1. [relevant medical source]
        2. [relevant medical source]

        Keep the medical assessment very brief (1-2 sentences max). Focus on the most appropriate imaging based on the clinical presentation: {user_query}
        """
        
        try:
            # Get analysis from LLM
            analysis_result = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            return {
                "messages": [analysis_result],
                "analysis_result": analysis_result.content,
                "next_step": "final_response"
            }
            
        except Exception as e:
            error_message = AIMessage(content=f"Analysis failed: {str(e)}")
            return {
                "messages": [error_message],
                "analysis_result": f"Analysis error: {str(e)}",
                "next_step": "error_handling"
            }

def create_medical_graphrag_workflow(llm) -> StateGraph:
    """
    Create a complete LangGraph workflow with GraphRAG integration.
    
    Args:
        llm: Your LangChain LLM instance
        
    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    
    # Initialize nodes
    graphrag_node = MedicalGraphRAGNode()
    analysis_node = MedicalAnalysisNode(llm)
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("graphrag_retrieval", graphrag_node)
    workflow.add_node("medical_analysis", analysis_node)
    workflow.add_node("final_response", lambda state: {
        "final_answer": state.get("analysis_result", "No analysis available"),
        "next_step": "complete"
    })
    
    # Define the flow
    workflow.set_entry_point("graphrag_retrieval")
    
    # Add conditional edges based on next_step
    workflow.add_conditional_edges(
        "graphrag_retrieval",
        lambda state: state.get("next_step", "analysis"),
        {
            "analysis": "medical_analysis",
            "error_handling": "final_response"
        }
    )
    
    workflow.add_conditional_edges(
        "medical_analysis", 
        lambda state: state.get("next_step", "final_response"),
        {
            "final_response": "final_response",
            "error_handling": "final_response"
        }
    )
    
    workflow.add_edge("final_response", END)
    
    return workflow.compile()

# Example usage function
async def run_medical_query_workflow(user_query: str, llm) -> str:
    """
    Run the complete medical query workflow.
    
    Args:
        user_query: User's medical question
        llm: LangChain LLM instance
        
    Returns:
        str: Final analysis result
    """
    
    # Create workflow
    workflow = create_medical_graphrag_workflow(llm)
    
    # Initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=user_query)],
        user_query=user_query,
        graphrag_context="",
        graphrag_entities=[],
        analysis_result="",
        final_answer="",
        next_step="graphrag_retrieval"
    )
    
    # Run the workflow
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state.get("final_answer", "No answer generated")

# Simplified single-node version for basic integration
class SimpleGraphRAGNode:
    """
    Simplified version for quick integration into existing LangGraph workflows.
    Just add this as a single node to your existing graph.
    """
    
    def __init__(self, working_dir: str = "./nano_graphrag_cache_simple"):
        self.graphrag_node = MedicalGraphRAGNode(working_dir)
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple node that just adds GraphRAG context to state"""
        result = await self.graphrag_node(state)
        
        # Just return the GraphRAG context for use by other nodes
        return {
            "graphrag_context": result.get("graphrag_context", ""),
            "messages": result.get("messages", [])
        } 