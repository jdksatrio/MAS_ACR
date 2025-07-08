"""
LangGraph Integration for Medical Graph RAG + ACR Retrieval

This package provides LangGraph nodes and workflows for integrating
the Medical Graph RAG system and ACR procedure recommendations 
into multi-agent LLM workflows.

Basic Usage:
    from langgraph_integration import SimpleGraphRAGNode, MedicalGraphRAGNode
    
    # Simple integration
    graphrag_node = SimpleGraphRAGNode()
    workflow.add_node("graphrag", graphrag_node)
    
    # Full workflow
    from langgraph_integration import create_medical_graphrag_workflow
    workflow = create_medical_graphrag_workflow(llm)

Enhanced Usage with ACR Integration:
    from langgraph_integration import run_medical_workflow
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(model="alibayram/medgemma:latest", temperature=0.1)
    
    # Run combined GraphRAG + ACR workflow
    result = await run_medical_workflow(
        user_query="Patient with chest pain and shortness of breath",
        llm=llm,
        neo4j_password="your_neo4j_password"
    )

Enhanced Supervisor Usage:
    from langgraph_integration.enhanced_medical_workflow import run_enhanced_medical_workflow
    
    # Run enhanced supervisor with enriched rationales
    result = await run_enhanced_medical_workflow(
        user_query="Patient with chest pain and elevated troponins",
        llm=llm,
        neo4j_password="your_neo4j_password"
    )
"""

from .graphrag_node import (
    MedicalGraphRAGNode,
    SimpleGraphRAGNode, 
    create_medical_graphrag_workflow,
    run_medical_query_workflow,
    AgentState,
    MedicalAnalysisNode
)

# Import ACR retrieval components
from .acr_retrieval_node import (
    ACRRetrievalNode,
    ACRAnalysisNode
)

# Import enhanced medical workflow
try:
    from .enhanced_medical_workflow import (
        run_enhanced_medical_workflow,
        EnrichedRationaleRetriever,
        EnhancedMedicalSupervisorNode,
        create_enhanced_medical_workflow
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

__all__ = [
    # Core GraphRAG components
    "MedicalGraphRAGNode",
    "SimpleGraphRAGNode", 
    "create_medical_graphrag_workflow",
    "run_medical_query_workflow",
    "AgentState",
    "MedicalAnalysisNode",
    
    # ACR retrieval components
    "ACRRetrievalNode",
    "ACRAnalysisNode",
]

# Add enhanced components if available
if ENHANCED_AVAILABLE:
    __all__.extend([
        "run_enhanced_medical_workflow",
        "EnrichedRationaleRetriever", 
        "EnhancedMedicalSupervisorNode",
        "create_enhanced_medical_workflow"
    ])

__version__ = "1.0.0" 