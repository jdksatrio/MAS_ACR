"""
Enhanced Medical LangGraph Integration - Interactive Mode

Simple interactive session for testing the combined GraphRAG + ACR retrieval workflow.
"""

import asyncio
import os
from getpass import getpass
from langchain_openai import ChatOpenAI

# Import our medical workflow
# Note: Run this script as: python3 -m langgraph_integration.medical_analysis
from langgraph_integration.medical_workflow import run_medical_workflow


async def main():
    """Main interactive session for medical analysis"""
    
    print("🏥 Medical Analysis - Interactive Session")
    print("=" * 45)
    print("Combines GraphRAG knowledge extraction with ACR procedure recommendations")
    
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = getpass("Enter your OpenAI API key: ")
    
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0.1
    )
    
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        neo4j_password = getpass("Enter your Neo4j password: ")
    
    print("\n📋 Workflow: GraphRAG → ACR → Combined Analysis")
    
    while True:
        print("\n" + "-" * 45)
        
        # Get user input
        query = input("\nEnter medical query (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\n🔄 Processing medical query...")
        
        try:
            result = await run_medical_workflow(
                user_query=query,
                llm=llm,
                neo4j_password=neo4j_password
            )
            
            if result['success']:
                print(f"\n✅ Analysis Complete!")
                print(f"\n{result['final_answer']}")
                
            else:
                print(f"❌ Analysis failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Session ended. Thank you!")

if __name__ == "__main__":
    asyncio.run(main()) 