"""
Enhanced Medical Supervisor - Simplified

Integrates GraphRAG knowledge, ACR procedure recommendations, 
and enriched clinical rationales from Perplexity API.
"""

import asyncio
import os
from getpass import getpass
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import enhanced supervisor workflow
try:
    from .enhanced_medical_workflow import run_enhanced_medical_workflow
    print("Enhanced supervisor loaded")
except ImportError:
    try:
        from enhanced_medical_workflow import run_enhanced_medical_workflow
        print("Enhanced supervisor loaded")
    except ImportError as e:
        print(f"Enhanced supervisor not available: {e}")
        exit(1)

def main():
    """Main interface for the enhanced medical supervisor"""
    
    print("Enhanced Medical Supervisor")
    print("==================================================")
    print("Integrates: GraphRAG + ACR + Enriched Clinical Rationales")
    
    # Get Neo4j credentials
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password or neo4j_password == "your_neo4j_password_here":
        print("Warning: NEO4J_PASSWORD not found in environment variables or still has placeholder value.")
        neo4j_password = getpass("Enter Neo4j password: ")
    else:
        print("✓ Neo4j password loaded from environment")
    
    # Initialize local LLM
    llm = ChatOllama(
        model="alibayram/medgemma:latest",
        temperature=0.0
    )
    print("Using Local alibayram/medgemma:latest")
    
    print("System ready!")
    
    print("\n--------------------------------------------------")
    
    while True:
        try:
            query = input("\nEnter medical query (or 'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: '{query}'")
            print("Integrating GraphRAG + ACR + Enriched Rationales...")
            
            # Run enhanced workflow
            result = asyncio.run(run_enhanced_medical_workflow(
                user_query=query,
                llm=llm,
                neo4j_password=neo4j_password
            ))
            
            if result["success"]:
                print("\nEnhanced Analysis Complete!")
                
                # Get results
                graphrag_context = result.get("graphrag_context", "")
                acr_recommendations = result.get("acr_recommendations", {})
                enriched_rationales = result.get("enriched_rationales", {})
                
                print(f"\nData Sources:")
                print(f"   GraphRAG: {'Available' if graphrag_context else 'Unavailable'} (length: {len(graphrag_context) if graphrag_context else 0})")
                print(f"   ACR: {'Available' if acr_recommendations and 'error' not in acr_recommendations else 'Unavailable'}")
                
                # Enhanced rationale status
                if enriched_rationales and "error" not in enriched_rationales:
                    summary = enriched_rationales.get("summary", {})
                    enriched_count = summary.get("enriched_procedures", 0)
                    total_count = summary.get("total_procedures", 0)
                    print(f"   Enriched: Available ({enriched_count}/{total_count} procedures)")
                else:
                    print(f"   Enriched: Unavailable")
                
                # Show GraphRAG context preview
                if graphrag_context and graphrag_context.strip():
                    print(f"\nGraphRAG Preview: {graphrag_context[:200]}...")
                else:
                    print(f"\nGraphRAG returned empty context")
                
                # Show main analysis
                final_answer = result.get("final_answer", "")
                if final_answer:
                    print(f"\nENHANCED ANALYSIS:")
                    print("==================================================")
                    print(final_answer)
                
            else:
                print(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print(f"\nError: {str(e)}")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nEnhanced medical analysis session ended!")


if __name__ == "__main__":
    main() 