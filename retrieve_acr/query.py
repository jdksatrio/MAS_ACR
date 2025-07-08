import sys
from ragatouille import RAGPretrainedModel

def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"<query>\"")
        return
    
    query_text = " ".join(sys.argv[1:])
    
    # Load model and specify the existing index
    model = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/acr_variants_index")
    
    results = model.search(query_text, k=10)
    
    print(f"Query: {query_text}")
    print(f"Results: {len(results)}")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   {result['content']}")
        print()

if __name__ == "__main__":
    main() 