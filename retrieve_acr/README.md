# Fine-tuned ColBERT Model for ACR Appropriateness Criteria

Ready-to-use package with a fine-tuned ColBERT model for ACR (American College of Radiology) appropriateness criteria matching. **Pre-built index included** - no setup time required!

## Quick Start (30 seconds)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Query the model:**
```bash
python query.py "45-year-old male with acute chest pain"
```

That's it! Get relevant ACR criteria instantly.

## Model Performance

- **Top-1 Accuracy**: 51% (2.3x improvement over baseline 22%)
- **Top-3 Accuracy**: 77%
- **Top-5 Accuracy**: 90%
- **Top-10 Accuracy**: 95%
- **Index Size**: 1,105 unique ACR appropriateness criteria variants
- **No GPU required** for inference

## Usage Examples

```bash
# Chest pain scenarios
python query.py "patient with chest pain and shortness of breath"

# Orthopedic cases  
python query.py "elderly woman with hip fracture after fall"

# Pediatric cases
python query.py "child with fever and abdominal pain"

# Trauma cases
python query.py "28M with shoulder dislocation after sports injury"
```

## Package Contents

```
query.py                 # Simple query script - returns top 10 results
checkpoints/colbert/     # Fine-tuned model weights (~419MB)
.ragatouille/            # Pre-built index (~2.6MB) - ready to use!
requirements.txt         # Dependencies (ragatouille, torch, etc.)
README.md               # This file
```

## Integration in Your Code

```python
# Import and use directly
import sys
from ragatouille import RAGPretrainedModel

# Load model with pre-built index
model = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/acr_variants_index")

# Search for relevant ACR variants
results = model.search("patient description here", k=10)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content']}")
```

## LangGraph/Multi-Agent Integration

```python
class ACRRetrieverTool:
    def __init__(self):
        self.model = RAGPretrainedModel.from_index(
            ".ragatouille/colbert/indexes/acr_variants_index"
        )
    
    def retrieve_acr_variants(self, patient_description: str, k: int = 5):
        results = self.model.search(patient_description, k=k)
        return [
            {
                "variant": result['content'],
                "relevance_score": result['score']
            }
            for result in results
        ]
```

## Technical Details

- **Base Model**: ColBERT v2.0 (colbert-ir/colbertv2.0)
- **Training Data**: 31,800 synthetic patient case triplets
- **Architecture**: Late interaction retrieval with dense embeddings
- **Index Type**: IVF-HNSW for efficient similarity search
- **Memory Usage**: ~1GB RAM when loaded
- **Response Time**: ~1-2 seconds per query

## Requirements

- Python 3.8+
- Dependencies in `requirements.txt`
- ~422MB disk space total
- No GPU required

## What's Different from Original

This package includes:
- ✅ **Pre-built index** - no 2-3 minute setup wait
- ✅ **Simple query script** - one command to get results  
- ✅ **Clean file structure** - only essential files included
- ✅ **Fast loading** - ready to query immediately

Perfect for production deployments and easy reproduction! 