"""
PubMedBERT Embedding Provider

This module provides embeddings using the NeuML/pubmedbert-base-embeddings model
for medical text understanding. This model is specifically fine-tuned for medical 
literature and provides superior performance for medical queries.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch


class PubMedBERTEmbeddingProvider:
    """
    Embedding provider using PubMedBERT model optimized for medical text.
    
    Uses the NeuML/pubmedbert-base-embeddings model which provides 768-dimensional
    embeddings specifically trained on medical literature.
    """
    
    def __init__(self, model_name: str = "NeuML/pubmedbert-base-embeddings", device: str = None):
        """
        Initialize the PubMedBERT embedding provider.
        
        Args:
            model_name: HuggingFace model name (default: NeuML/pubmedbert-base-embeddings)
            device: Device to run the model on (cuda/cpu/mps). Auto-detected if None.
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"🚀 Loading PubMedBERT model on device: {device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ PubMedBERT model loaded successfully")
            print(f"📊 Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Error loading PubMedBERT model: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.get_embeddings_batch([text1, text2], show_progress=False)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def __str__(self):
        return f"PubMedBERTEmbeddingProvider(model={self.model_name}, device={self.device}, dim={self.embedding_dim})"


def test_pubmedbert_provider():
    """Test function to verify PubMedBERT provider works correctly."""
    print("=== Testing PubMedBERT Embedding Provider ===\n")
    
    try:
        # Initialize provider
        provider = PubMedBERTEmbeddingProvider()
        print(f"Provider: {provider}\n")
        
        # Test single embedding
        test_text = "Patient presents with chest pain and shortness of breath"
        embedding = provider.get_embedding(test_text)
        print(f"Test text: {test_text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding type: {type(embedding)}")
        print(f"First 5 values: {embedding[:5]}\n")
        
        # Test batch embeddings
        test_texts = [
            "Acute myocardial infarction with ST elevation",
            "Pneumonia with bilateral infiltrates",
            "Chronic kidney disease stage 3"
        ]
        
        batch_embeddings = provider.get_embeddings_batch(test_texts, show_progress=False)
        print(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test similarity
        similarity = provider.compute_similarity(test_texts[0], test_texts[1])
        print(f"Similarity between '{test_texts[0]}' and '{test_texts[1]}': {similarity:.4f}")
        
        print("\n✅ PubMedBERT provider test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ PubMedBERT provider test failed: {e}")
        return False


if __name__ == "__main__":
    test_pubmedbert_provider() 