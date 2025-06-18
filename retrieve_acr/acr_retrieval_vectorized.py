"""
ACR Retrieval Wrapper with Vector Index Support

This module provides a wrapper around MedicalProcedureRecommenderVectorized 
with debugging capabilities and vector index optimization.
"""

import os
from typing import List, Dict
from medical_procedure_recommender_vectorized import MedicalProcedureRecommenderVectorized


class ACRRetrievalVectorized:
    """
    Wrapper class for ACR procedure retrieval using VECTOR INDEXES.
    
    This class wraps MedicalProcedureRecommenderVectorized and uses
    Neo4j vector indexes for fast similarity search instead of manual calculations.
    """
    
    def __init__(self, debug: bool = True):
        """
        Initialize ACR retrieval with vector index support.
        
        Args:
            debug: Enable debugging output to show Neo4j queries
        """
        self.debug = debug
        
        # Get Neo4j password from environment
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        if self.debug:
            print("🚀 [DEBUG] Initializing VECTORIZED ACR Retrieval System...")
            print(f"🔧 [DEBUG] Neo4j URI: bolt://localhost:7687")
            print(f"🔧 [DEBUG] Using password from environment: {'✅' if os.getenv('NEO4J_PASSWORD') else '❌ (using default)'}")
        
        # Initialize the vectorized medical procedure recommender
        self.recommender = MedicalProcedureRecommenderVectorized(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j", 
            neo4j_password=neo4j_password,
            embedding_provider="pubmedbert",
            embedding_model="NeuML/pubmedbert-base-embeddings"
        )
        
        if self.debug:
            print("✅ [DEBUG] VECTORIZED ACR Retrieval System initialized")
    
    def retrieve_recommendations(self, query: str, top_conditions: int = 1) -> List[Dict]:
        """
        Retrieve ACR procedure recommendations using VECTOR INDEXES.
        
        Args:
            query: Medical query text
            top_conditions: Number of top conditions to consider
            
        Returns:
            List of procedure recommendations
        """
        if self.debug:
            print(f"\n🚀 [DEBUG] === VECTORIZED ACR Retrieval Debug Trace ===")
            print(f"🔍 [DEBUG] Input Query: '{query}'")
            print(f"🔍 [DEBUG] Top Conditions: {top_conditions}")
            print(f"🚀 [DEBUG] Using VECTOR INDEXES for similarity search")
        
        try:
            # Call the vectorized recommender
            recommendations = self.recommender.recommend_procedures(
                query_text=query, 
                top_conditions=top_conditions
            )
            
            if self.debug:
                self._debug_print_vector_queries(query)
                self._debug_print_recommendations(recommendations)
            
            # Convert to the format expected by main.py
            formatted_recommendations = []
            
            if isinstance(recommendations, dict):
                for condition_id, condition_data in recommendations.items():
                    if isinstance(condition_data, dict) and 'variants' in condition_data:
                        for variant_id, variant_data in condition_data['variants'].items():
                            if isinstance(variant_data, dict) and 'procedures' in variant_data:
                                for procedure in variant_data['procedures']:
                                    formatted_recommendations.append({
                                        'title': procedure.get('title', 'Unknown Procedure'),
                                        'appropriateness': procedure.get('appropriateness', 'N/A'),
                                        'condition_id': condition_id,
                                        'variant_id': variant_id,
                                        'procedure_id': procedure.get('id', 'N/A'),
                                        'condition_similarity': condition_data.get('similarity', 0.0),
                                        'variant_similarity': variant_data.get('similarity', 0.0)
                                    })
            
            if self.debug:
                print(f"✅ [DEBUG] Formatted {len(formatted_recommendations)} recommendations")
            
            return formatted_recommendations[:10]  # Return top 10
            
        except Exception as e:
            if self.debug:
                print(f"❌ [DEBUG] Error in retrieve_recommendations: {str(e)}")
            raise
    
    def _debug_print_vector_queries(self, query: str):
        """Print the vector index queries that are executed"""
        print(f"\n🚀 [DEBUG] === Vector Index Queries Executed ===")
        
        # Query 1: Vector similarity search for conditions
        print(f"🚀 [DEBUG] Query 1 - Vector similarity search for conditions:")
        print(f"🚀 [DEBUG]   CALL db.index.vector.queryNodes('condition_embedding_index', 5, $query_embedding)")
        print(f"🚀 [DEBUG]   YIELD node, score")
        print(f"🚀 [DEBUG]   WHERE node.gid = 'ACR_BATCH_1'")
        print(f"🚀 [DEBUG]   RETURN node.id as condition_id, score as similarity_score")
        
        # Query 2: Vector similarity search for variants
        print(f"🚀 [DEBUG] Query 2 - Vector similarity search for variants:")
        print(f"🚀 [DEBUG]   CALL db.index.vector.queryNodes('variant_embedding_index', 15, $query_embedding)")
        print(f"🚀 [DEBUG]   YIELD node, score")
        print(f"🚀 [DEBUG]   WHERE node.gid = 'ACR_BATCH_1'")
        print(f"🚀 [DEBUG]   AND EXISTS {{ MATCH (c:Condition {{id: $condition_id}})-[:HAS_VARIANT]->(node) }}")
        print(f"🚀 [DEBUG]   RETURN node.id as variant_id, score as similarity_score")
        
        # Query 3: Get procedures for variant (unchanged)
        print(f"🚀 [DEBUG] Query 3 - Get procedures for variant:")
        print(f"🚀 [DEBUG]   MATCH (v:Variant {{id: $variant_id}})-[r]->(p:Procedure)")
        print(f"🚀 [DEBUG]   WHERE type(r) IN ['USUALLY_APPROPRIATE', 'MAYBE_APPROPRIATE', 'USUALLY_NOT_APPROPRIATE']")
        print(f"🚀 [DEBUG]   RETURN type(r) as appropriateness, p.id, p.title, p.peds_rrl_dosage")
        
        print(f"🚀 [DEBUG] === PERFORMANCE IMPROVEMENT ===")
        print(f"🚀 [DEBUG] ✅ Using vector indexes instead of manual similarity calculations")
        print(f"🚀 [DEBUG] ✅ O(log n) search instead of O(n) linear search")
        print(f"🚀 [DEBUG] ✅ Much faster for large datasets")
        print(f"🚀 [DEBUG] === End Vector Index Queries ===\n")
    
    def _debug_print_recommendations(self, recommendations):
        """Print debug information about the recommendations"""
        print(f"\n🎯 [DEBUG] === Vectorized Recommendation Results ===")
        
        if isinstance(recommendations, dict):
            for condition_id, condition_data in recommendations.items():
                print(f"🎯 [DEBUG] Condition: {condition_id}")
                
                if isinstance(condition_data, dict):
                    if 'similarity' in condition_data:
                        print(f"🎯 [DEBUG]   🚀 Vector Similarity Score: {condition_data['similarity']:.4f}")
                    
                    if 'variants' in condition_data:
                        print(f"🎯 [DEBUG]   Variants: {len(condition_data['variants'])}")
                        
                        for variant_id, variant_data in condition_data['variants'].items():
                            if isinstance(variant_data, dict):
                                print(f"🎯 [DEBUG]     Variant: {variant_id}")
                                if 'similarity' in variant_data:
                                    print(f"🎯 [DEBUG]       🚀 Vector Similarity: {variant_data['similarity']:.4f}")
                                if 'procedures' in variant_data:
                                    print(f"🎯 [DEBUG]       Procedures: {len(variant_data['procedures'])}")
                                    for proc in variant_data['procedures'][:3]:  # Show first 3
                                        print(f"🎯 [DEBUG]         - {proc.get('title', 'N/A')} (Appropriateness: {proc.get('appropriateness', 'N/A')})")
        else:
            print(f"🎯 [DEBUG] Recommendations format: {type(recommendations)}")
            print(f"🎯 [DEBUG] Raw data: {str(recommendations)[:200]}...")
        
        print(f"🎯 [DEBUG] === End Vectorized Recommendations ===\n")
    
    def close(self):
        """Close the underlying recommender connection"""
        if self.debug:
            print("🔧 [DEBUG] Closing VECTORIZED ACR Retrieval System...")
        
        if hasattr(self.recommender, 'close'):
            self.recommender.close()
        
        if self.debug:
            print("✅ [DEBUG] VECTORIZED ACR Retrieval System closed")


def main():
    """Test the vectorized ACR retrieval"""
    
    # Initialize vectorized ACR retrieval
    acr_retrieval = ACRRetrievalVectorized(debug=True)
    
    # Test queries
    test_queries = [
        "26 year old woman with chest pain",
        "elderly man with abdominal pain",
        "child with headache"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🧪 Testing query: '{query}'")
        print(f"{'='*60}")
        
        try:
            recommendations = acr_retrieval.retrieve_recommendations(query, top_conditions=1)
            
            print(f"\n📋 FINAL RESULTS:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"{i}. {rec['title']}")
                print(f"   Appropriateness: {rec['appropriateness']}")
                print(f"   Condition: {rec['condition_id']} (similarity: {rec.get('condition_similarity', 0):.4f})")
                print(f"   Variant: {rec['variant_id']} (similarity: {rec.get('variant_similarity', 0):.4f})")
                print()
            
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
    
    # Close connection
    acr_retrieval.close()


if __name__ == "__main__":
    main() 