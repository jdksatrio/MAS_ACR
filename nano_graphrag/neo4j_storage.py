"""
Pure Neo4j storage implementations for nano-graphrag
Replaces Milvus and JSON storage with unified Neo4j storage
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .base import (
    BaseGraphStorage, 
    BaseVectorStorage, 
    BaseKVStorage,
    SingleCommunitySchema,
    TextChunkSchema
)
from ._utils import logger, EmbeddingFunc

class Neo4jVectorStorage(BaseVectorStorage):
    """Vector storage using Neo4j vector indexes"""
    
    def __init__(
        self,
        namespace: str,
        global_config: dict,
        embedding_func: EmbeddingFunc,
        embedding_dim: int = 1536,
        meta_fields: set = None,
        **kwargs
    ):
        # Initialize base class properly
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set()
        )
        self.embedding_dim = embedding_dim
        
        # Neo4j connection details
        neo4j_url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
        neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Label for vector nodes
        self.vector_label = f"{namespace}_vector"
        
    async def __aenter__(self):
        await self._ensure_vector_index()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.driver.close()
        
    async def _ensure_vector_index(self):
        """Create vector index if it doesn't exist"""
        async with self.driver.session() as session:
            # Create vector index
            try:
                await session.run(f"""
                    CREATE VECTOR INDEX {self.vector_label}_embedding_index IF NOT EXISTS
                    FOR (n:{self.vector_label}) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Vector index ensured for {self.vector_label}")
            except Exception as e:
                logger.warning(f"Vector index creation warning: {e}")
                
    async def upsert(self, data: Dict[str, Dict]):
        """Insert or update vector data"""
        # Compute embeddings from content like the original implementation
        contents = [v["content"] for v in data.values()]
        max_batch_size = 16  # Default batch size
        batches = [
            contents[i : i + max_batch_size]
            for i in range(0, len(contents), max_batch_size)
        ]
        
        # Use asyncio for concurrent embedding computation
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        
        # Store vectors with computed embeddings
        async with self.driver.session() as session:
            for i, (key, value) in enumerate(data.items()):
                await session.run(f"""
                    MERGE (n:{self.vector_label} {{id: $id}})
                    SET n.embedding = $embedding,
                        n.entity_name = $entity_name,
                        n += $properties
                """, {
                    "id": key,
                    "embedding": embeddings[i].tolist(),  # Convert numpy array to list for Neo4j
                    "entity_name": value.get("entity_name", ""),
                    "properties": {k: v for k, v in value.items() if k not in ["embedding", "entity_name", "content"]}
                })
                
    async def query(self, query: str, top_k: int = 10) -> List[Dict]:
        """Query similar vectors"""
        # Ensure vector index exists before querying
        await self._ensure_vector_index()
        
        # Check if there are any vectors in the database first
        async with self.driver.session() as session:
            check_result = await session.run(f"""
                MATCH (n:{self.vector_label})
                RETURN count(n) AS count
            """)
            record = await check_result.single()
            if record["count"] == 0:
                logger.warning(f"No vectors found in {self.vector_label}, returning empty results")
                return []
        
        # Compute embedding for the query string
        embedding = await self.embedding_func([query])
        query_embedding = embedding[0].tolist()  # Convert numpy array to list for Neo4j
        
        async with self.driver.session() as session:
            try:
                result = await session.run(f"""
                    CALL db.index.vector.queryNodes('{self.vector_label}_embedding_index', $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node.id AS id, node.entity_name AS entity_name, score,
                           properties(node) AS properties
                    ORDER BY score DESC
                """, {
                    "query_embedding": query_embedding,
                    "top_k": top_k
                })
                
                records = []
                async for record in result:
                    records.append({
                        "id": record["id"],
                        "entity_name": record["entity_name"],
                        "distance": 1 - record["score"],  # Convert similarity to distance
                        **record["properties"]
                    })
                return records
            except Exception as e:
                logger.error(f"Vector query failed: {e}")
                # Return empty results if vector query fails
                return []


class Neo4jKVStorage(BaseKVStorage):
    """Key-Value storage using Neo4j nodes"""
    
    def __init__(self, namespace: str, global_config: dict, **kwargs):
        # Initialize base class properly
        super().__init__(namespace=namespace, global_config=global_config)
        
        # Neo4j connection details
        neo4j_url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
        neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j") 
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Label for KV nodes
        self.kv_label = f"{namespace}_kv"
        
    async def __aenter__(self):
        await self._ensure_indexes()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.driver.close()
        
    async def _ensure_indexes(self):
        """Create indexes for efficient lookups"""
        async with self.driver.session() as session:
            try:
                await session.run(f"""
                    CREATE INDEX {self.kv_label}_id_index IF NOT EXISTS
                    FOR (n:{self.kv_label}) ON (n.id)
                """)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
                
    async def all_keys(self) -> List[str]:
        """Get all keys"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.kv_label})
                RETURN n.id AS id
            """)
            keys = []
            async for record in result:
                keys.append(record["id"])
            return keys
            
    async def get_by_id(self, id: str) -> Optional[Dict]:
        """Get value by key"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.kv_label} {{id: $id}})
                RETURN n.data AS data
            """, {"id": id})
            
            record = await result.single()
            if record:
                return json.loads(record["data"])
            return None
            
    async def get_by_ids(self, ids: List[str]) -> List[Optional[Dict]]:
        """Get multiple values by keys"""
        results = []
        for id in ids:
            result = await self.get_by_id(id)
            results.append(result)
        return results
        
    async def filter_keys(self, data: List[str]) -> List[str]:
        """Filter existing keys"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                UNWIND $ids AS id
                MATCH (n:{self.kv_label} {{id: id}})
                RETURN id
            """, {"ids": data})
            
            existing_keys = set()
            async for record in result:
                existing_keys.add(record["id"])
                
            return [key for key in data if key not in existing_keys]
            
    async def upsert(self, data: Dict[str, Dict]):
        """Insert or update key-value pairs"""
        async with self.driver.session() as session:
            for key, value in data.items():
                await session.run(f"""
                    MERGE (n:{self.kv_label} {{id: $id}})
                    SET n.data = $data
                """, {
                    "id": key,
                    "data": json.dumps(value, ensure_ascii=False)
                })
                
    async def drop(self):
        """Drop all data"""
        async with self.driver.session() as session:
            await session.run(f"""
                MATCH (n:{self.kv_label})
                DELETE n
            """)


class Neo4jGraphStorage(BaseGraphStorage):
    """Enhanced Neo4j graph storage"""
    
    def __init__(self, namespace: str, global_config: dict, **kwargs):
        # Initialize base class properly
        super().__init__(namespace=namespace, global_config=global_config)
        
        # Neo4j connection details
        neo4j_url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
        neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Labels for graph nodes and relationships
        self.node_label = f"{namespace}_entity"
        self.edge_label = f"{namespace}_relation"
        
    async def __aenter__(self):
        await self._ensure_indexes()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.driver.close()
        
    async def _ensure_indexes(self):
        """Create indexes for efficient graph operations"""
        async with self.driver.session() as session:
            try:
                # Index on node ID
                await session.run(f"""
                    CREATE INDEX {self.node_label}_id_index IF NOT EXISTS
                    FOR (n:{self.node_label}) ON (n.id)
                """)
                
                # Index on node entity_name
                await session.run(f"""
                    CREATE INDEX {self.node_label}_entity_name_index IF NOT EXISTS
                    FOR (n:{self.node_label}) ON (n.entity_name)
                """)
                
            except Exception as e:
                logger.warning(f"Graph index creation warning: {e}")
                
    async def has_node(self, node_id: str) -> bool:
        """Check if node exists"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.node_label} {{id: $node_id}})
                RETURN count(n) > 0 AS exists
            """, {"node_id": node_id})
            
            record = await result.single()
            return record["exists"] if record else False
            
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if edge exists"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (s:{self.node_label} {{id: $source_id}})-[r]-(t:{self.node_label} {{id: $target_id}})
                RETURN count(r) > 0 AS exists
            """, {"source_id": source_node_id, "target_id": target_node_id})
            
            record = await result.single()
            return record["exists"] if record else False
            
    async def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node data"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.node_label} {{id: $node_id}})
                RETURN properties(n) AS props
            """, {"node_id": node_id})
            
            record = await result.single()
            return dict(record["props"]) if record else None
            
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict]:
        """Get edge data"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (s:{self.node_label} {{id: $source_id}})-[r]-(t:{self.node_label} {{id: $target_id}})
                RETURN properties(r) AS props, type(r) AS rel_type
            """, {"source_id": source_node_id, "target_id": target_node_id})
            
            record = await result.single()
            if record:
                props = dict(record["props"])
                props["relationship_type"] = record["rel_type"]
                return props
            return None
            
    async def get_node_edges(self, node_id: str) -> List[Tuple[str, str]]:
        """Get all edges for a node"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.node_label} {{id: $node_id}})-[r]-(m:{self.node_label})
                RETURN n.id AS source, m.id AS target
            """, {"node_id": node_id})
            
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges
            
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """Insert or update node"""
        async with self.driver.session() as session:
            # Convert all values to strings for Neo4j compatibility
            neo4j_data = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                         for k, v in node_data.items()}
            neo4j_data["id"] = node_id
            
            # Use entity_type as additional label for better organization
            entity_type = node_data.get("entity_type", "UNKNOWN").replace('"', '').strip()
            # Sanitize entity type for use as Neo4j label (no spaces, special chars)
            sanitized_entity_type = self._sanitize_label(entity_type)
            
            # Create node with both generic label and specific entity type label
            await session.run(f"""
                MERGE (n:{self.node_label} {{id: $node_id}})
                SET n += $properties
                SET n:{sanitized_entity_type}
            """, {"node_id": node_id, "properties": neo4j_data})
            
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]):
        """Insert or update edge"""
        async with self.driver.session() as session:
            # Convert all values to strings for Neo4j compatibility
            neo4j_data = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                         for k, v in edge_data.items()}
            
            # Extract meaningful relationship type from description
            relationship_type = self._extract_relationship_type(edge_data.get("description", ""))
            
            await session.run(f"""
                MATCH (s:{self.node_label} {{id: $source_id}})
                MATCH (t:{self.node_label} {{id: $target_id}})
                MERGE (s)-[r:{relationship_type}]-(t)
                SET r += $properties
            """, {
                "source_id": source_node_id,
                "target_id": target_node_id,
                "properties": neo4j_data
            })
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize a string to be used as a Neo4j label"""
        import re
        
        # Remove quotes and strip whitespace
        sanitized = label.replace('"', '').strip()
        
        # Replace spaces and hyphens with underscores
        sanitized = re.sub(r'[\s\-]+', '_', sanitized)
        
        # Remove any characters that aren't alphanumeric or underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
        
        # Ensure it starts with a letter (Neo4j requirement)
        if sanitized and not sanitized[0].isalpha():
            sanitized = f"TYPE_{sanitized}"
        
        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = "UNKNOWN"
        elif len(sanitized) > 50:
            sanitized = sanitized[:50]
            
        return sanitized.upper()
    
    def _extract_relationship_type(self, description: str) -> str:
        """Extract relationship type from description"""
        if not description:
            return "RELATED_TO"
            
        # Remove quotes and clean the description
        description_clean = description.strip().strip('"').strip("'").lower()
        
        #print(f"DEBUG: Processing relationship description: '{description_clean[:100]}...'")
        
        # Medical relationship mappings - check for keywords/patterns
        # TREATS - medications, treatments, therapies
        if any(word in description_clean for word in [
            "treat", "treating", "treats", "treatment", "therapy", "therapeutic",
            "medication for", "drug for", "prescribed for", "given for", 
            "helps with", "relieves", "manages", "controls"
        ]):
            #print("DEBUG: Matched TREATS")
            return "TREATS"
        
        # DIAGNOSES - diagnostic procedures, tests, assessments
        elif any(word in description_clean for word in [
            "diagnos", "detect", "identify", "assess", "evaluat", "examin", 
            "test", "screen", "check", "find", "discover", "determine"
        ]):
            #print("DEBUG: Matched DIAGNOSES")
            return "DIAGNOSES"
            
        # USED_FOR - general usage, application, administration
        elif any(phrase in description_clean for phrase in [
            "used to", "used for", "used in", "applied to", "employed for", 
            "utilized for", "administered", "given to", "provided to"
        ]):
            #print("DEBUG: Matched USED_FOR")
            return "USED_FOR"
        
        # CAUSES - causation, triggers, results in
        elif any(word in description_clean for word in [
            "cause", "causing", "causes", "results in", "leads to", "contributes to",
            "brings about", "produces", "triggers", "induces", "creates"
        ]):
            #print("DEBUG: Matched CAUSES") 
            return "CAUSES"
            
        # AFFECTS - impact, influence, damage
        elif any(word in description_clean for word in [
            "affect", "affects", "impact", "influence", "damage", "impair", 
            "compromise", "involve", "concern"
        ]):
            #print("DEBUG: Matched AFFECTS")
            return "AFFECTS"
        
        # HAS - patient conditions, possession
        elif any(phrase in description_clean for phrase in [
            "has", "have", "suffer", "diagnosed with", "patient with", "patient has",
            "suffering from", "presents with", "history of"
        ]):
            #print("DEBUG: Matched HAS")
            return "HAS"
            
        # PERFORMED - procedures, surgeries, interventions
        elif any(word in description_clean for word in [
            "performed", "conducted", "done", "carried out", "executed",
            "underwent", "received", "had"
        ]):
            #print("DEBUG: Matched PERFORMED")
            return "PERFORMED"
            
        # SYMPTOM_OF - symptoms, signs, manifestations
        elif any(phrase in description_clean for phrase in [
            "symptom of", "sign of", "indicates", "suggests", "manifestation of",
            "associated with", "related to", "secondary to"
        ]):
            #print("DEBUG: Matched SYMPTOM_OF")
            return "SYMPTOM_OF"
            
        # LOCATED_IN - anatomical locations
        elif any(phrase in description_clean for phrase in [
            "located in", "part of", "within", "inside", "situated in", "in the"
        ]):
            #print("DEBUG: Matched LOCATED_IN")
            return "LOCATED_IN"
            
        else:
            #print(f"DEBUG: No match found, using fallback for: '{description_clean[:50]}...'")
            return "RELATED_TO"
            
    async def node_degree(self, node_id: str) -> int:
        """Get node degree (number of connections)"""
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (n:{self.node_label} {{id: $node_id}})-[r]-()
                RETURN count(r) AS degree
            """, {"node_id": node_id})
            
            record = await result.single()
            return record["degree"] if record else 0
            
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get edge degree (sum of connected node degrees)"""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree
        
    async def clustering(self, algorithm: str):
        """Perform graph clustering"""
        if algorithm == "leiden":
            await self._leiden_clustering()
        else:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
    
    async def _leiden_clustering(self):
        """Perform Leiden clustering on the graph"""
        import networkx as nx
        from graspologic.partition import hierarchical_leiden
        from collections import defaultdict
        import json
        
        # First, build a NetworkX graph from Neo4j data
        async with self.driver.session() as session:
            # Get all nodes
            nodes_result = await session.run(f"""
                MATCH (n:{self.node_label})
                RETURN n.id AS id, properties(n) AS props
            """)
            
            # Get all edges  
            edges_result = await session.run(f"""
                MATCH (a:{self.node_label})-[r]->(b:{self.node_label})
                RETURN a.id AS source, b.id AS target, properties(r) AS props
            """)
            
            # Build NetworkX graph
            nx_graph = nx.Graph()
            
            # Add nodes
            async for record in nodes_result:
                node_id = record["id"]
                props = dict(record["props"])
                nx_graph.add_node(node_id, **props)
            
            # Add edges
            async for record in edges_result:
                source = record["source"]
                target = record["target"]
                props = dict(record["props"])
                nx_graph.add_edge(source, target, **props)
        
        if nx_graph.number_of_nodes() == 0:
            logger.warning("No nodes found for clustering")
            return
            
        logger.info(f"Clustering graph with {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        
        # Get largest connected component
        from graspologic.utils import largest_connected_component
        import html
        
        graph = nx_graph.copy()
        graph = largest_connected_component(graph)
        
        # Perform hierarchical leiden clustering
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )
        
        # Process clustering results
        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
            
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        
        # Store cluster information back to Neo4j nodes
        await self._cluster_data_to_subgraphs(node_communities)
    
    async def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        """Store cluster information in Neo4j nodes"""
        import json
        
        async with self.driver.session() as session:
            for node_id, clusters in cluster_data.items():
                await session.run(f"""
                    MATCH (n:{self.node_label} {{id: $node_id}})
                    SET n.clusters = $clusters_json
                """, {
                    "node_id": node_id,
                    "clusters_json": json.dumps(clusters)
                })

    async def community_schema(self) -> Dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        from collections import defaultdict
        import json
        from .prompt import GRAPH_FIELD_SEP
        
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
            )
        )
        
        max_num_ids = 0
        
        async with self.driver.session() as session:
            # Get all nodes with cluster information
            result = await session.run(f"""
                MATCH (n:{self.node_label})
                WHERE n.clusters IS NOT NULL
                RETURN n.id AS node_id, n.clusters AS clusters, n.source_id AS source_id,
                       [(n)-[r]-(other:{self.node_label}) | [n.id, other.id]] AS edges
            """)
            
            async for record in result:
                node_id = record["node_id"]
                clusters_json = record["clusters"]
                source_id = record["source_id"] or ""
                node_edges = record["edges"] or []
                
                if not clusters_json:
                    continue
                    
                clusters = json.loads(clusters_json)
                
                for cluster in clusters:
                    level = cluster["level"]
                    cluster_key = str(cluster["cluster"])
                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [tuple(sorted(e)) for e in node_edges]
                    )
                    if source_id:
                        results[cluster_key]["chunk_ids"].update(
                            source_id.split(GRAPH_FIELD_SEP)
                        )
                    max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))
        
        # Convert sets to lists and calculate occurrence
        for k, v in results.items():
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids if max_num_ids > 0 else 0.0
            
        return dict(results) 