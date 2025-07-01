from neo4j import GraphDatabase
import os

def check_neo4j_structure():
    """Check the current Neo4j database structure to understand available ACR data."""
    
    # Neo4j connection
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    with driver.session() as session:
        print("=== NEO4J ACR_BATCH_1 DATABASE STRUCTURE ===\n")
        
        # Check node types and counts for ACR_BATCH_1 only
        result = session.run('MATCH (n) WHERE n.gid = "ACR_BATCH_1" RETURN labels(n) as labels, count(n) as count ORDER BY count DESC')
        print('ACR_BATCH_1 node types and counts:')
        for record in result:
            print(f'  {record["labels"]}: {record["count"]}')
        
        print()
        
        # Check relationships for ACR_BATCH_1 only
        result = session.run('MATCH (n)-[r]->(m) WHERE n.gid = "ACR_BATCH_1" AND m.gid = "ACR_BATCH_1" RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
        print('ACR_BATCH_1 relationship types and counts:')
        for record in result:
            print(f'  {record["rel_type"]}: {record["count"]}')
        
        print()
        
        # Check what properties different node types have in ACR_BATCH_1
        for node_type in ['Condition', 'Variant', 'Procedure']:
            result = session.run(f'MATCH (n:{node_type}) WHERE n.gid = "ACR_BATCH_1" RETURN keys(n) as props LIMIT 1')
            record = result.single()
            if record:
                print(f'{node_type} properties: {record["props"]}')
        
        print()
        
        # Sample some data to see the structure
        print("=== SAMPLE ACR_BATCH_1 DATA ===")
        
        # Get a sample condition with its variants and procedures using actual relationship names
        sample_query = """
        MATCH (c:Condition)-[r1]->(v:Variant)-[r2]->(p:Procedure)
        WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        RETURN c.id as condition, type(r1) as cond_to_variant_rel, v.id as variant, 
               type(r2) as variant_to_proc_rel, p.id as procedure
        LIMIT 10
        """
        
        result = session.run(sample_query)
        print("Sample Condition -> Variant -> Procedure chains:")
        for record in result:
            print(f"  {record['condition']} -[{record['cond_to_variant_rel']}]-> {record['variant']} -[{record['variant_to_proc_rel']}]-> {record['procedure']}")
        
        print()
        
        # Check all unique relationship types between variants and procedures
        rel_types_query = """
        MATCH (v:Variant)-[r]->(p:Procedure)
        WHERE v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        RETURN DISTINCT type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        result = session.run(rel_types_query)
        print("Variant -> Procedure relationship types:")
        for record in result:
            print(f"  {record['relationship_type']}: {record['count']}")
        
        print()
        
        # Check if there are any existing enrichment fields
        enrichment_check = """
        MATCH (p:Procedure)
        WHERE p.gid = "ACR_BATCH_1" AND 
              (p.clinical_rationale IS NOT NULL OR p.evidence_quality IS NOT NULL OR p.references IS NOT NULL)
        RETURN count(p) as enriched_count
        """
        
        result = session.run(enrichment_check)
        record = result.single()
        if record:
            print(f"ACR_BATCH_1 procedures already enriched: {record['enriched_count']}")
        
        # Total procedures in ACR_BATCH_1
        total_query = """
        MATCH (p:Procedure)
        WHERE p.gid = "ACR_BATCH_1"
        RETURN count(p) as total_count
        """
        
        result = session.run(total_query)
        record = result.single()
        if record:
            print(f"Total ACR_BATCH_1 procedures: {record['total_count']}")
        
    driver.close()

if __name__ == "__main__":
    check_neo4j_structure() 