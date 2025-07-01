#!/usr/bin/env python3
"""
Check what enrichment data was actually saved to Neo4j.
"""

from neo4j import GraphDatabase

def check_saved_enrichment():
    """Check what was saved to Neo4j."""
    
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    query = '''
    MATCH (p:Procedure {id: "Aortography abdomen", gid: "ACR_BATCH_1"})
    RETURN p.clinical_rationale as rationale, 
           p.evidence_quality as evidence, 
           p.references as refs,
           p.enriched_timestamp as timestamp,
           p.enrichment_source as source
    LIMIT 1
    '''
    
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        
        if record:
            print('📝 SAVED DATA IN NEO4J:')
            print('=' * 50)
            print(f'Clinical Rationale: "{record["rationale"]}"')
            print(f'Length: {len(record["rationale"]) if record["rationale"] else 0} characters')
            print()
            print(f'Evidence Quality: "{record["evidence"]}"')  
            print(f'Length: {len(record["evidence"]) if record["evidence"] else 0} characters')
            print()
            print(f'References: "{record["refs"]}"')
            print(f'Length: {len(record["refs"]) if record["refs"] else 0} characters')
            print()
            print(f'Timestamp: {record["timestamp"]}')
            print(f'Source: {record["source"]}')
            
            # Check if they're null/empty
            print(f'\n🔍 NULL/EMPTY CHECK:')
            print(f'Rationale is None: {record["rationale"] is None}')
            print(f'Evidence is None: {record["evidence"] is None}')
            print(f'References is None: {record["refs"] is None}')
            
        else:
            print('❌ No enriched procedure found')
    
    driver.close()

if __name__ == "__main__":
    check_saved_enrichment() 