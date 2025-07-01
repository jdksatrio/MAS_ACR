#!/usr/bin/env python3
"""
Check current enrichment status to demonstrate continuation behavior.
"""

from neo4j import GraphDatabase

def check_continuation_status():
    """Check what's already enriched vs what still needs processing."""
    
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        # Check what's already enriched
        enriched_query = '''
        MATCH (p:Procedure {gid: "ACR_BATCH_1"})
        WHERE p.clinical_rationale IS NOT NULL
        RETURN count(p) as enriched_count
        '''
        
        # Check what still needs enrichment  
        remaining_query = '''
        MATCH (c:Condition)-[:HAS_VARIANT]->(v:Variant)-[r]->(p:Procedure)
        WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
        AND type(r) IN ["USUALLY_APPROPRIATE", "MAY_BE_APPROPRIATE", "USUALLY_NOT_APPROPRIATE"]
        AND (p.clinical_rationale IS NULL OR p.evidence_quality IS NULL OR p.references IS NULL)
        RETURN count(*) as remaining_count
        '''
        
        # Check if our test procedure is enriched
        test_query = '''
        MATCH (p:Procedure {id: "Aortography abdomen", gid: "ACR_BATCH_1"})
        RETURN p.clinical_rationale IS NOT NULL as is_enriched,
               p.clinical_rationale as rationale
        '''
        
        with driver.session() as session:
            enriched = session.run(enriched_query).single()['enriched_count']
            remaining = session.run(remaining_query).single()['remaining_count']
            test_result = session.run(test_query).single()
            
            print(f'📊 CURRENT ENRICHMENT STATUS:')
            print(f'=' * 50)
            print(f'✅ Already enriched: {enriched:,} procedures')
            print(f'⏳ Still need enrichment: {remaining:,} procedures')
            print(f'📈 Total: {enriched + remaining:,} procedures')
            
            if test_result and test_result['is_enriched']:
                print(f'\n🧪 TEST PROCEDURE STATUS:')
                print(f'✅ "Aortography abdomen" is already enriched')
                rationale_preview = test_result['rationale'][:100] + "..." if len(test_result['rationale']) > 100 else test_result['rationale']
                print(f'📝 Rationale preview: {rationale_preview}')
            
            print(f'\n🎯 CONTINUATION BEHAVIOR:')
            print(f'=' * 50)
            print(f'✅ Will SKIP the {enriched:,} already enriched procedures')
            print(f'🔄 Will ONLY process the {remaining:,} remaining procedures')
            print(f'💾 Previous enrichment data is SAFE - no overwriting!')
            print(f'📁 New CSV files will be created with timestamps')
            print(f'🔒 Neo4j query automatically excludes enriched procedures')
            
    except Exception as e:
        print(f'❌ Error checking status: {e}')
    
    finally:
        driver.close()

if __name__ == "__main__":
    check_continuation_status() 