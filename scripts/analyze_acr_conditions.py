import json
import pandas as pd
from collections import Counter
from neo4j import GraphDatabase

def analyze_acr_pdf_data():
    """Analyze conditions, variants, and procedures from the extracted ACR PDF data."""
    
    print("=== ANALYZING ACR PDF EXTRACTED DATA ===\n")
    
    # Load the cleaned JSON data
    try:
        with open("acr_structured_discussions_cleaned.json", "r") as f:
            data = json.load(f)
        print(f"📄 Loaded {len(data)} discussions from cleaned JSON")
    except FileNotFoundError:
        print("❌ acr_structured_discussions_cleaned.json not found")
        return
    
    # Extract conditions, variants, and procedures
    conditions = []
    variants = []
    procedures = []
    
    for discussion in data:
        condition = discussion.get('condition', '').strip()
        variant = discussion.get('variant', '').strip()
        procedure = discussion.get('procedure', '').strip()
        
        if condition:
            conditions.append(condition)
        if variant:
            variants.append(variant)
        if procedure:
            procedures.append(procedure)
    
    # Get unique counts
    unique_conditions = list(set(conditions))
    unique_variants = list(set(variants))
    unique_procedures = list(set(procedures))
    
    print(f"📊 EXTRACTED PDF DATA STATISTICS:")
    print(f"   Total discussions: {len(data)}")
    print(f"   Unique conditions: {len(unique_conditions)}")
    print(f"   Unique variants: {len(unique_variants)}")
    print(f"   Unique procedures: {len(unique_procedures)}")
    print()
    
    # Show top conditions by frequency
    condition_counts = Counter(conditions)
    print("🔝 TOP 10 CONDITIONS BY FREQUENCY:")
    for condition, count in condition_counts.most_common(10):
        print(f"   {condition}: {count} discussions")
    print()
    
    # Show sample of unique conditions
    print("📋 SAMPLE OF UNIQUE CONDITIONS:")
    for i, condition in enumerate(sorted(unique_conditions)[:15]):
        print(f"   {i+1}. {condition}")
    if len(unique_conditions) > 15:
        print(f"   ... and {len(unique_conditions) - 15} more")
    print()
    
    return {
        'total_discussions': len(data),
        'unique_conditions': len(unique_conditions),
        'unique_variants': len(unique_variants),
        'unique_procedures': len(unique_procedures),
        'condition_list': unique_conditions,
        'variant_list': unique_variants,
        'procedure_list': unique_procedures,
        'raw_data': data
    }

def analyze_current_neo4j():
    """Analyze current ACR_BATCH_1 data in Neo4j."""
    
    print("=== ANALYZING CURRENT NEO4J ACR_BATCH_1 DATA ===\n")
    
    try:
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
        
        with driver.session() as session:
            # Get conditions
            result = session.run("""
                MATCH (c:Condition) 
                WHERE c.gid = "ACR_BATCH_1" 
                RETURN c.id as condition
                ORDER BY c.id
            """)
            neo4j_conditions = [record["condition"] for record in result]
            
            # Get variants
            result = session.run("""
                MATCH (v:Variant) 
                WHERE v.gid = "ACR_BATCH_1" 
                RETURN v.id as variant
                ORDER BY v.id
            """)
            neo4j_variants = [record["variant"] for record in result]
            
            # Get procedures
            result = session.run("""
                MATCH (p:Procedure) 
                WHERE p.gid = "ACR_BATCH_1" 
                RETURN p.id as procedure
                ORDER BY p.id
            """)
            neo4j_procedures = [record["procedure"] for record in result]
            
        driver.close()
        
        print(f"📊 CURRENT NEO4J ACR_BATCH_1 STATISTICS:")
        print(f"   Conditions: {len(neo4j_conditions)}")
        print(f"   Variants: {len(neo4j_variants)}")
        print(f"   Procedures: {len(neo4j_procedures)}")
        print()
        
        # Show sample of conditions
        print("📋 SAMPLE OF NEO4J CONDITIONS:")
        for i, condition in enumerate(neo4j_conditions[:15]):
            print(f"   {i+1}. {condition}")
        if len(neo4j_conditions) > 15:
            print(f"   ... and {len(neo4j_conditions) - 15} more")
        print()
        
        return {
            'conditions': neo4j_conditions,
            'variants': neo4j_variants,
            'procedures': neo4j_procedures
        }
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return None

def compare_datasets(pdf_data, neo4j_data):
    """Compare the PDF extracted data with current Neo4j data."""
    
    if not neo4j_data:
        print("❌ Cannot compare - Neo4j data not available")
        return
    
    print("=== COMPARISON: PDF DATA vs CURRENT NEO4J ===\n")
    
    pdf_conditions = set(pdf_data['condition_list'])
    neo4j_conditions = set(neo4j_data['conditions'])
    
    pdf_variants = set(pdf_data['variant_list'])
    neo4j_variants = set(neo4j_data['variants'])
    
    pdf_procedures = set(pdf_data['procedure_list'])
    neo4j_procedures = set(neo4j_data['procedures'])
    
    # Conditions comparison
    common_conditions = pdf_conditions.intersection(neo4j_conditions)
    pdf_only_conditions = pdf_conditions - neo4j_conditions
    neo4j_only_conditions = neo4j_conditions - pdf_conditions
    
    print("🔍 CONDITIONS COMPARISON:")
    print(f"   PDF extracted: {len(pdf_conditions)}")
    print(f"   Neo4j current: {len(neo4j_conditions)}")
    print(f"   Common: {len(common_conditions)} ({len(common_conditions)/max(len(pdf_conditions),1)*100:.1f}% of PDF)")
    print(f"   PDF only: {len(pdf_only_conditions)}")
    print(f"   Neo4j only: {len(neo4j_only_conditions)}")
    print()
    
    # Show some examples of differences
    if pdf_only_conditions:
        print("📋 SAMPLE CONDITIONS ONLY IN PDF DATA:")
        for i, condition in enumerate(sorted(list(pdf_only_conditions))[:10]):
            print(f"   {i+1}. {condition}")
        if len(pdf_only_conditions) > 10:
            print(f"   ... and {len(pdf_only_conditions) - 10} more")
        print()
    
    if neo4j_only_conditions:
        print("📋 SAMPLE CONDITIONS ONLY IN NEO4J:")
        for i, condition in enumerate(sorted(list(neo4j_only_conditions))[:10]):
            print(f"   {i+1}. {condition}")
        if len(neo4j_only_conditions) > 10:
            print(f"   ... and {len(neo4j_only_conditions) - 10} more")
        print()
    
    # Procedures comparison
    common_procedures = pdf_procedures.intersection(neo4j_procedures)
    print("🔍 PROCEDURES COMPARISON:")
    print(f"   PDF extracted: {len(pdf_procedures)}")
    print(f"   Neo4j current: {len(neo4j_procedures)}")
    print(f"   Common: {len(common_procedures)} ({len(common_procedures)/max(len(pdf_procedures),1)*100:.1f}% of PDF)")
    print()
    
    # Overall assessment
    print("=== REPLACEMENT FEASIBILITY ASSESSMENT ===\n")
    
    condition_coverage = len(common_conditions) / len(neo4j_conditions) * 100 if neo4j_conditions else 0
    procedure_coverage = len(common_procedures) / len(neo4j_procedures) * 100 if neo4j_procedures else 0
    
    print(f"📊 COVERAGE ANALYSIS:")
    print(f"   PDF data would cover {condition_coverage:.1f}% of current Neo4j conditions")
    print(f"   PDF data would cover {procedure_coverage:.1f}% of current Neo4j procedures")
    print()
    
    if condition_coverage > 80 and procedure_coverage > 80:
        print("✅ RECOMMENDATION: Good candidate for replacement")
        print("   High overlap suggests PDF data could effectively replace current data")
    elif condition_coverage > 60 and procedure_coverage > 60:
        print("⚠️ RECOMMENDATION: Partial replacement feasible")
        print("   Moderate overlap - consider keeping both datasets or merging")
    else:
        print("❌ RECOMMENDATION: Replacement not recommended")
        print("   Low overlap - PDF data would lose significant existing coverage")
    
    print()
    print("💡 CONSIDERATIONS:")
    print("   • PDF data may be more recent/complete from source documents")
    print("   • Current Neo4j data may have been curated/processed differently")
    print("   • Consider data quality, completeness, and consistency")
    print("   • PDF data includes clinical rationale that current data lacks")

def main():
    """Main analysis function."""
    
    # Analyze PDF data
    pdf_analysis = analyze_acr_pdf_data()
    
    if pdf_analysis:
        # Analyze current Neo4j data
        neo4j_analysis = analyze_current_neo4j()
        
        # Compare datasets
        compare_datasets(pdf_analysis, neo4j_analysis)
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Review the coverage analysis above")
    print("   2. Consider data quality differences")
    print("   3. Decide on replacement vs. enrichment strategy")
    print("   4. Plan migration approach if replacement is chosen")

if __name__ == "__main__":
    main() 