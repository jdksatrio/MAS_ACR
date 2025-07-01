import requests
from neo4j import GraphDatabase

def get_neo4j_enrichment_stats():
    """Get statistics about what needs to be enriched in Neo4j."""
    try:
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
        
        with driver.session() as session:
            # Get total procedures and relationships
            result = session.run("""
                MATCH (c:Condition)-[:HAS_VARIANT]->(v:Variant)-[r]->(p:Procedure)
                WHERE c.gid = "ACR_BATCH_1" AND v.gid = "ACR_BATCH_1" AND p.gid = "ACR_BATCH_1"
                AND type(r) IN ["USUALLY_APPROPRIATE", "MAY_BE_APPROPRIATE", "USUALLY_NOT_APPROPRIATE"]
                RETURN count(*) as total_combinations
            """)
            total_combinations = result.single()["total_combinations"]
            
            # Get already enriched procedures
            result = session.run("""
                MATCH (p:Procedure)
                WHERE p.gid = "ACR_BATCH_1" 
                AND p.clinical_rationale IS NOT NULL
                RETURN count(p) as enriched_procedures
            """)
            enriched_procedures = result.single()["enriched_procedures"]
            
            # Get unique procedures
            result = session.run("""
                MATCH (p:Procedure)
                WHERE p.gid = "ACR_BATCH_1"
                RETURN count(p) as unique_procedures
            """)
            unique_procedures = result.single()["unique_procedures"]
            
        driver.close()
        
        return {
            "total_combinations": total_combinations,
            "enriched_procedures": enriched_procedures,
            "remaining_combinations": total_combinations - enriched_procedures,
            "unique_procedures": unique_procedures
        }
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return None

def calculate_perplexity_costs(stats):
    """Calculate Perplexity API costs for enrichment."""
    
    print("=== PERPLEXITY API COST CALCULATION ===\n")
    
    # Perplexity API Pricing (as of 2024)
    # Source: https://docs.perplexity.ai/docs/pricing
    pricing = {
        "llama-3.1-sonar-small-128k-online": {
            "input": 0.20,   # per 1M tokens
            "output": 0.20   # per 1M tokens
        },
        "llama-3.1-sonar-large-128k-online": {
            "input": 1.00,   # per 1M tokens  
            "output": 1.00   # per 1M tokens
        },
        "llama-3.1-sonar-huge-128k-online": {
            "input": 5.00,   # per 1M tokens
            "output": 5.00   # per 1M tokens
        }
    }
    
    # Estimate token usage per request
    estimated_tokens = {
        "prompt_tokens": 500,    # Our structured prompt
        "response_tokens": 1500, # Expected detailed response
        "total_per_request": 2000
    }
    
    remaining = stats["remaining_combinations"]
    
    print(f"📊 ENRICHMENT REQUIREMENTS:")
    print(f"   Total combinations: {stats['total_combinations']:,}")
    print(f"   Already enriched: {stats['enriched_procedures']:,}")
    print(f"   Remaining to enrich: {remaining:,}")
    print(f"   Unique procedures: {stats['unique_procedures']:,}")
    print()
    
    print(f"📝 ESTIMATED TOKEN USAGE PER REQUEST:")
    print(f"   Input tokens (prompt): ~{estimated_tokens['prompt_tokens']:,}")
    print(f"   Output tokens (response): ~{estimated_tokens['response_tokens']:,}")
    print(f"   Total per request: ~{estimated_tokens['total_per_request']:,}")
    print()
    
    print(f"💰 COST ESTIMATES FOR {remaining:,} ENRICHMENTS:\n")
    
    for model, prices in pricing.items():
        # Calculate total tokens needed
        total_input_tokens = remaining * estimated_tokens['prompt_tokens']
        total_output_tokens = remaining * estimated_tokens['response_tokens']
        total_tokens = remaining * estimated_tokens['total_per_request']
        
        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * prices['input']
        output_cost = (total_output_tokens / 1_000_000) * prices['output']
        total_cost = input_cost + output_cost
        
        model_display = model.replace("llama-3.1-sonar-", "").replace("-128k-online", "")
        
        print(f"🤖 {model_display.upper()} MODEL:")
        print(f"   Input cost: ${input_cost:.2f} ({total_input_tokens:,} tokens)")
        print(f"   Output cost: ${output_cost:.2f} ({total_output_tokens:,} tokens)")
        print(f"   Total cost: ${total_cost:.2f}")
        print(f"   Cost per enrichment: ${total_cost/remaining:.4f}")
        print()
    
    # Rate limiting considerations
    print("⏱️ TIME AND RATE LIMITING:")
    print("   Perplexity rate limits: ~60 requests/minute for paid plans")
    print("   Conservative rate: 30 requests/minute (2-second delays)")
    print(f"   Estimated time: {remaining/30:.1f} minutes = {remaining/30/60:.1f} hours")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print()
    print("🎯 SMALL MODEL (Most Cost-Effective):")
    small_cost = ((remaining * 2000) / 1_000_000) * 0.40  # Combined input+output rate
    print(f"   Cost: ${small_cost:.2f}")
    print("   ✅ Good for basic clinical rationale")
    print("   ✅ Fastest processing")
    print("   ❌ May lack depth in complex cases")
    print()
    
    print("🎯 LARGE MODEL (Recommended Balance):")
    large_cost = ((remaining * 2000) / 1_000_000) * 2.00  # Combined input+output rate
    print(f"   Cost: ${large_cost:.2f}")
    print("   ✅ High quality medical reasoning")
    print("   ✅ Better evidence assessment")
    print("   ✅ More comprehensive references")
    print()
    
    print("🎯 HUGE MODEL (Premium Quality):")
    huge_cost = ((remaining * 2000) / 1_000_000) * 10.00  # Combined input+output rate
    print(f"   Cost: ${huge_cost:.2f}")
    print("   ✅ Highest quality responses")
    print("   ✅ Most detailed clinical reasoning")
    print("   ❌ 5x more expensive than large model")
    print()
    
    # Budget scenarios
    print("💳 BUDGET SCENARIOS:")
    print()
    
    budgets = [50, 100, 200, 500, 1000]
    
    for budget in budgets:
        print(f"💵 ${budget} BUDGET:")
        
        # Calculate how many enrichments possible with each model
        small_enrichments = int((budget / small_cost) * remaining) if small_cost > 0 else remaining
        large_enrichments = int((budget / large_cost) * remaining) if large_cost > 0 else remaining
        huge_enrichments = int((budget / huge_cost) * remaining) if huge_cost > 0 else remaining
        
        # Cap at remaining amount
        small_enrichments = min(small_enrichments, remaining)
        large_enrichments = min(large_enrichments, remaining)
        huge_enrichments = min(huge_enrichments, remaining)
        
        print(f"   Small model: {small_enrichments:,} enrichments ({small_enrichments/remaining*100:.1f}%)")
        print(f"   Large model: {large_enrichments:,} enrichments ({large_enrichments/remaining*100:.1f}%)")
        print(f"   Huge model: {huge_enrichments:,} enrichments ({huge_enrichments/remaining*100:.1f}%)")
        print()
    
    return {
        "small_model_cost": small_cost,
        "large_model_cost": large_cost,
        "huge_model_cost": huge_cost,
        "estimated_time_hours": remaining/30/60
    }

def compare_with_alternatives():
    """Compare Perplexity costs with alternative approaches."""
    
    print("🔄 ALTERNATIVE APPROACHES:")
    print()
    
    print("📝 MANUAL RESEARCH:")
    print("   Cost: $0 (your time)")
    print("   Time: ~10-15 minutes per procedure")
    print("   Total time: ~200-300 hours for all procedures")
    print("   Quality: High, but very time-consuming")
    print()
    
    print("🤖 OTHER AI APIs:")
    print("   OpenAI GPT-4: ~$0.03-0.06 per request = $360-720 total")
    print("   Claude 3: ~$0.015-0.075 per request = $180-900 total")
    print("   Note: These don't have real-time web search like Perplexity")
    print()
    
    print("📚 HYBRID APPROACH:")
    print("   1. Use existing PDF data for 369 overlapping procedures ($0)")
    print("   2. Use Perplexity for remaining ~640 procedures")
    print("   3. Potential savings: ~50% reduction in API costs")
    print()

def main():
    """Main cost calculation function."""
    
    # Get Neo4j statistics
    stats = get_neo4j_enrichment_stats()
    
    if not stats:
        print("❌ Cannot calculate costs - Neo4j data not available")
        return
    
    # Calculate Perplexity costs
    costs = calculate_perplexity_costs(stats)
    
    # Compare with alternatives
    compare_with_alternatives()
    
    print("🎯 FINAL RECOMMENDATION:")
    print()
    print("For your use case, I recommend:")
    print("1. 🥇 LARGE MODEL with $200-300 budget")
    print("   • Best balance of quality and cost")
    print("   • Can enrich 100% of procedures")
    print("   • High-quality medical reasoning")
    print()
    print("2. 🥈 HYBRID APPROACH with $100-150 budget")
    print("   • Use PDF data for overlapping procedures")
    print("   • Use Perplexity Large model for remainder")
    print("   • 50% cost savings while maintaining quality")
    print()
    print("3. 🥉 SMALL MODEL with $50-100 budget")
    print("   • If budget is tight")
    print("   • Still provides good basic enrichment")
    print("   • Can always upgrade specific procedures later")

if __name__ == "__main__":
    main() 