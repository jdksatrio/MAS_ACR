#!/usr/bin/env python3
"""
Debug script to test Perplexity API response and parsing.
"""

import os
import json
import requests
from streaming_enrichment import StreamingPerplexityEnricher

def test_perplexity_response():
    """Test a single Perplexity API call and show raw response."""
    
    # Get API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create a test prompt
    prompt = """Based on the American College of Radiology (ACR) Appropriateness Criteria, provide detailed information about:

**Medical Condition:** Abdominal Aortic Aneurysm Follow-up (Without Repair)
**Clinical Variant:** Asymptomatic abdominal aortic aneurysm surveillance (without repair)
**Imaging Procedure:** Aortography abdomen
**ACR Appropriateness Rating:** Usually Not Appropriate

Please provide comprehensive information in the following structured format:

1. **CLINICAL_RATIONALE:** Explain the clinical reasoning behind why this imaging procedure is rated as "usually not appropriate" for this specific condition and variant. Include relevant pathophysiology, diagnostic considerations, and clinical decision-making factors.

2. **EVIDENCE_QUALITY:** Describe the quality and strength of scientific evidence supporting this recommendation. Include level of evidence, study types, sample sizes, and strength of recommendation according to ACR criteria.

3. **REFERENCES:** List the most relevant and authoritative sources supporting this recommendation including ACR Appropriateness Criteria documents, recent peer-reviewed medical literature, and clinical practice guidelines.

Please focus specifically on ACR Appropriateness Criteria and ensure all information is accurate and current. Format your response exactly as:

CLINICAL_RATIONALE: [detailed clinical explanation]

EVIDENCE_QUALITY: [evidence assessment and quality description]

REFERENCES: [list of relevant sources with proper citations]"""
    
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert radiologist and medical researcher specializing in ACR Appropriateness Criteria. Provide accurate, evidence-based information about medical imaging recommendations."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": ["acr.org"],
        "return_images": False,
        "return_related_questions": False
    }
    
    print("🔄 Sending test request to Perplexity...")
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n📝 RAW API RESPONSE:")
            print("=" * 60)
            print(json.dumps(data, indent=2))
            
            print("\n📋 EXTRACTED CONTENT:")
            print("=" * 60)
            content = data['choices'][0]['message']['content']
            print(content)
            
            print("\n🔗 CITATIONS:")
            print("=" * 60)
            citations = data.get('citations', [])
            for i, cite in enumerate(citations, 1):
                print(f"{i}. {cite}")
            
            # Test parsing
            print("\n🔍 TESTING PARSING:")
            print("=" * 60)
            enricher = StreamingPerplexityEnricher()
            parsed = enricher._parse_perplexity_response(content, citations)
            
            print(f"✅ Parse Success: {parsed['success']}")
            print(f"📝 Clinical Rationale: '{parsed['clinical_rationale']}'")
            print(f"📊 Evidence Quality: '{parsed['evidence_quality']}'")
            print(f"🔗 References: '{parsed['references'][:200]}...'")
            
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_perplexity_response() 