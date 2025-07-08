# Perplexity Enrichment System

## 🚀 Quick Start
```bash
python3 streaming_enrichment.py
```

## 📁 Directory Structure

### **Main Files**
- `streaming_enrichment.py` - Main enrichment system with priority options
- `run_enrichment.py` - Environment checker and launcher

### **Organized Folders**

#### `enrichment_outputs/`
- `enrichment_progress_YYYYMMDD_HHMMSS.csv` - Progress tracking files
- `enrichment_state_YYYYMMDD_HHMMSS.json` - Session state for resuming

#### `scripts/`
- `analyze_acr_conditions.py` - Data overlap analysis
- `calculate_perplexity_cost.py` - Cost estimation
- `check_neo4j_structure.py` - Database structure analysis
- `check_continuation_status.py` - Status checker
- `debug_perplexity_response.py` - API response testing
- `test_priority_breakdown.py` - Priority level testing

#### `archive/`
- `perplexity_enrichment.py` - Original enrichment system
- `perplexity_enrichment_streaming.py` - Early streaming version

## 🎯 Priority Options

1. **Full enrichment** - All procedures (~11,331)
2. **Test single procedure** - Validation test  
3. **Limited enrichment** - Specify count
4. **Resume from checkpoint** - Continue previous session
5. **Usually Appropriate only** - ~2,373 procedures (~$15, ~2 hours)
6. **Usually + May Be Appropriate** - ~5,179 procedures (~$25, ~3.5 hours)
7. **Usually Not Appropriate only** - ~6,857 procedures (~$25, ~3.5 hours)

## 💾 Safety Features

- **No overwriting** - Skips already enriched procedures
- **Progress saving** - Auto-save every 5 procedures
- **Safe cancellation** - Ctrl+C saves and exits gracefully
- **Resume capability** - Can restart from any point
- **Organized outputs** - All files saved in dedicated folders

## 🔄 Continuation Behavior

The system automatically:
- ✅ **Skips** procedures already enriched
- 🔄 **Processes** only remaining procedures  
- 💾 **Preserves** all existing data
- 📁 **Creates** new timestamped files per session 