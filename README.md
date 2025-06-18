# Medical Graph RAG

A comprehensive medical decision support system that integrates **GraphRAG knowledge graphs** with **ACR Appropriateness Criteria** and **enriched clinical rationales** to provide evidence-based imaging recommendations.

## 🎯 System Overview

The Medical Graph RAG system combines:

- **📊 GraphRAG Knowledge Graphs**: Semantic medical entity relationships from MIMIC data
- **🏥 ACR Appropriateness Criteria**: Evidence-based imaging procedure recommendations
- **🧠 PubMedBERT Embeddings**: Medical-domain optimized semantic search
- **💡 Enriched Clinical Rationales**: Expert-level explanations for medical decisions
- **🔄 LangGraph Workflow**: Orchestrated multi-agent medical analysis

## ✨ Key Features

### 🔍 **Semantic Medical Search**
- **768-dimensional PubMedBERT embeddings** for all medical entities
- **Vector similarity search** with Neo4j for fast retrieval
- **Automatic clinical concept matching** (e.g., "stroke" → TIA, cerebral infarction)

### 🏥 **ACR Integration**
- **Vectorized condition/variant matching** with high precision
- **Multiple imaging procedures** with appropriateness categories
- **Clinical rationales** from enriched medical knowledge

### 📋 **Enhanced Output Format**
```
**Imaging:** CT head without IV contrast : Usually Appropriate
**Rationale:** [Detailed clinical justification]
**References:** [ACR criteria and evidence sources]

------

**Imaging:** MRI head without IV contrast : Usually Appropriate
**Rationale:** [Alternative clinical rationale]
**References:** [Supporting evidence]
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Neo4j 5.0+** with Graph Data Science library
- **OpenAI API key**
- **Conda/Mamba** environment manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Medical-Graph-RAG.git
cd Medical-Graph-RAG
```

2. **Create conda environment:**
```bash
conda env create -f medgraphrag.yml
conda activate medgraphrag
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Neo4j:**
   - Install Neo4j with GDS plugin
   - Import MIMIC medical data and ACR criteria
   - Ensure vector indexes are created

### Running the System

```bash
cd langgraph_integration
python main.py
```

The system will prompt for:
- **OpenAI API key**
- **Neo4j password**

## 🏗️ System Architecture

```
User Query
    ↓
┌─────────────────┐
│   GraphRAG      │ ← PubMedBERT semantic search
│   Knowledge     │   3,318 medical entities
└─────────────────┘
    ↓
┌─────────────────┐
│   ACR Retrieval │ ← Vector matching conditions/variants
│   System        │   232 conditions, 1,105 variants
└─────────────────┘
    ↓
┌─────────────────┐
│   Enhanced      │ ← Enriched clinical rationales
│   Analysis      │   Expert medical explanations
└─────────────────┘
    ↓
Enhanced Medical Recommendation
```

## 📁 Repository Structure

```
Medical-Graph-RAG/
├── 📂 langgraph_integration/     # Main system components
│   ├── 🐍 main.py                # Entry point
│   ├── 🧠 graphrag_node.py       # GraphRAG with embeddings
│   ├── 🏥 acr_retrieval_node.py  # ACR matching system
│   ├── ✨ enhanced_medical_workflow.py  # Enhanced analysis
│   └── 📚 *.md                   # Documentation
├── 📂 scripts/                   # Utility scripts
├── 📂 archive/                   # Legacy implementations
├── 🔧 requirements.txt           # Python dependencies
├── 🐍 medgraphrag.yml           # Conda environment
└── 📖 ENRICHMENT_README.md      # Enrichment system docs
```

## 🎯 Example Usage

**Input:** `"acute stroke assessment"`

**Output:**
```
Based on the condition of acute stroke assessment with suspected cerebral infarction, 
timely imaging is crucial for treatment decisions. The appropriate imaging options include:

**Imaging:** CT head without IV contrast : Usually Appropriate
**Rationale:** Rapid availability and effectiveness in identifying acute hemorrhage, 
critical for initial stroke evaluation and treatment pathway decisions.
**References:**
1. ACR Appropriateness Criteria®: Acute Stroke
2. Emergency stroke imaging guidelines

------

**Imaging:** MRI head without IV contrast : Usually Appropriate  
**Rationale:** Superior soft tissue contrast for detecting acute ischemic changes
and small vessel disease, providing detailed stroke characterization.
**References:**
1. ACR Appropriateness Criteria®: Focal Neurologic Deficit
2. Advanced stroke imaging protocols

------
```

## 🔧 Key Improvements

### ✅ **PubMedBERT Integration**
- Replaced general embeddings with medical-domain optimized PubMedBERT
- **768-dimensional vectors** for consistent semantic search
- **96%+ similarity scores** for exact medical concept matches

### ✅ **Enhanced GraphRAG**
- **3,318 medical entities** with embedded descriptions
- **Automatic semantic relationship discovery**
- **Clinical context integration** with ACR recommendations

### ✅ **Multi-Procedure Output**
- **Up to 3 imaging procedures** with appropriateness categories
- **Individual clinical rationales** for each procedure
- **Evidence-based references** for clinical validation

## 📊 Performance Metrics

- **GraphRAG entities**: 3,318 with embeddings
- **ACR conditions**: 232 with vector search
- **ACR variants**: 1,105 with vector matching
- **Embedding dimensions**: 768 (PubMedBERT)
- **Search accuracy**: 96%+ for medical concepts
- **Response time**: ~2-3 seconds per query

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ACR Appropriateness Criteria** for evidence-based imaging guidelines
- **MIMIC-III Database** for medical knowledge graphs
- **PubMedBERT** for medical-domain embeddings
- **Neo4j** for graph database capabilities
- **LangGraph** for workflow orchestration

---

**🏥 Medical-Graph-RAG**: Bridging the gap between medical knowledge and clinical decision support through advanced AI and graph technologies. 