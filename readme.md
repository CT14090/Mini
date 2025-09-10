# Specialized Window Characteristics Extraction Agent

An AI-powered system that focuses on extracting specific window characteristics from construction documents with targeted LLM feedback loops.

## 🎯 Specialized Characteristics

The system focuses on four critical window characteristics:

### 1. **Anchor Types** 🔩

- Directly Into Concrete
- Directly Into Wood
- Into Wood via 1By Buck
- Into Concrete via 1By Buck
- Into Concrete via 2By Buck
- Self Drilling Screws Into Metal Structures

### 2. **Glazing Specifications** 🪟

- Glass type and thickness
- Low-E coatings and performance
- Insulated glass configurations
- Laminated vs tempered specifications

### 3. **Impact Rating** 🌪️

- Small Missile Impact
- Large Missile Impact
- Both Missile Impact
- Hurricane/storm compliance

### 4. **Design Pressure** 📊

- Design pressure tables and ratings
- Positive/negative pressure specifications
- Wind load ratings (PSF, PA, KPA)
- Structural performance data

## 🏗️ Architecture

### Base Code Structure

All characteristics share the same extraction engine but with specialized:

- **Keyword dictionaries** for each characteristic
- **Computer vision analysis** tuned for specific features
- **LLM prompts** focused on characteristic-specific evaluation
- **Parameter files** optimized for each extraction type

### Characteristic-Specific Components

- `adaptive_agent.py` - Main extraction engine with characteristic focus
- `llm_feedback.py` - Specialized LLM analysis for each characteristic
- `parameters_[characteristic].json` - Tuned parameters per characteristic
- `feedback_log_[characteristic].json` - Characteristic-specific feedback history

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Azure OpenAI Setup

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 3. Test Connection

```bash
python llm_feedback.py --test-connection
```

## 🎯 Characteristic-Specific Commands

### Extract Anchor Information

```bash
python adaptive_agent.py --source document.pdf --characteristic anchors
```

**Focus**: Extracts anchor types, fastener specifications, installation methods

### Extract Glazing Specifications

```bash
python adaptive_agent.py --source document.pdf --characteristic glazing
```

**Focus**: Extracts glass types, thicknesses, coatings, IGU specifications

### Extract Impact Ratings

```bash
python adaptive_agent.py --source document.pdf --characteristic impact_rating
```

**Focus**: Extracts missile impact ratings, hurricane compliance, test results

### Extract Design Pressure Data

```bash
python adaptive_agent.py --source document.pdf --characteristic design_pressure
```

**Focus**: Extracts pressure tables, DP ratings, wind load specifications

## 📋 Command Options

### Basic Usage

```bash
python adaptive_agent.py --source [PDF_PATH] --characteristic [TYPE]
```

### Advanced Options

```bash
# Debug mode for detailed output
python adaptive_agent.py --source document.pdf --characteristic anchors --debug

# Test current parameters
python adaptive_agent.py --characteristic glazing --test-params --source dummy.pdf
```

### LLM Feedback Commands

```bash
# Manual LLM analysis
python llm_feedback.py --analyze-characteristic anchors [DOC_ID]

# View characteristic-specific logs
python llm_feedback.py --show-log --characteristic glazing

# View all feedback logs
python llm_feedback.py --show-log
```

## 📊 Viewing Results

```bash
# Launch web interface to view all extractions
streamlit run feedback_interface.py
```

## 🔧 Parameter Tuning

Each characteristic maintains its own parameter file:

- `parameters_anchors.json` - Anchor extraction parameters
- `parameters_glazing.json` - Glazing extraction parameters
- `parameters_impact_rating.json` - Impact rating parameters
- `parameters_design_pressure.json` - Design pressure parameters

### Key Parameters

```json
{
  "confidence_threshold": 0.4, // AI confidence required
  "content_classification_threshold": 0.3, // Content relevance threshold
  "image_size_min": 150, // Minimum image size (pixels)
  "table_relevance_threshold": 2, // Keywords needed for table extraction
  "max_extractions": 20 // Maximum items per document
}
```

## 🤖 LLM Feedback System

### Automatic Analysis

The LLM analyzer evaluates:

1. **Accuracy**: How well were characteristic items identified?
2. **Completeness**: Was all relevant information captured?
3. **Relevance**: Are extracted items actually characteristic-related?

### Characteristic-Specific Prompts

Each characteristic uses specialized prompts that understand:

- Industry-specific terminology
- Expected data formats
- Quality benchmarks
- Common extraction challenges

## 📈 Workflow Examples

### Complete Anchor Analysis

```bash
# 1. Extract anchor information
python adaptive_agent.py --source window_spec.pdf --characteristic anchors

# 2. View results
streamlit run feedback_interface.py

# 3. Check LLM feedback
python llm_feedback.py --show-log --characteristic anchors

# Output:
# ✅ Found: Directly Into Concrete anchors
# ✅ Found: #10 x 3" concrete screws
# ✅ Found: Installation spacing requirements
# ⚠️  Missed: 2By buck installation details
```

### Complete Glazing Analysis

```bash
# Extract glazing specifications
python adaptive_agent.py --source glazing_doc.pdf --characteristic glazing --debug

# Output:
# ✅ Found: Low-E coating specifications
# ✅ Found: 6mm + 12mm + 6mm IGU configuration
# ✅ Found: Thermal performance data
# ✅ Found: Laminated safety glass requirements
```

### Batch Processing Multiple Characteristics

```bash
# Process same document for all characteristics
python adaptive_agent.py --source comprehensive_spec.pdf --characteristic anchors
python adaptive_agent.py --source comprehensive_spec.pdf --characteristic glazing
python adaptive_agent.py --source comprehensive_spec.pdf --characteristic impact_rating
python adaptive_agent.py --source comprehensive_spec.pdf --characteristic design_pressure
```

## 📁 File Structure

```
window-characteristics-agent/
├── adaptive_agent.py              # Main extraction engine
├── llm_feedback.py               # Characteristic-specific LLM analyzer
├── feedback_interface.py          # Streamlit viewer (unchanged)
├── requirements.txt              # Dependencies
├── .env                          # Azure OpenAI credentials
├── .env.example                  # Template
│
├── parameters_anchors.json        # Anchor extraction parameters
├── parameters_glazing.json        # Glazing extraction parameters
├── parameters_impact_rating.json  # Impact rating parameters
├── parameters_design_pressure.json # Design pressure parameters
│
├── feedback_log_anchors.json      # Anchor feedback history
├── feedback_log_glazing.json      # Glazing feedback history
├── feedback_log_impact_rating.json # Impact rating feedback history
├── feedback_log_design_pressure.json # Design pressure feedback history
│
├── feedback_data/                 # Extraction results
│   ├── anchors_extraction_[id].json
│   ├── glazing_extraction_[id].json
│   ├── impact_rating_extraction_[id].json
│   └── design_pressure_extraction_[id].json
│
└── data/
    └── input_pdfs/               # Source documents
```

## 🔍 Characteristic-Specific Features

### Anchors 🔩

- **Computer Vision**: Detects circular/hex shapes (screws, bolts)
- **Keywords**: anchor, fastener, concrete screw, buck installation
- **Tables**: Fastener schedules, installation specifications
- **Analysis**: Anchor type coverage, installation method completeness

### Glazing 🪟

- **Computer Vision**: Detects parallel lines (glass layers)
- **Keywords**: glazing, IGU, low-e, laminated, thickness
- **Tables**: Glass specifications, thermal properties
- **Analysis**: Glass type coverage, performance data completeness

### Impact Rating 🌪️

- **Computer Vision**: Detects test result tables
- **Keywords**: missile impact, hurricane, ASTM, compliance
- **Tables**: Test results, certification data
- **Analysis**: Impact type coverage, compliance verification

### Design Pressure 📊

- **Computer Vision**: Detects tabular structures
- **Keywords**: design pressure, DP, wind load, PSF
- **Tables**: Pressure ratings, load specifications
- **Analysis**: Pressure range coverage, rating completeness

## 🚨 Troubleshooting

### Common Issues

**"No extractions found"**

```bash
# Check if document contains the characteristic
python adaptive_agent.py --source doc.pdf --characteristic anchors --debug

# Lower thresholds if needed
# Edit parameters_anchors.json:
{
  "confidence_threshold": 0.25,
  "content_classification_threshold": 0.2
}
```

**"LLM feedback failed"**

```bash
# Test connection
python llm_feedback.py --test-connection

# Check credentials
cat .env
```

**"Characteristic not recognized"**

```bash
# Valid characteristics:
# anchors, glazing, impact_rating, design_pressure
python adaptive_agent.py --source doc.pdf --characteristic glazing
```

## 💡 Best Practices

### Document Quality

- Use high-resolution PDFs for better image extraction
- Ensure text is searchable (not scanned images only)
- Multi-page documents work better than single images

### Characteristic Selection

- Use **anchors** for installation and fastener specifications
- Use **glazing** for glass types and performance data
- Use **impact_rating** for compliance and test results
- Use **design_pressure** for structural and wind load data

### Parameter Optimization

- Start with default parameters
- Let LLM feedback adjust automatically
- Manual tuning for specific document types
- Monitor feedback logs for quality trends

## 📊 Success Metrics

### Extraction Quality Indicators

- **High Accuracy** (4-5/5): Correct characteristic identification
- **High Completeness** (4-5/5): All relevant data captured
- **High Relevance** (4-5/5): No false positives

### Coverage Benchmarks

- **Anchors**: 3+ anchor types identified = comprehensive
- **Glazing**: 2+ glass specifications = comprehensive
- **Impact Rating**: 1+ impact types = comprehensive
- **Design Pressure**: 2+ pressure values = comprehensive

## 🔄 Continuous Improvement

The system automatically improves through:

1. **LLM Feedback**: Analyzes each extraction for quality
2. **Parameter Adjustment**: Optimizes thresholds based on results
3. **Learning History**: Tracks improvements over time
4. **Characteristic Focus**: Specialized analysis per window feature

---

**Ready to extract specific window characteristics with AI precision!** 🎯
