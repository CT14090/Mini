# README.md

# Characteristic-Based Construction Document Agent - FIXED VERSION

An AI-powered system that intelligently extracts specific construction characteristics from PDF documents using diagram detection, similarity matching, and Azure OpenAI vision validation. **Fixed to prevent infinite loops and processing issues.**

## 🎯 What This System Does

### **Intelligent Content Extraction**

- **🎯 Characteristic-Specific**: Extracts only content relevant to your chosen construction characteristic
- **📊 Diagram Focus**: Prioritizes technical drawings over text paragraphs
- **🤖 Vision-Validated**: Azure OpenAI confirms extraction accuracy (optional)
- **⚡ Timeout Protection**: Prevents infinite loops with built-in processing limits

### **4 Construction Characteristics Supported**

1. **🔗 Anchors** - Anchor details, attachment methods, and fastening systems
2. **💨 Design Pressure** - Pressure ratings, wind load data, and structural calculations
3. **🔍 Glazing** - Glass specifications, glazing details, and glazing systems
4. **💥 Impact Rating** - Impact resistance ratings, test results, and compliance data

## 🚀 Quick Start

### 1. Installation

```bash
# Install Python dependencies
pip install opencv-python pillow pdf2image numpy langchain-openai streamlit

# System dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: Download poppler from https://github.com/oschwartz10612/poppler-windows
```

### 2. Setup Training Data

```bash
# Create training data structure
python adaptive_agent.py --setup-labeled-data
```

**Add your reference images** (JPG/PNG) to each category folder:

```
labeled_data/
├── anchors/              # Add anchor detail images
├── design_pressure/      # Add pressure rating images
├── glazing/             # Add glazing specification images
└── impact_rating/       # Add impact rating images
```

### 3. Configure Azure OpenAI (Optional)

Create `.env` file:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-gpt4-vision-deployment
```

### 4. Test System

```bash
# Quick system check
python diagnostic.py --quick

# Full diagnostic
python diagnostic.py --full
```

### 5. Process Documents

```bash
# List available characteristics
python adaptive_agent.py --list-characteristics

# Extract specific characteristic
python adaptive_agent.py --source document.pdf --characteristic anchors

# Extract all characteristics
python adaptive_agent.py --source document.pdf --all-characteristics

# With debug output
python adaptive_agent.py --source document.pdf --characteristic glazing --debug
```

### 6. View Results

```bash
# Launch results viewer
streamlit run feedback_interface.py
```

## 📁 File Structure

```
construction-doc-system/
├── adaptive_agent.py                    # Main processor with timeout protection
├── characteristic_based_extractor.py    # Content extraction engine
├── llm_feedback.py                     # Azure OpenAI vision feedback
├── feedback_interface.py               # Streamlit results viewer
├── diagnostic.py                       # System diagnostic tool
├── README.md                           # This file
├── .env                               # Azure OpenAI credentials (create this)
├── learning_parameters.json           # Auto-tuned parameters (auto-created)
├── labeled_data/                      # Training images
│   ├── anchors/                      # Anchor reference images
│   ├── design_pressure/              # Pressure rating references
│   ├── glazing/                     # Glazing specification references
│   └── impact_rating/               # Impact rating references
├── feedback_data/                    # Extraction results (auto-created)
└── diagnostic_output/                # Diagnostic reports (auto-created)
```

## 🔧 Key Fixes Applied

### **Infinite Loop Prevention**

- ✅ **Timeout Protection**: 5-minute processing limit per characteristic
- ✅ **Page Limits**: Maximum 20 pages processed per document
- ✅ **Region Limits**: Maximum 5 extractions per page
- ✅ **Memory Limits**: Reduced feature extraction to prevent memory issues

### **Improved Reliability**

- ✅ **Fixed Similarity Calculation**: All scores properly bounded [0,1]
- ✅ **Error Handling**: Graceful failures with informative messages
- ✅ **Safe Processing**: Individual timeouts for pages and vision analysis
- ✅ **Resource Management**: Limited Azure OpenAI calls to prevent overuse

### **Simplified Architecture**

- ✅ **Streamlined Extraction**: Grid-based region detection (most reliable)
- ✅ **Limited Feature Extraction**: Reduced SIFT features to prevent slowdowns
- ✅ **Bounded Parameters**: All thresholds validated and capped
- ✅ **Clear Error Messages**: Specific guidance when issues occur

## 📋 Usage Examples

### **Single Characteristic Extraction**

```bash
python adaptive_agent.py --source construction_specs.pdf --characteristic anchors
```

**Output:**

```
🎯 CHARACTERISTIC-BASED DOCUMENT PROCESSING
📄 Source: construction_specs.pdf
🏗️ Target Characteristic: Anchors
🆔 Document ID: abc123
⏰ Timeout: 300s

Results for Anchors:
  🔢 Total items: 3
  📄 Pages processed: 9/9
  ⏱️ Processing time: 45.2s

✅ Processing completed successfully
```

### **All Characteristics Extraction**

```bash
python adaptive_agent.py --source document.pdf --all-characteristics
```

**Output:**

```
🔄 Processing document for ALL characteristics...

✅ Completed: anchors -> abc123 (3 items)
✅ Completed: design_pressure -> def456 (2 items)
✅ Completed: glazing -> ghi789 (5 items)
✅ Completed: impact_rating -> jkl012 (1 items)

Successful: 4/4 characteristics
```

## 🧪 System Diagnostic

The diagnostic tool helps identify and fix issues:

### **Quick Check**

```bash
python diagnostic.py --quick
```

### **Full Diagnostic**

```bash
python diagnostic.py --full
```

**Sample Output:**

```
🧪 COMPREHENSIVE SYSTEM DIAGNOSTIC
✅ PASS Dependencies
✅ PASS Core Files
✅ PASS Configuration
⚠️  FAIL Training Data
✅ PASS Similarity Calculation
✅ PASS Processing Pipeline
✅ PASS Recent Extractions
⚠️  FAIL Azure OpenAI

Overall Score: 6/8 tests passed
```

### **Specific Tests**

```bash
# Test training data only
python diagnostic.py --training-data

# Test Azure OpenAI only
python diagnostic.py --azure
```

## 🎯 Training Data Guidelines

### **Quality Standards**

- **Resolution**: Minimum 200x200 pixels
- **Content**: Clear technical diagrams, not text paragraphs
- **Quantity**: 3-10 examples per characteristic
- **Format**: JPG or PNG files

### **Category Examples**

**anchors/**

- Anchor detail drawings
- Fastening system diagrams
- Attachment method illustrations

**design_pressure/**

- Pressure rating charts
- Wind load calculations
- Structural performance data

**glazing/**

- Glass specification diagrams
- Glazing system details
- IGU (Insulated Glass Unit) drawings

**impact_rating/**

- Impact test results
- Compliance certificates
- Rating classification charts

## 🤖 Azure OpenAI Vision Feedback

When configured, the system provides intelligent feedback:

### **What It Does**

- **Visual Validation**: Compares extracted images with training data
- **Classification Review**: Confirms categories match visual content
- **Parameter Tuning**: Automatically adjusts processing thresholds
- **Quality Assessment**: Identifies extraction accuracy issues

### **Sample Feedback**

```bash
python llm_feedback.py --analyze-and-apply document_id
```

**Output:**

```
🔍 Analyzing extraction abc123 (timeout: 120s)...
📊 Analyzing 5 items (limited from 12)
📈 Accuracy: 80% (4/5)
⚙️ Applying 2 parameter adjustments...
✅ Vision-based feedback completed
```

## 🔧 Troubleshooting

### **Common Issues**

**"Processing timed out"**

- Document too large or complex
- Reduce page count or split document
- Check for infinite loop indicators

**"No items extracted"**

```bash
# Check system status
python diagnostic.py --quick

# Add training data
python adaptive_agent.py --setup-labeled-data

# Process with debug
python adaptive_agent.py --source doc.pdf --characteristic anchors --debug
```

**"Wrong content extracted"**

- Add more relevant training examples
- Check training image quality
- Run vision feedback analysis

**"Azure OpenAI errors"**

```bash
# Test connection
python llm_feedback.py --test-connection

# Check .env configuration
# Verify API quotas and limits
```

### **Performance Issues**

**Too Few Extractions:**

- Add more training examples
- Check document contains target characteristic
- Review similarity thresholds in learning_parameters.json

**Too Many Extractions:**

- Improve training data quality
- Let vision feedback adjust parameters
- Increase confidence thresholds

**Processing Too Slow:**

- Large documents are automatically limited to 20 pages
- Complex pages timeout at 30 seconds each
- Vision analysis limited to 5 items maximum

## 📊 Results Analysis

### **Streamlit Interface**

```bash
streamlit run feedback_interface.py
```

**Features:**

- 📊 **Overview**: Document statistics and processing summary
- 🔍 **Details**: Item-by-item analysis with images
- 📚 **Status**: System health and training data quality
- 🤖 **Logs**: Vision feedback analysis history

### **Data Structure**

Each extraction creates detailed JSON results:

```json
{
  "document_id": "abc123",
  "target_characteristic": "anchors",
  "total_sections": 3,
  "processing_time": 45.2,
  "extraction_summary": {
    "total_items": 3,
    "diagram_items": 3,
    "table_items": 0,
    "avg_confidence": 0.73
  }
}
```

## ⚙️ Advanced Configuration

### **Processing Parameters**

Automatically tuned via `learning_parameters.json`:

```json
{
  "confidence_threshold": 0.5, // Minimum classification confidence
  "min_region_size": 10000, // Minimum region area (pixels)
  "similarity_threshold": 0.6 // Training data similarity threshold
}
```

### **Timeout Settings**

Built-in limits (not configurable to prevent infinite loops):

- **Total processing**: 300 seconds per characteristic
- **Page processing**: 30 seconds per page
- **Vision analysis**: 120 seconds total
- **Page limit**: 20 pages maximum

## 🆘 Getting Help

### **Check System Status**

```bash
python diagnostic.py --full
```

### **View Processing Logs**

```bash
python llm_feedback.py --show-log
```

### **Test Components**

```bash
# Test training data
python diagnostic.py --training-data

# Test Azure OpenAI
python diagnostic.py --azure

# Test extraction
python adaptive_agent.py --test-system
```

## 📈 System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~1GB for dependencies + training data
- **Network**: For Azure OpenAI (optional)
- **OS**: Windows, macOS, Linux (with poppler)

## 🔄 Workflow Integration

### **Batch Processing**

```bash
# Process multiple PDFs for single characteristic
for pdf in *.pdf; do
    python adaptive_agent.py --source "$pdf" --characteristic anchors
done

# Process single PDF for all characteristics
python adaptive_agent.py --source document.pdf --all-characteristics
```

### **API Integration**

```python
from characteristic_based_extractor import CharacteristicBasedExtractor

extractor = CharacteristicBasedExtractor()
characteristics = extractor.get_available_characteristics()
# Returns: ['anchors', 'design_pressure', 'glazing', 'impact_rating']
```

---

## 🎯 Key Changes Made

This version fixes the critical infinite loop and processing issues while maintaining the sophisticated characteristic-based extraction capabilities. The system now has proper timeout protection, resource limits, and simplified processing logic that prevents the runaway processing you experienced.

The core functionality remains the same - intelligent extraction of construction characteristics using training data and optional Azure OpenAI validation - but now it's reliable and won't get stuck in infinite processing loops.
