# Enhanced Window Characteristic Document Agent

An AI-powered system that extracts and classifies window characteristics from construction documents using labeled reference data, computer vision, and Azure OpenAI feedback.

## Overview

This enhanced system transforms window document processing by:

1. **Characteristic-Specific Extraction**: Focuses on 4 key window characteristics (anchors, glazing, impact rating, design pressure)
2. **Reference Data Training**: Uses your labeled images and descriptions for accurate classification
3. **Multi-Modal Content**: Extracts text summaries, images, and tables for each characteristic
4. **AI-Powered Feedback**: Azure OpenAI analyzes extractions and optimizes parameters
5. **Continuous Learning**: System improves with each processed document and feedback cycle

## Key Features

### 🎯 Window Characteristic Focus

- **Anchors**: Connection and fastening systems (concrete, wood, buck installations)
- **Glazing**: Glass specifications, IGU details, coatings, thermal properties
- **Impact Rating**: Small/large missile ratings, ASTM compliance, hurricane certifications
- **Design Pressure**: DP ratings, wind loads, structural performance data

### 🖼️ Enhanced Content Extraction

- **Text Summaries**: Keyword-filtered summaries relevant to each characteristic
- **Image Classification**: Reference-based matching with confidence scoring
- **Table Extraction**: Filtered tables containing characteristic-specific data
- **Page Intelligence**: Smart page skipping and region detection

### 🤖 Azure OpenAI Integration

- **Comprehensive Analysis**: AI evaluates content relevance and extraction quality
- **Reference Alignment**: Compares extractions against your training data
- **Parameter Optimization**: Automatic threshold adjustments for better accuracy
- **Detailed Feedback**: Quality scores and specific improvement recommendations

## Quick Start

### 1. Installation

```bash
# Core dependencies
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils

# System dependencies (macOS with Homebrew)
brew install poppler
```

### 2. Configure Azure OpenAI

```bash
cp .env.example .env
# Edit .env with your Azure credentials:
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

### 3. Setup Reference Data

```bash
# Create the reference data structure
python adaptive_agent.py --setup-reference-data
```

This creates a `labeled_data/` directory with characteristic-specific folders:

```
labeled_data/
├── anchors/              # Anchor connection examples
├── glazing/              # Glass and glazing examples
├── impact_rating/        # Impact test and rating examples
└── design_pressure/      # Pressure and load examples
```

**Add your reference data**:

- Place example images (JPG/PNG) in each folder
- Edit the `descriptions.json` file in each folder with detailed descriptions
- The system uses these references to identify similar content in new documents

### 4. Process Documents

```bash
# Process all window characteristics
python adaptive_agent.py --source document.pdf

# Process specific characteristics only
python adaptive_agent.py --source document.pdf --characteristics anchors glazing

# With debug output
python adaptive_agent.py --source document.pdf --debug
```

### 5. Test Azure OpenAI Connection

```bash
# Verify Azure OpenAI is working
python llm_feedback.py --test-connection
```

### 6. View Results

```bash
# Launch the results viewer (if available)
streamlit run feedback_interface.py
```

## Workflow

### Step 1: Document Processing

The system converts PDF pages to images and extracts text using Docling for comprehensive analysis.

### Step 2: Characteristic Extraction

For each window characteristic:

- **Text Analysis**: Extracts keyword-filtered summaries from document text
- **Image Processing**: Uses computer vision to find regions, then matches against reference images
- **Table Extraction**: Identifies and filters tables containing relevant specifications

### Step 3: Reference-Based Classification

- **Image Matching**: SIFT/ORB features, histogram correlation, edge density comparison
- **Text Filtering**: Keyword matching against characteristic-specific vocabularies
- **Confidence Scoring**: Multi-metric similarity scoring with adaptive thresholds

### Step 4: AI Feedback and Optimization

Azure OpenAI with vision capabilities:

- **Content Evaluation**: Assesses relevance, completeness, and accuracy
- **Reference Alignment**: Compares extractions against your training data
- **Parameter Tuning**: Recommends threshold adjustments for improved performance
- **Quality Scoring**: Provides detailed feedback on extraction quality

## File Structure

```
enhanced-window-agent/
├── adaptive_agent.py              # Main processing script
├── llm_feedback.py                # Azure OpenAI feedback analyzer
├── requirements.txt               # Dependencies
├── .env                          # Azure OpenAI configuration
├── labeled_data/                 # Reference training data
│   ├── anchors/                  # Anchor reference images & descriptions
│   ├── glazing/                  # Glazing reference images & descriptions
│   ├── impact_rating/            # Impact rating references
│   └── design_pressure/          # Design pressure references
├── feedback_data/                # Extraction results
│   ├── anchors_extraction_*.json # Anchor extraction data
│   ├── glazing_extraction_*.json # Glazing extraction data
│   └── ...                       # Other characteristic extractions
├── parameters_*.json             # Characteristic-specific parameters
└── feedback_log_*.json          # AI feedback history
```

## Parameters

Each characteristic has its own optimizable parameters:

### Extraction Parameters

- **confidence_threshold** (0.2-0.8): Minimum similarity to reference data
- **content_classification_threshold** (0.15-0.4): Content relevance threshold
- **skip_pages** (0-10): Number of early pages to skip
- **image_size_min** (50-500): Minimum image region size
- **max_extractions** (5-50): Maximum items per characteristic

### Processing Parameters

- **text_summary_enabled** (true/false): Enable text summary extraction
- **reference_matching_enabled** (true/false): Enable reference data matching
- **table_relevance_threshold** (1-5): Minimum table relevance score
- **min_section_length** (50-500): Minimum text section length

## Example Usage

```bash
# First time setup
python adaptive_agent.py --setup-reference-data

# Add your reference images and descriptions to labeled_data/ folders

# Test current parameters for a characteristic
python adaptive_agent.py --test-params anchors

# Process a document for all characteristics
python adaptive_agent.py --source noa_window_document.pdf

# Process specific characteristics with debug
python adaptive_agent.py --source document.pdf --characteristics glazing impact_rating --debug

# Test Azure OpenAI integration
python llm_feedback.py --test-connection

# View feedback history for a characteristic
python llm_feedback.py --show-log --characteristic anchors

# Manual analysis of specific extraction
python llm_feedback.py --enhanced-analyze glazing doc123 --source-pdf document.pdf
```

## Reference Data Guidelines

### Image Quality

- **Resolution**: Minimum 200x200 pixels, ideally 300x300+
- **Clarity**: Sharp, high-contrast images showing characteristic details
- **Variety**: Multiple examples per characteristic type
- **Relevance**: Clear examples of the specific characteristic

### Descriptions Format

Edit `descriptions.json` in each characteristic folder:

```json
{
  "anchor_type_1": "Detailed description of this anchor type...",
  "anchor_type_2": "Another anchor variation description...",
  "installation_method": "Description of installation approach..."
}
```

### Characteristic-Specific Examples

**anchors/**: Fastener diagrams, connection details, installation specifications
**glazing/**: Glass specs, IGU details, coating information, thermal data  
**impact_rating/**: Test certificates, missile ratings, compliance documentation
**design_pressure/**: DP ratings, wind load data, structural performance charts

## Troubleshooting

### Common Issues

**"Azure OpenAI configuration missing"**

```bash
# Check .env file has all required variables
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

**"No reference data found"**

```bash
python adaptive_agent.py --setup-reference-data
# Then add your training images and descriptions
```

**"Low extraction count"**

- Add more reference images to relevant characteristics
- Lower confidence thresholds in parameters
- Check if PDF contains the expected content types

**"High irrelevant extractions"**

- Improve reference data quality and specificity
- Increase confidence and classification thresholds
- Review and refine characteristic descriptions

### Debug Commands

```bash
# Test Azure OpenAI connection
python llm_feedback.py --test-connection

# Show current parameters for characteristic
python adaptive_agent.py --test-params glazing

# Process with detailed debug output
python adaptive_agent.py --source document.pdf --debug

# View detailed feedback history
python llm_feedback.py --show-log --characteristic anchors

# Manual extraction analysis
python llm_feedback.py --enhanced-analyze impact_rating doc123 --source-pdf document.pdf --debug
```

### Performance Optimization

**Improving Accuracy:**

- Add more diverse, high-quality reference images
- Refine characteristic descriptions with specific terminology
- Use debug mode to identify classification issues
- Review AI feedback recommendations

**Reducing False Positives:**

- Increase confidence_threshold (e.g., 0.4 → 0.6)
- Increase content_classification_threshold (e.g., 0.2 → 0.3)
- Improve reference data specificity
- Remove poor quality reference images

**Increasing Recall:**

- Lower confidence thresholds moderately
- Add more reference examples covering edge cases
- Reduce skip_pages if content appears early
- Increase max_extractions limit

## Advanced Features

### AI Feedback Integration

The system automatically runs AI analysis after each extraction:

1. **Content Evaluation**: Scores extraction relevance and quality
2. **Reference Comparison**: Checks alignment with your training data
3. **Parameter Recommendations**: Suggests specific threshold adjustments
4. **Learning Loop**: Continuously improves extraction accuracy

### Characteristic-Specific Optimization

Each window characteristic has:

- Individual parameter files that adapt over time
- Specific keyword vocabularies for filtering
- Tailored reference matching algorithms
- Characteristic-focused AI analysis prompts

### Multi-Modal Output

Extractions include:

- **Text summaries** with keyword-filtered content
- **Images** with confidence scores and reference matches
- **Tables** with relevance scoring and data point analysis
- **Metadata** including extraction methods and quality metrics

## System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended for large documents
- **Storage**: ~2GB for dependencies, varies with document processing
- **Azure OpenAI**: Required for optimal feedback and parameter tuning
- **Poppler**: Required for PDF to image conversion

## Dependencies

**Core Processing:**

- pdf2image: PDF to image conversion
- opencv-python: Computer vision and image processing
- Pillow: Image manipulation and processing
- numpy: Numerical processing and feature extraction

**Text and Table Extraction:**

- docling: Advanced PDF text and table extraction
- docling-core: Core PDF processing functionality

**AI Integration:**

- langchain-openai: Azure OpenAI integration
- langchain-core: Core LangChain functionality for LLM interaction

**Environment:**

- python-dotenv: Environment variable management

## Support and Optimization

**Getting Better Results:**

1. **Improve Reference Data**: Add more high-quality, diverse examples
2. **Refine Descriptions**: Use specific terminology in descriptions.json
3. **Monitor AI Feedback**: Review feedback logs for improvement suggestions
4. **Tune Parameters**: Use AI recommendations to optimize thresholds
5. **Iterative Improvement**: System learns and improves with each processed document

**Configuration Help:**

- Verify Azure OpenAI credentials are correctly set
- Ensure reference data covers your document types
- Use debug mode to understand extraction decisions
- Review AI feedback for parameter optimization guidance

This enhanced system provides comprehensive window characteristic extraction with continuous AI-powered optimization, ensuring high accuracy and relevance for construction document analysis.
