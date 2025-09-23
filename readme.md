# Enhanced Visual Content Extraction System

**Intelligent extraction of diagrams, tables, and technical drawings from construction documents using advanced computer vision and machine learning.**

## System Overview

This system extracts specific visual content (anchor diagrams, pressure tables, glazing details, impact ratings) from PDF construction documents using a multi-method approach combining sliding window detection, traditional computer vision techniques, and similarity-based classification.

### Key Capabilities

- **Multi-Scale Detection**: Systematic scanning at 7 different window sizes (150x150 to 500x250 pixels)
- **Visual Content Focus**: Targets actual diagrams and tables, not text regions
- **Construction-Specific**: Validates content using construction drawing characteristics
- **Active Learning**: Improves through Azure OpenAI vision feedback
- **Quality Ranking**: Prioritizes best candidates using comprehensive scoring

## Quick Start

### Prerequisites

```bash
pip install opencv-python pillow pdf2image numpy langchain-openai streamlit pandas
```

**System Requirements:**

- PDF processing: `poppler-utils` (Linux/macOS) or Poppler Windows binary
- Python 3.8+
- Optional: Azure OpenAI account for vision feedback

### Basic Usage

1. **Setup training data**:

```bash
python adaptive_agent.py --setup-labeled-data
# Add visual examples to labeled_data/anchors/, labeled_data/design_pressure/, etc.
```

2. **Process a document**:

```bash
# Single characteristic
python adaptive_agent.py --source document.pdf --characteristic anchors --debug

# All characteristics
python adaptive_agent.py --source document.pdf --all-characteristics
```

3. **View results**:

```bash
streamlit run feedback_interface.py
```

## Architecture

### Detection Pipeline

**Stage 1: Multi-Method Region Detection**

- **Sliding Window**: Systematic scanning with 50% overlap at multiple scales
- **Table Detection**: Morphological line detection for structured data
- **Diagram Detection**: Edge analysis and contour detection for technical drawings
- **Content Block Detection**: High-contrast structured visual elements

**Stage 2: Quality Assessment & Ranking**

- Content scoring based on construction-specific features
- Quality ranking using detection method, size, aspect ratio, and visual characteristics
- Overlap resolution keeping highest-quality candidates

**Stage 3: Visual Classification**

- Multi-dimensional similarity: Feature-based (60%) + Visual similarity (40%)
- 40+ extracted features including edge density, line structures, texture patterns
- Construction validation using technical drawing characteristics

**Stage 4: Active Learning Feedback**

- Azure OpenAI vision analysis of extraction quality
- Intelligent parameter tuning based on accuracy metrics
- Continuous system improvement

### File Structure

```
enhanced-construction-doc-system/
├── adaptive_agent.py                    # Main processing engine
├── characteristic_based_extractor.py    # Core visual detection system
├── llm_feedback.py                     # Azure OpenAI vision feedback
├── feedback_interface.py               # Streamlit results viewer
├── diagnostic.py                       # System validation & testing
├── README.md                           # This file
├── .env                               # Azure OpenAI credentials
├── labeled_data/                      # Visual training examples
│   ├── anchors/                      # Anchor detail diagrams
│   ├── design_pressure/              # Pressure tables & charts
│   ├── glazing/                     # Glazing specifications
│   └── impact_rating/               # Impact test results
├── feedback_data/                    # Extraction results
└── learning_parameters.json          # Auto-tuned system parameters
```

## Training Data Guidelines

**Critical**: The system requires VISUAL training examples, not text descriptions.

### Effective Training Examples

- **Anchors**: Technical drawings of fastener details, connection assemblies, bolt specifications
- **Design Pressure**: Data tables with pressure ratings, wind load charts, performance matrices
- **Glazing**: Cross-section drawings, glass specifications, IGU assembly details
- **Impact Rating**: Test result tables, compliance charts, certification data with visual elements

### Training Data Quality

- **Resolution**: 200x200+ pixels minimum (higher preferred)
- **Content**: Focus on diagrams, tables, charts, technical drawings
- **Quantity**: 5-10 diverse examples per category
- **Avoid**: Pure text paragraphs, unclear scans, non-technical content

## Configuration

### System Parameters

Parameters are auto-tuned based on extraction performance:

```json
{
  "confidence_threshold": 0.5, // Classification confidence (0.25-0.85)
  "min_region_size": 20000, // Minimum region area in pixels
  "max_region_size": 2000000, // Maximum region area in pixels
  "similarity_threshold": 0.6 // Training similarity threshold
}
```

### Azure OpenAI Setup (Optional)

Create `.env` file:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

## Usage Examples

### Single Document Processing

```bash
# Process with debug output
python adaptive_agent.py --source specs.pdf --characteristic anchors --debug

# Expected output shows detection process:
# "Sliding window stats: 4666 total positions, 612 analyzed, 12 candidates found"
# "Region quality scores (top 10): sliding_window: 0.550 (22,500 pixels)"
```

### Batch Processing

```bash
# Process all characteristics
python adaptive_agent.py --source document.pdf --all-characteristics

# Results:
# anchors -> doc_id_1 (8 items extracted)
# design_pressure -> doc_id_2 (5 items extracted)
# glazing -> doc_id_3 (12 items extracted)
# impact_rating -> doc_id_4 (3 items extracted)
```

### Vision Feedback & Optimization

```bash
# Analyze extraction quality
python llm_feedback.py --analyze-and-apply doc_id_123

# Expected improvements:
# "Accuracy: 87.5%, Visual content rate: 100.0%"
# "Applied 2 parameter adjustments based on vision analysis"
```

## System Diagnostics

### Comprehensive Testing

```bash
# Full system validation
python diagnostic.py --full

# Quick health check
python diagnostic.py --quick

# Test specific components
python diagnostic.py --test visual-detection
```

### Performance Monitoring

```bash
# Check visual detection capabilities
python adaptive_agent.py --visual-test

# Verify system configuration
python adaptive_agent.py --test-system
```

## Expected Performance

### Processing Metrics

- **Throughput**: ~4 pages/minute with thorough analysis
- **Coverage**: 99%+ region coverage via sliding window
- **Accuracy**: 70-85% with proper training data
- **Detection Rate**: 0.2-1.0 items per page depending on content

### Quality Indicators

- **High similarity scores**: 0.6-0.9 to training examples
- **Construction validation**: Pass rate >80% for technical content
- **Visual content focus**: 100% visual elements (no text extraction)

## Troubleshooting

### Common Issues

**No Extractions Found**

```bash
# Check training data quality
python adaptive_agent.py --setup-labeled-data
# Ensure visual examples match document content style

# Verify system detection
python adaptive_agent.py --source doc.pdf --characteristic anchors --debug
# Look for "sliding window stats" and similarity scores
```

**Low Accuracy**

```bash
# Run vision feedback
python llm_feedback.py --analyze-and-apply doc_id

# Add more diverse training examples
# Focus on visual content similar to target documents
```

**Processing Too Slow**

- Expected: 30-60 seconds per characteristic (thorough analysis)
- If >2 minutes: Check document size, reduce pages if necessary

**Construction Validation Failures**

- Review training data for construction-appropriate content
- Ensure examples contain technical drawing characteristics
- Avoid logos, badges, or decorative elements in training data

### Debug Output Analysis

```
Construction validation: True/False
Reason: Construction: 2.5, Logo: 1.0
Confidence boost: 0.15
```

- **True + high boost**: Good technical content
- **False**: Likely logo/badge, review training data alignment

## System Versions

**v1.0**: Basic grid-based extraction (deprecated)
**v2.0**: Enhanced sliding window with construction validation (current)

### Recent Improvements (v2.0)

- Sliding window detection for comprehensive coverage
- Construction-appropriate validation logic
- Quality-based region ranking and selection
- Multi-dimensional visual similarity analysis
- Enhanced Azure OpenAI vision feedback integration

## Support & Development

### Getting Help

- Run `python diagnostic.py --full` for system health check
- Use `--debug` flag for detailed processing information
- Check `feedback_data/` for extraction results and metadata

### Contributing Training Data

Focus on clear, high-resolution visual examples:

- Technical drawings with dimension lines
- Data tables with structured information
- Assembly diagrams with detail callouts
- Charts and graphs with construction data

This system represents a comprehensive solution for extracting visual content from technical documents, specifically designed for construction industry applications with focus on accuracy, reliability, and ease of use.
