# Window Characteristics Extraction Agent

AI-powered system for extracting specific window characteristics from construction documents with reference data matching and LLM feedback.

## Features

- **Text-First Approach**: Summarizes relevant text before extracting images/tables
- **Reference Data Matching**: Uses your example images for better content identification
- **LLM Feedback**: Azure OpenAI analyzes extractions and optimizes parameters
- **Page Filtering**: Skips irrelevant pages (first 3 by default)
- **4 Characteristics**: Anchors, Glazing, Impact Rating, Design Pressure

## Quick Start

### 1. Setup

```bash
python setup.py
```

### 2. Configure Azure OpenAI (Optional but Recommended)

Edit `.env` file with your credentials:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

### 3. Add Reference Data

Add example images to `labeled_data/{characteristic}/` folders:

- `labeled_data/anchors/` - Anchor installation images
- `labeled_data/glazing/` - Glass section images
- `labeled_data/impact_rating/` - Test result images
- `labeled_data/design_pressure/` - Pressure table images

### 4. Extract Content

```bash
# Extract anchor information
python adaptive_agent.py --source document.pdf --characteristic anchors

# Extract with debug output
python adaptive_agent.py --source document.pdf --characteristic glazing --debug
```

### 5. View Results

```bash
streamlit run feedback_interface.py
```

## Commands

### Basic Extraction

```bash
python adaptive_agent.py --source document.pdf --characteristic anchors
python adaptive_agent.py --source document.pdf --characteristic glazing
python adaptive_agent.py --source document.pdf --characteristic impact_rating
python adaptive_agent.py --source document.pdf --characteristic design_pressure
```

### Debug Mode

```bash
python adaptive_agent.py --source document.pdf --characteristic anchors --debug
```

### View Parameters

```bash
python adaptive_agent.py --test-params --characteristic anchors --source dummy
```

## File Structure

```
window-agent/
├── adaptive_agent.py                 # Main extraction agent
├── feedback_interface.py             # Streamlit viewer
├── setup.py                         # Setup script
├── requirements.txt                  # Dependencies
├── .env                             # Azure OpenAI config
│
├── data/
│   └── input_pdfs/                  # Place PDFs here
│
├── labeled_data/                    # Reference data
│   ├── README.md                    # Reference instructions
│   ├── anchors/
│   │   ├── descriptions.json       # Anchor descriptions
│   │   └── [reference images]      # Add .jpg/.png files
│   ├── glazing/
│   ├── impact_rating/
│   └── design_pressure/
│
├── feedback_data/                   # Extraction results
│   ├── anchors_extraction_[id].json
│   ├── glazing_extraction_[id].json
│   └── ...
│
├── parameters_anchors.json          # Anchor parameters
├── parameters_glazing.json          # Glazing parameters
├── parameters_impact_rating.json    # Impact parameters
├── parameters_design_pressure.json  # Pressure parameters
│
└── feedback_log_[characteristic].json # LLM feedback logs
```

## Window Characteristics

### 1. Anchors

Extracts anchor types and installation methods:

- Directly Into Concrete
- Directly Into Wood
- Into Wood via 1By Buck
- Into Concrete via 1By Buck
- Into Concrete via 2By Buck
- Self Drilling Screws Into Metal

### 2. Glazing

Extracts glass specifications:

- Glass type and thickness
- Low-E coatings
- IGU configurations
- Laminated/tempered specifications

### 3. Impact Rating

Extracts impact resistance data:

- Small Missile Impact
- Large Missile Impact
- Both Missile Impact
- Hurricane compliance

### 4. Design Pressure

Extracts pressure specifications:

- Design pressure tables
- Wind load ratings
- Structural performance data

## How It Works

1. **Page Filtering**: Skips first 3 pages (logos, covers, etc.)
2. **Text Summarization**: Finds and summarizes relevant text content
3. **Image Extraction**: Extracts images using context + reference matching
4. **Table Extraction**: Finds tables with characteristic-specific data
5. **LLM Feedback**: Azure OpenAI evaluates results and improves parameters

## Parameters

Each characteristic has its own parameter file:

```json
{
  "confidence_threshold": 0.25,
  "min_section_length": 100,
  "max_extractions": 15,
  "image_size_min": 100,
  "skip_pages": 3,
  "content_classification_threshold": 0.15
}
```

These are automatically optimized by LLM feedback.

## Reference Data

Adding reference images dramatically improves accuracy:

1. **Anchors**: Add photos of anchor installations, connection details
2. **Glazing**: Add glass section drawings, IGU details
3. **Impact Rating**: Add test certificates, rating tables
4. **Design Pressure**: Add DP tables, load charts

Use descriptive filenames like `concrete_anchor_detail.jpg`.

## Troubleshooting

### "Docling not available"

```bash
pip install docling>=1.0.0
```

### "No relevant content found"

- Add reference images to `labeled_data/{characteristic}/`
- Lower thresholds in `parameters_{characteristic}.json`
- Use `--debug` to see what's being analyzed

### "LLM feedback failed"

- Configure Azure OpenAI in `.env` file
- System works without LLM but parameters won't auto-optimize

### "Computer vision not available"

```bash
pip install opencv-python Pillow numpy
```

## Dependencies

```bash
pip install docling python-dotenv streamlit requests
pip install opencv-python Pillow numpy  # For reference matching
pip install langchain-openai langchain-core  # For LLM feedback
```

## Examples

### Extract Anchors

```bash
python adaptive_agent.py --source specs.pdf --characteristic anchors
```

Output:

- Finds anchor-related text sections
- Extracts images showing connection details
- Extracts tables with fastener specifications
- Matches against your reference anchor images
- LLM optimizes parameters for better results

### View Results

```bash
streamlit run feedback_interface.py
```

See extracted images, tables, and text organized by characteristic with confidence scores.

---

**System Requirements**: Python 3.8+, ~2GB RAM for document processing
