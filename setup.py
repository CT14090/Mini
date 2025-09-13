#!/usr/bin/env python3
"""
Setup script for Window Characteristics Extraction Agent
"""
import os
import pathlib
import json

def setup_system():
    print("Setting up Window Characteristics Extraction Agent")
    print("=" * 50)
    
    # Create directories
    directories = [
        "data/input_pdfs",
        "feedback_data", 
        "labeled_data/anchors",
        "labeled_data/glazing",
        "labeled_data/impact_rating",
        "labeled_data/design_pressure"
    ]
    
    for directory in directories:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}/")
    
    # Create reference descriptions
    descriptions = {
        "anchors": {
            "concrete_anchor": "Anchor bolts or screws going directly into concrete substrate",
            "wood_anchor": "Screws or fasteners going directly into wood framing", 
            "buck_anchor": "Anchors going through wood buck into substrate",
            "self_drilling": "Self-drilling screws for metal structures",
            "structural_connection": "Connection details showing anchor placement and spacing"
        },
        "glazing": {
            "glass_section": "Cross-section showing glass layers and thickness",
            "igu_detail": "Insulated glass unit construction details",
            "glazing_spec": "Glass specifications and performance data",
            "low_e_coating": "Low-E coating application and properties"
        },
        "impact_rating": {
            "missile_test": "Impact test results and missile impact ratings",
            "hurricane_rating": "Hurricane and storm resistance certifications",
            "test_compliance": "ASTM/AAMA test compliance documentation"
        },
        "design_pressure": {
            "pressure_table": "Design pressure ratings and load tables",
            "wind_load": "Wind load specifications and ratings",
            "structural_performance": "Structural performance and load resistance"
        }
    }
    
    for char_type, desc_dict in descriptions.items():
        desc_file = pathlib.Path(f"labeled_data/{char_type}/descriptions.json")
        with open(desc_file, 'w') as f:
            json.dump(desc_dict, f, indent=2)
        print(f"Created: {desc_file}")
    
    # Create default parameters
    default_params = {
        "confidence_threshold": 0.25,
        "min_section_length": 100,
        "max_extractions": 15,
        "image_size_min": 100,
        "skip_pages": 3,
        "content_classification_threshold": 0.15
    }
    
    for char_type in descriptions.keys():
        param_file = f"parameters_{char_type}.json"
        with open(param_file, 'w') as f:
            json.dump(default_params, f, indent=2)
        print(f"Created: {param_file}")
    
    # Create .env template if it doesn't exist
    if not os.path.exists(".env"):
        env_content = """# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("Created: .env (please configure with your Azure OpenAI credentials)")
    
    # Create README
    readme_content = """# Reference Data Instructions

## Adding Reference Images

1. Add example images to the appropriate folder:
   - `labeled_data/anchors/` - Images showing anchor installations, connection details
   - `labeled_data/glazing/` - Images showing glass sections, IGU details
   - `labeled_data/impact_rating/` - Images showing test results, certifications
   - `labeled_data/design_pressure/` - Images showing pressure tables, load data

2. Use descriptive filenames (e.g., `concrete_anchor_detail.jpg`, `igu_section.png`)

3. Supported formats: .jpg, .jpeg, .png

4. The system uses these images to better identify relevant content in documents

## Example Reference Images to Add

### Anchors
- Photos of screw/bolt installations
- Technical drawings showing anchor placement
- Connection detail drawings
- Fastener schedule tables

### Glazing  
- Glass section drawings
- IGU construction details
- Glazing specification tables

### Impact Rating
- Test certificates
- Impact rating tables
- Compliance documents

### Design Pressure
- DP rating tables
- Wind load charts
- Performance data tables
"""
    
    with open("labeled_data/README.md", "w") as f:
        f.write(readme_content)
    print("Created: labeled_data/README.md")
    
    print("\nSetup Complete!")
    print("\nNext Steps:")
    print("1. Configure Azure OpenAI credentials in .env file")
    print("2. Add reference images to labeled_data/{characteristic}/ folders")
    print("3. Test: python adaptive_agent.py --source document.pdf --characteristic anchors")
    print("4. View: streamlit run feedback_interface.py")

if __name__ == "__main__":
    setup_system()