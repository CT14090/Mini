#!/usr/bin/env python3
# feedback_interface.py
"""
Streamlit Interface for Construction Document Analysis - FIXED VERSION
Simplified and reliable interface for viewing results
"""

import streamlit as st
import json
import os
import pathlib
from datetime import datetime
import pandas as pd
import base64
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Construction Document Agent - Results",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def load_extraction_files():
    """Load all extraction files (cached)"""
    feedback_dir = pathlib.Path("feedback_data")
    if not feedback_dir.exists():
        return []
    
    extraction_files = []
    for file in feedback_dir.glob("extraction_*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                data['filename'] = file.name
                extraction_files.append(data)
        except Exception as e:
            st.warning(f"Error loading {file.name}: {e}")
    
    # Sort by timestamp (newest first)
    extraction_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return extraction_files

@st.cache_data
def load_feedback_logs():
    """Load feedback analysis logs (cached)"""
    try:
        with open("feedback_log.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.warning(f"Error loading feedback log: {e}")
        return []

def display_extraction_overview(extractions):
    """Display overview of all extractions"""
    if not extractions:
        st.info("🚀 No extraction files found. Process documents first!")
        st.markdown("""
        **To get started:**
        
        ```bash
        # List available characteristics
        python adaptive_agent.py --list-characteristics
        
        # Process a document
        python adaptive_agent.py --source document.pdf --characteristic anchors
        
        # Or process all characteristics
        python adaptive_agent.py --source document.pdf --all-characteristics
        ```
        """)
        return []
    
    st.header("📊 Extraction Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(extractions))
    
    with col2:
        total_sections = sum(ex.get('total_sections', 0) for ex in extractions)
        st.metric("Total Items", total_sections)
    
    with col3:
        # Show characteristics processed
        characteristics = set()
        for ex in extractions:
            char = ex.get('target_characteristic', 'unknown')
            if char != 'unknown':
                characteristics.add(char)
        st.metric("Characteristics", len(characteristics))
    
    with col4:
        if extractions:
            avg_processing_time = sum(ex.get('processing_time', 0) for ex in extractions) / len(extractions)
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
    
    # Recent extractions table
    st.subheader("📄 Recent Extractions")
    
    table_data = []
    for extraction in extractions[:15]:  # Show last 15
        doc_path = extraction.get('document_path', 'Unknown')
        doc_name = os.path.basename(doc_path)
        
        timestamp = extraction.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%m-%d %H:%M')
            except:
                timestamp = timestamp[:16]
        
        # Get characteristic info
        target_char = extraction.get('target_characteristic', 'All')
        char_display = target_char.replace('_', ' ').title() if target_char != 'All' else 'All'
        
        table_data.append({
            'Document': doc_name[:30] + ('...' if len(doc_name) > 30 else ''),
            'Characteristic': char_display,
            'Time': timestamp,
            'Items': extraction.get('total_sections', 0),
            'ID': extraction.get('document_id', 'Unknown')[:8]
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    return extractions

def display_extraction_details(extraction):
    """Display detailed extraction results"""
    st.header(f"🔍 Extraction Details")
    
    # Document info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document Information")
        doc_path = extraction.get('document_path', 'Unknown')
        st.text(f"Document: {os.path.basename(doc_path)}")
        st.text(f"Document ID: {extraction.get('document_id', 'Unknown')}")
        st.text(f"Processing Time: {extraction.get('processing_time', 0):.1f}s")
        
        # Characteristic info
        target_char = extraction.get('target_characteristic')
        if target_char:
            st.text(f"Target: {target_char.replace('_', ' ').title()}")
    
    with col2:
        st.subheader("📊 Results Summary")
        summary = extraction.get('extraction_summary', {})
        st.text(f"Total Items: {summary.get('total_items', 0)}")
        st.text(f"Diagrams: {summary.get('diagram_items', 0)}")
        st.text(f"Tables: {summary.get('table_items', 0)}")
        st.text(f"Avg Confidence: {summary.get('avg_confidence', 0):.2f}")
    
    # Page distribution chart
    summary = extraction.get('extraction_summary', {})
    pages_with_content = summary.get('pages_with_content', [])
    
    if pages_with_content:
        st.subheader("📄 Content Distribution")
        
        # Count items per page
        sections = extraction.get('extracted_sections', [])
        page_counts = {}
        for section in sections:
            page = section.get('page', 0)
            page_counts[page] = page_counts.get(page, 0) + 1
        
        if page_counts:
            chart_data = pd.DataFrame([
                {'Page': f"Page {page}", 'Items': count} 
                for page, count in sorted(page_counts.items())
            ])
            st.bar_chart(chart_data.set_index('Page'))

def display_extracted_content(extraction):
    """Display extracted content with filtering"""
    st.header("📋 Extracted Content")
    
    sections = extraction.get('extracted_sections', [])
    if not sections:
        st.info("No content was extracted from this document.")
        
        target_char = extraction.get('target_characteristic', 'unknown')
        st.markdown(f"""
        **Possible reasons:**
        
        1. No {target_char.replace('_', ' ')} content found in document
        2. Quality thresholds too strict
        3. Need more training data in `labeled_data/{target_char}/`
        4. Document quality issues
        
        **Next steps:**
        ```bash
        # Add training examples
        python adaptive_agent.py --setup-labeled-data
        
        # Process with debug
        python adaptive_agent.py --source document.pdf --characteristic {target_char} --debug
        ```
        """)
        return
    
    # Simple filtering
    col1, col2 = st.columns(2)
    
    with col1:
        # Content type filter
        content_types = sorted(set(section.get('type', 'unknown') for section in sections))
        selected_types = st.multiselect(
            "Content Types",
            content_types,
            default=content_types
        )
    
    with col2:
        # Confidence filter
        min_confidence = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.0, 0.1
        )
    
    # Apply filters
    filtered_sections = [
        section for section in sections
        if (section.get('type', 'unknown') in selected_types and
            section.get('confidence', 0) >= min_confidence)
    ]
    
    st.text(f"Showing {len(filtered_sections)} of {len(sections)} items")
    
    # Display content
    for i, section in enumerate(filtered_sections[:10]):  # Limit display
        type_display = section.get('type', 'unknown').replace('_', ' ').title()
        confidence = section.get('confidence', 0)
        page = section.get('page', 'Unknown')
        
        with st.expander(
            f"📄 {type_display} (Page {page}, Confidence: {confidence:.2f})",
            expanded=(i < 2)  # Expand first 2
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Content
                content = section.get('content', 'No content available')
                if len(content) > 1000:
                    st.text(content[:1000] + "\n... (truncated)")
                    if st.button(f"Show full content {i+1}"):
                        st.text(content)
                else:
                    st.text(content)
                
                # Basic metadata
                metadata = section.get('region_metadata', {})
                if metadata:
                    method = metadata.get('extraction_method', 'Unknown')
                    area = metadata.get('area', 0)
                    st.caption(f"Method: {method}, Area: {area:,} pixels")
            
            with col2:
                # Image if available
                if 'data_uri' in section and section['data_uri']:
                    try:
                        header, encoded = section['data_uri'].split(',', 1)
                        decoded = base64.b64decode(encoded)
                        image = Image.open(io.BytesIO(decoded))
                        st.image(image, caption=f"{type_display}", width=200)
                    except Exception as e:
                        st.error(f"Image error: {e}")

def display_system_status():
    """Display system status"""
    st.header("🎯 System Status")
    
    # Check files
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Core Files")
        files_to_check = [
            "adaptive_agent.py",
            "characteristic_based_extractor.py", 
            "llm_feedback.py",
            "diagnostic.py"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                st.success(f"✅ {file_path}")
            else:
                st.error(f"❌ {file_path}")
    
    with col2:
        st.subheader("📚 Training Data")
        labeled_path = pathlib.Path("labeled_data")
        
        if labeled_path.exists():
            categories = [d for d in labeled_path.iterdir() if d.is_dir()]
            total_images = 0
            
            for cat_dir in categories:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                total_images += len(images)
                
                if len(images) > 0:
                    st.success(f"✅ {cat_dir.name}: {len(images)} images")
                else:
                    st.warning(f"⚠️ {cat_dir.name}: No images")
            
            st.metric("Total Training Images", total_images)
        else:
            st.error("❌ No labeled_data directory")
            st.info("Run: `python adaptive_agent.py --setup-labeled-data`")
    
    # Check configuration
    st.subheader("⚙️ Configuration")
    
    # Check Azure OpenAI
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if not missing_vars:
        st.success("✅ Azure OpenAI configured")
    else:
        st.warning(f"⚠️ Azure OpenAI missing: {missing_vars}")
    
    # Check parameters
    if os.path.exists("learning_parameters.json"):
        st.success("✅ Learning parameters file exists")
        try:
            with open("learning_parameters.json") as f:
                params = json.load(f)
            st.json(params)
        except Exception as e:
            st.error(f"Error loading parameters: {e}")
    else:
        st.info("ℹ️ No learning parameters file (will be created automatically)")

def display_feedback_logs():
    """Display feedback analysis logs"""
    st.header("🤖 Feedback Analysis Logs")
    
    logs = load_feedback_logs()
    
    if not logs:
        st.info("No feedback logs found.")
        st.markdown("""
        Feedback logs are created when Azure OpenAI analyzes extractions.
        
        **To generate feedback:**
        ```bash
        python llm_feedback.py --analyze-and-apply document_id
        ```
        """)
        return
    
    st.text(f"Total log entries: {len(logs)}")
    
    # Show recent logs
    for i, log in enumerate(logs[-5:]):  # Show last 5
        timestamp = log.get('timestamp', 'Unknown')
        doc_id = log.get('document_id', 'Unknown')
        method = log.get('analysis_method', 'Unknown')
        
        with st.expander(f"📄 {doc_id} - {timestamp[:16]}"):
            st.text(f"Method: {method}")
            
            # Show vision analysis if available
            if 'vision_analysis_summary' in log:
                summary = log['vision_analysis_summary']
                accuracy = summary.get('accuracy_rate', 0)
                analyzed = summary.get('total_items_analyzed', 0)
                correct = summary.get('correct_classifications', 0)
                
                st.metric("Accuracy", f"{accuracy:.1%}")
                st.text(f"Analyzed: {analyzed} items")
                st.text(f"Correct: {correct} items")
            
            # Show parameter recommendations
            if 'parameter_recommendations' in log:
                recommendations = log['parameter_recommendations']
                reasoning = recommendations.get('reasoning', 'No reasoning provided')
                adjustments = recommendations.get('adjustments', {})
                
                st.text(f"Reasoning: {reasoning}")
                if adjustments:
                    st.json(adjustments)

def main():
    st.title("🎯 Construction Document Analysis Results")
    st.markdown("View extraction results and system status")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a view",
        [
            "📊 Overview", 
            "🔍 Document Details", 
            "📚 System Status",
            "🤖 Feedback Logs"
        ]
    )
    
    # Load data
    extractions = load_extraction_files()
    
    if page == "📊 Overview":
        display_extraction_overview(extractions)
    
    elif page == "🔍 Document Details":
        if not extractions:
            st.info("No extraction files found.")
        else:
            # Document selector
            doc_options = []
            for extraction in extractions:
                doc_path = extraction.get('document_path', 'Unknown')
                doc_name = os.path.basename(doc_path)
                doc_id = extraction.get('document_id', 'Unknown')[:8]
                target_char = extraction.get('target_characteristic', 'All')
                char_display = target_char.replace('_', ' ').title() if target_char != 'All' else 'All'
                
                doc_options.append(f"{doc_name} - {char_display} ({doc_id})")
            
            selected_idx = st.selectbox(
                "Select Document",
                range(len(doc_options)),
                format_func=lambda x: doc_options[x]
            )
            
            if selected_idx is not None:
                extraction = extractions[selected_idx]
                display_extraction_details(extraction)
                st.divider()
                display_extracted_content(extraction)
    
    elif page == "📚 System Status":
        display_system_status()
    
    elif page == "🤖 Feedback Logs":
        display_feedback_logs()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Quick Commands:**
    ```bash
    # Process document
    python adaptive_agent.py --source doc.pdf --characteristic anchors
    
    # Run diagnostic
    python diagnostic.py
    
    # Test system
    python adaptive_agent.py --test-system
    ```
    """)

if __name__ == "__main__":
    main()