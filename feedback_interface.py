#!/usr/bin/env python3
"""
Complete Fixed Streamlit Interface for Window Characteristic Extractions
"""
import streamlit as st
import json
import os
import pathlib
import base64
import re
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Optional plotly import
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="Window Characteristic Analysis",
    page_icon="🪟",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_extracted_text(text: str) -> str:
    """Clean up garbled extracted text"""
    if not text:
        return ""
    
    # Remove GLYPH patterns
    text = re.sub(r'GLYPH<[^>]*>', '', text)
    
    # Remove font references
    text = re.sub(r'font=/[A-Z\-+]+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    
    # Clean up common OCR errors
    text = text.replace('GLYPH', ' ')
    text = text.replace('DVGLYPH', ' ')
    
    # Remove patterns like "c=3," or "c=17,"
    text = re.sub(r'c=\d+,', '', text)
    
    return text.strip()

def clean_table_content(content: str) -> str:
    """Clean up table content for better display"""
    if not content:
        return ""
    
    # Basic cleaning
    content = clean_extracted_text(content)
    
    # Preserve table structure but clean up
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('|--'):  # Keep content, remove separator lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def load_extraction_files() -> Dict[str, List[Dict]]:
    """Load all extraction files organized by characteristic"""
    extraction_files = {}
    feedback_dir = pathlib.Path("feedback_data")
    
    if not feedback_dir.exists():
        return extraction_files
    
    characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
    
    for characteristic in characteristics:
        char_files = []
        pattern = f"{characteristic}_extraction_*.json"
        
        for file_path in feedback_dir.glob(pattern):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    data['file_path'] = str(file_path)
                    data['file_name'] = file_path.name
                    char_files.append(data)
            except Exception as e:
                st.error(f"Error loading {file_path.name}: {str(e)[:100]}")
        
        char_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        extraction_files[characteristic] = char_files
    
    return extraction_files

def load_feedback_logs() -> Dict[str, List[Dict]]:
    """Load feedback logs for each characteristic"""
    feedback_logs = {}
    characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
    
    for characteristic in characteristics:
        log_file = f"feedback_log_{characteristic}.json"
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    feedback_logs[characteristic] = json.load(f)
            except Exception:
                feedback_logs[characteristic] = []
        else:
            feedback_logs[characteristic] = []
    
    return feedback_logs

def check_azure_openai_status() -> bool:
    """Check if Azure OpenAI is configured properly"""
    try:
        # Force reload environment variables
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except:
        pass
    
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    
    # Check if all variables exist and are not empty
    for var in required_vars:
        value = os.getenv(var)
        if not value or len(str(value).strip()) == 0:
            return False
    
    # Try to import and test connection
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version="2024-02-01",
            temperature=0.1,
            max_tokens=5
        )
        
        # Quick test (don't actually call it in Streamlit to avoid costs)
        return True
        
    except Exception:
        return False

def display_image_from_data_uri(data_uri: str, caption: str = "", width: int = 300):
    """Display image from data URI"""
    try:
        if data_uri and data_uri.startswith('data:image'):
            base64_data = data_uri.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            st.image(image_bytes, caption=caption, width=width)
        else:
            st.write("No image data available")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)[:50]}")

def display_all_images(items: List[Dict]):
    """Display all extracted images"""
    cols = st.columns(3)  # 3 images per row
    
    for i, item in enumerate(items):
        with cols[i % 3]:
            st.write(f"**Image {i+1}** - Page {item.get('page', 'Unknown')}")
            st.write(f"Confidence: {item.get('confidence', 0):.3f}")
            
            data_uri = item.get('data_uri')
            if data_uri:
                display_image_from_data_uri(data_uri, width=200)
            
            # Show metadata
            metadata = item.get('metadata', {})
            if metadata.get('reference_name'):
                st.write(f"Reference: {metadata.get('reference_name')}")
            if metadata.get('extraction_method'):
                st.write(f"Method: {metadata.get('extraction_method')}")
            
            st.write("---")

def display_all_summaries(items: List[Dict]):
    """Display all text summaries"""
    for i, item in enumerate(items, 1):
        with st.expander(f"Summary {i} - Page {item.get('page', 'Unknown')} (Confidence: {item.get('confidence', 0):.3f})"):
            content = item.get('content', '')
            if content:
                # Clean up garbled text
                cleaned_content = clean_extracted_text(content)
                if cleaned_content and len(cleaned_content.strip()) > 50:
                    st.write(cleaned_content)
                else:
                    st.warning("Text summary appears to be corrupted or too short")
                    st.text(content[:200] + "..." if len(content) > 200 else content)
            
            # Show metadata
            metadata = item.get('metadata', {})
            if metadata:
                st.write("**Extraction Details:**")
                st.write(f"- Keyword matches: {metadata.get('keyword_matches', 0)}")
                st.write(f"- Source pages: {metadata.get('source_pages', [])}")
                st.write(f"- Extraction method: {metadata.get('extraction_method', 'Unknown')}")

def display_all_tables(items: List[Dict]):
    """Display all extracted tables"""
    for i, item in enumerate(items, 1):
        with st.expander(f"Table {i} - Page {item.get('page', 'Unknown')} (Confidence: {item.get('confidence', 0):.3f})"):
            content = item.get('content', '')
            if content:
                # Clean table content
                cleaned_content = clean_table_content(content)
                st.text(cleaned_content)
            
            # Show metadata
            metadata = item.get('metadata', {})
            if metadata:
                st.write("**Table Analysis:**")
                st.write(f"- Relevance score: {metadata.get('relevance_score', 0):.1f}")
                
                table_analysis = metadata.get('table_analysis', {})
                if table_analysis:
                    data_points = table_analysis.get('data_points_found', [])
                    if data_points:
                        st.write(f"- Data points found: {', '.join(data_points)}")
                    st.write(f"- Table type: {table_analysis.get('table_type', 'Unknown')}")

def display_all_other_content(items: List[Dict]):
    """Display other content types"""
    for i, item in enumerate(items, 1):
        with st.expander(f"Item {i} - Page {item.get('page', 'Unknown')} (Confidence: {item.get('confidence', 0):.3f})"):
            content = item.get('content', '')
            if content:
                cleaned_content = clean_extracted_text(content)
                if cleaned_content:
                    st.write(cleaned_content)
                else:
                    st.text(content)
            
            metadata = item.get('metadata', {})
            if metadata:
                st.write("**Metadata:**")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)) and key != 'bbox':
                        st.write(f"- {key}: {value}")

def display_content_breakdown(sections: List[Dict]):
    """Display all extracted content organized by type"""
    # Group by type
    content_types = {}
    for section in sections:
        section_type = section.get('type', 'unknown')
        clean_type = section_type.replace('_', ' ').title()
        if clean_type not in content_types:
            content_types[clean_type] = []
        content_types[clean_type].append(section)
    
    # Show all content types
    for content_type, items in content_types.items():
        st.write(f"### {content_type} ({len(items)} items)")
        
        if 'image' in content_type.lower():
            display_all_images(items)
        elif 'summary' in content_type.lower():
            display_all_summaries(items)
        elif 'table' in content_type.lower():
            display_all_tables(items)
        else:
            display_all_other_content(items)
        
        st.write("---")

def display_extraction_summary(extraction_files: Dict[str, List[Dict]]):
    """Display extraction summary statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = sum(len(files) for files in extraction_files.values())
    total_items = 0
    avg_confidence = 0
    
    for char_files in extraction_files.values():
        for file_data in char_files:
            sections = file_data.get('extracted_sections', [])
            total_items += len(sections)
            if sections:
                avg_confidence += sum(item.get('confidence', 0) for item in sections) / len(sections)
    
    if total_docs > 0:
        avg_confidence /= total_docs
    
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Items", total_items)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col4:
        azure_status = "Connected" if check_azure_openai_status() else "Missing"
        st.metric("Azure OpenAI", azure_status)

def display_recent_extractions(extraction_files: Dict[str, List[Dict]]):
    """Display recent extractions in a clean table"""
    recent_data = []
    for characteristic, files in extraction_files.items():
        for file_data in files[:2]:  # Top 2 per characteristic
            recent_data.append({
                'Characteristic': characteristic.replace('_', ' ').title(),
                'Document': file_data.get('document_id', 'Unknown')[:8],
                'Items': file_data.get('total_sections', 0),
                'Date': file_data.get('timestamp', '')[:10],
                'Time': f"{file_data.get('processing_time', 0):.1f}s"
            })
    
    if recent_data:
        df = pd.DataFrame(recent_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No extractions found. Process some documents to get started.")

def display_characteristic_page(characteristic: str, extraction_files: List[Dict], feedback_logs: List[Dict]):
    """Display characteristic-specific page"""
    char_title = characteristic.replace('_', ' ').title()
    st.header(f"{char_title} Analysis")
    
    if not extraction_files:
        st.info(f"No {char_title.lower()} extractions found.")
        st.code(f"python adaptive_agent.py --source document.pdf --characteristics {characteristic}")
        return
    
    # Document selector
    doc_options = []
    for i, f in enumerate(extraction_files):
        doc_id = f.get('document_id', 'Unknown')[:8]
        path = os.path.basename(f.get('document_path', 'Unknown'))
        timestamp = f.get('timestamp', '')[:10]
        doc_options.append(f"{doc_id} - {path} ({timestamp})")
    
    selected_idx = st.selectbox(
        f"Select {char_title} Document:",
        range(len(doc_options)),
        format_func=lambda x: doc_options[x],
        key=f"doc_selector_{characteristic}"
    )
    
    if selected_idx is not None:
        selected_doc = extraction_files[selected_idx]
        display_document_analysis(characteristic, selected_doc, feedback_logs)

def display_document_analysis(characteristic: str, doc_data: Dict, feedback_logs: List[Dict]):
    """Display comprehensive document analysis"""
    doc_id = doc_data.get('document_id', 'Unknown')
    
    # Find feedback for this document
    doc_feedback = next((log for log in feedback_logs if log.get('document_id') == doc_id), None)
    
    # Document overview
    st.subheader("Document Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**ID:** {doc_id}")
        st.write(f"**Source:** {os.path.basename(doc_data.get('document_path', 'Unknown'))}")
        st.write(f"**Pages:** {doc_data.get('total_pages', 'Unknown')}")
    
    with col2:
        timestamp = doc_data.get('timestamp', '')
        if timestamp:
            formatted_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
            st.write(f"**Processed:** {formatted_time}")
        
        st.write(f"**Items Extracted:** {doc_data.get('total_sections', 0)}")
        processing_time = doc_data.get('processing_time', 0)
        if processing_time:
            st.write(f"**Processing Time:** {processing_time:.1f}s")
    
    # AI Feedback Analysis
    if doc_feedback:
        st.subheader("AI Analysis Results")
        display_ai_feedback(doc_feedback)
    
    # Content breakdown
    sections = doc_data.get('extracted_sections', [])
    if sections:
        st.subheader("Extracted Content")
        display_content_breakdown(sections)
    
    # Text summary
    text_summary = doc_data.get('text_summary', {})
    if text_summary.get('summary'):
        st.subheader("Text Summary")
        
        summary_text = text_summary['summary']
        cleaned_summary = clean_extracted_text(summary_text)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.metric("Confidence", f"{text_summary.get('confidence', 0):.2f}")
            st.write(f"**Source Pages:** {text_summary.get('source_pages', [])}")
            st.write(f"**Sentences:** {text_summary.get('sentence_count', 0)}")
            
            # Additional metadata
            if 'total_sentences_analyzed' in text_summary:
                st.write(f"**Total Analyzed:** {text_summary.get('total_sentences_analyzed', 0)}")
            if 'relevant_sentences_found' in text_summary:
                st.write(f"**Relevant Found:** {text_summary.get('relevant_sentences_found', 0)}")
        
        with col1:
            if cleaned_summary and len(cleaned_summary.strip()) > 50:
                # Split into sentences for better readability
                sentences = cleaned_summary.split('. ')
                for i, sentence in enumerate(sentences[:5], 1):  # Show first 5 sentences clearly
                    if sentence.strip():
                        st.write(f"**{i}.** {sentence.strip()}.")
                
                if len(sentences) > 5:
                    with st.expander(f"Show remaining {len(sentences) - 5} sentences"):
                        for i, sentence in enumerate(sentences[5:], 6):
                            if sentence.strip():
                                st.write(f"**{i}.** {sentence.strip()}.")
            else:
                st.warning("Text summary appears to contain extraction errors")
                st.text(summary_text[:300] + "..." if len(summary_text) > 300 else summary_text)

def display_ai_feedback(feedback: Dict):
    """Display AI feedback in a clean format"""
    quality_scores = feedback.get('quality_scores', {})
    
    if quality_scores:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Content Relevance", f"{quality_scores.get('content_relevance', 0)}/5")
        with col2:
            st.metric("Extraction Quality", f"{quality_scores.get('extraction_quality', 0)}/5")
        with col3:
            st.metric("Reference Alignment", f"{quality_scores.get('reference_alignment', 0)}/5")
    
    # Parameter changes
    param_changes = feedback.get('parameter_changes', [])
    if param_changes:
        st.write("**Parameter Updates Applied:**")
        for change in param_changes[:3]:  # Show top 3
            st.write(f"• {change}")
    
    # AI reasoning (condensed)
    reasoning = feedback.get('reasoning', '')
    if reasoning and len(reasoning) > 0:
        with st.expander("AI Analysis Details"):
            st.write(reasoning[:300] + "..." if len(reasoning) > 300 else reasoning)

def display_analytics_page(extraction_files: Dict[str, List[Dict]], feedback_logs: Dict[str, List[Dict]]):
    """Display analytics and trends"""
    st.header("Analytics Dashboard")
    
    # Performance summary
    st.subheader("Performance Summary")
    perf_data = []
    for characteristic, files in extraction_files.items():
        if files:
            total_items = sum(len(f.get('extracted_sections', [])) for f in files)
            total_docs = len(files)
            avg_items = total_items / total_docs if total_docs > 0 else 0
            
            perf_data.append({
                'Characteristic': characteristic.replace('_', ' ').title(),
                'Documents': total_docs,
                'Total Items': total_items,
                'Avg Items/Doc': f"{avg_items:.1f}"
            })
    
    if perf_data:
        df = pd.DataFrame(perf_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(df, x='Characteristic', y='Total Items', 
                             title='Items Extracted by Characteristic')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.pie(df, values='Documents', names='Characteristic',
                             title='Documents by Characteristic')
                st.plotly_chart(fig2, use_container_width=True)
    
    # Recent parameter changes
    st.subheader("Recent Parameter Changes")
    all_changes = []
    for characteristic, logs in feedback_logs.items():
        for log in logs[-3:]:  # Last 3 per characteristic
            changes = log.get('parameter_changes', [])
            for change in changes:
                all_changes.append({
                    'Characteristic': characteristic.replace('_', ' ').title(),
                    'Date': log.get('timestamp', '')[:10],
                    'Change': change
                })
    
    if all_changes:
        changes_df = pd.DataFrame(all_changes)
        st.dataframe(changes_df.tail(8), use_container_width=True, hide_index=True)
    else:
        st.info("No parameter changes recorded yet.")

def main():
    """Main Streamlit application"""
    st.title("Window Characteristic Analysis")
    st.markdown("AI-powered document analysis for window construction specifications")
    
    # Load data
    try:
        extraction_files = load_extraction_files()
        feedback_logs = load_feedback_logs()
    except Exception as e:
        st.error(f"Error loading data: {str(e)[:100]}")
        return
    
    # Check for data
    total_files = sum(len(files) for files in extraction_files.values())
    
    if total_files == 0:
        st.warning("No extraction data found!")
        
        st.subheader("Getting Started")
        st.write("1. **Setup reference data:**")
        st.code("python adaptive_agent.py --setup-reference-data")
        
        st.write("2. **Process a document:**")
        st.code("python adaptive_agent.py --source document.pdf")
        
        st.write("3. **Test Azure OpenAI (optional):**")
        st.code("python llm_feedback.py --test-connection")
        
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page_options = ["Overview", "Anchors", "Glazing", "Impact Rating", "Design Pressure", "Analytics"]
    selected_page = st.sidebar.selectbox(
        "Select Page:",
        page_options,
        key="main_page_selector"
    )
    
    # System status in sidebar
    st.sidebar.subheader("System Status")
    azure_connected = check_azure_openai_status()
    st.sidebar.write(f"Azure OpenAI: {'✅' if azure_connected else '❌'}")
    st.sidebar.write(f"Documents: {total_files}")
    
    total_feedback = sum(len(logs) for logs in feedback_logs.values())
    st.sidebar.write(f"AI Feedback: {total_feedback}")
    
    # Reference data status
    labeled_data_path = pathlib.Path("labeled_data")
    if labeled_data_path.exists():
        st.sidebar.subheader("Reference Data")
        for char in ['anchors', 'glazing', 'impact_rating', 'design_pressure']:
            char_path = labeled_data_path / char
            if char_path.exists():
                image_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_count += len(list(char_path.glob(ext)))
                st.sidebar.write(f"{char}: {image_count} images")
    
    # Main content
    if selected_page == "Overview":
        st.header("System Overview")
        display_extraction_summary(extraction_files)
        
        st.subheader("Recent Extractions")
        display_recent_extractions(extraction_files)
    
    elif selected_page == "Anchors":
        display_characteristic_page("anchors", extraction_files.get("anchors", []), feedback_logs.get("anchors", []))
    
    elif selected_page == "Glazing":
        display_characteristic_page("glazing", extraction_files.get("glazing", []), feedback_logs.get("glazing", []))
    
    elif selected_page == "Impact Rating":
        display_characteristic_page("impact_rating", extraction_files.get("impact_rating", []), feedback_logs.get("impact_rating", []))
    
    elif selected_page == "Design Pressure":
        display_characteristic_page("design_pressure", extraction_files.get("design_pressure", []), feedback_logs.get("design_pressure", []))
    
    elif selected_page == "Analytics":
        display_analytics_page(extraction_files, feedback_logs)
    
    # Quick actions in sidebar
    st.sidebar.subheader("Quick Actions")
    if st.sidebar.button("Refresh Data", key="refresh_button"):
        st.rerun()

if __name__ == "__main__":
    main()