#!/usr/bin/env python3
"""
Specialized Window Characteristics Feedback Interface
Views extraction results organized by characteristic type
"""
import streamlit as st
import json
import os
import pathlib
from datetime import datetime
from typing import Dict, List, Optional
import base64

def load_characteristic_extractions() -> Dict[str, List[Dict]]:
    """Load all characteristic extraction files"""
    extractions = {
        'anchors': [],
        'glazing': [],
        'impact_rating': [],
        'design_pressure': []
    }
    
    feedback_dir = pathlib.Path("feedback_data")
    if not feedback_dir.exists():
        return extractions
    
    # Load all extraction files
    for file_path in feedback_dir.glob("*.json"):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Determine characteristic from filename or data
            characteristic = None
            filename = file_path.stem
            
            if filename.startswith('anchors_'):
                characteristic = 'anchors'
            elif filename.startswith('glazing_'):
                characteristic = 'glazing'  
            elif filename.startswith('impact_rating_'):
                characteristic = 'impact_rating'
            elif filename.startswith('design_pressure_'):
                characteristic = 'design_pressure'
            elif 'characteristic_focus' in data:
                characteristic = data['characteristic_focus']
            
            if characteristic and characteristic in extractions:
                extractions[characteristic].append(data)
                
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    
    # Sort by timestamp (newest first)
    for char_type in extractions:
        extractions[char_type].sort(
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
    
    return extractions

def load_feedback_logs() -> Dict[str, List[Dict]]:
    """Load characteristic-specific feedback logs"""
    logs = {
        'anchors': [],
        'glazing': [],
        'impact_rating': [],
        'design_pressure': []
    }
    
    for characteristic in logs.keys():
        log_file = f"feedback_log_{characteristic}.json"
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    logs[characteristic] = json.load(f)
            except Exception as e:
                st.error(f"Error loading {log_file}: {e}")
    
    return logs

def display_characteristic_overview(extractions: Dict[str, List[Dict]]):
    """Display overview of all characteristics"""
    st.header("🎯 Window Characteristics Overview")
    
    # Summary metrics
    total_docs = sum(len(docs) for docs in extractions.values())
    
    if total_docs == 0:
        st.warning("📭 No extractions found. Run some characteristic extractions first!")
        st.code("""
# Example commands:
python adaptive_agent.py --source document.pdf --characteristic anchors
python adaptive_agent.py --source document.pdf --characteristic glazing
python adaptive_agent.py --source document.pdf --characteristic impact_rating
python adaptive_agent.py --source document.pdf --characteristic design_pressure
        """)
        return
    
    # Characteristic cards
    cols = st.columns(4)
    
    characteristics_info = {
        'anchors': {'icon': '🔩', 'name': 'Anchor Types', 'color': '#FF6B6B'},
        'glazing': {'icon': '🪟', 'name': 'Glazing Specs', 'color': '#4ECDC4'},
        'impact_rating': {'icon': '🌪️', 'name': 'Impact Rating', 'color': '#45B7D1'},
        'design_pressure': {'icon': '📊', 'name': 'Design Pressure', 'color': '#96CEB4'}
    }
    
    for i, (char_type, info) in enumerate(characteristics_info.items()):
        with cols[i]:
            doc_count = len(extractions[char_type])
            total_items = sum(len(doc.get('extracted_sections', [])) for doc in extractions[char_type])
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
                border: 1px solid {info['color']}44;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin-bottom: 10px;
            ">
                <div style="font-size: 2rem; margin-bottom: 10px;">{info['icon']}</div>
                <div style="font-weight: bold; color: {info['color']}; margin-bottom: 5px;">
                    {info['name']}
                </div>
                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;">
                    {doc_count} Documents
                </div>
                <div style="color: #666; font-size: 0.9rem;">
                    {total_items} Items Extracted
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("📅 Recent Extractions")
    
    # Combine all recent extractions
    all_recent = []
    for char_type, docs in extractions.items():
        for doc in docs[:3]:  # Top 3 per characteristic
            doc['characteristic_type'] = char_type
            all_recent.append(doc)
    
    # Sort by timestamp
    all_recent.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for doc in all_recent[:8]:  # Show top 8 recent
        char_type = doc['characteristic_type']
        info = characteristics_info[char_type]
        
        timestamp = datetime.fromisoformat(doc.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
        doc_name = os.path.basename(doc.get('document_path', 'Unknown'))
        item_count = len(doc.get('extracted_sections', []))
        
        with st.expander(f"{info['icon']} {info['name']} - {doc_name} ({timestamp})"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Document**: {doc_name}")
                st.write(f"**Focus**: {char_type.replace('_', ' ').title()}")
            
            with col2:
                st.metric("Items Extracted", item_count)
            
            with col3:
                processing_time = doc.get('processing_time', 0)
                st.metric("Processing Time", f"{processing_time:.1f}s")

def display_characteristic_details(char_type: str, extractions: List[Dict], logs: List[Dict]):
    """Display detailed view of specific characteristic"""
    info_map = {
        'anchors': {'icon': '🔩', 'name': 'Anchor Types', 'color': '#FF6B6B'},
        'glazing': {'icon': '🪟', 'name': 'Glazing Specifications', 'color': '#4ECDC4'},
        'impact_rating': {'icon': '🌪️', 'name': 'Impact Rating', 'color': '#45B7D1'},
        'design_pressure': {'icon': '📊', 'name': 'Design Pressure', 'color': '#96CEB4'}
    }
    
    info = info_map[char_type]
    
    st.header(f"{info['icon']} {info['name']} Extractions")
    
    if not extractions:
        st.warning(f"No {char_type} extractions found.")
        st.code(f"python adaptive_agent.py --source document.pdf --characteristic {char_type}")
        return
    
    # Summary stats
    total_items = sum(len(doc.get('extracted_sections', [])) for doc in extractions)
    avg_confidence = 0
    confidence_count = 0
    
    for doc in extractions:
        for section in doc.get('extracted_sections', []):
            if 'confidence' in section:
                avg_confidence += section['confidence']
                confidence_count += 1
    
    if confidence_count > 0:
        avg_confidence /= confidence_count
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(extractions))
    with col2:
        st.metric("Total Items", total_items)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col4:
        recent_feedback = len([log for log in logs if 
                              (datetime.now() - datetime.fromisoformat(log.get('timestamp', datetime.now().isoformat()))).days < 7])
        st.metric("Recent Feedback", recent_feedback)
    
    # Document selector
    st.subheader("📄 Select Document")
    doc_options = []
    for i, doc in enumerate(extractions):
        doc_name = os.path.basename(doc.get('document_path', f'Document {i+1}'))
        timestamp = datetime.fromisoformat(doc.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
        item_count = len(doc.get('extracted_sections', []))
        doc_options.append(f"{doc_name} | {timestamp} | {item_count} items")
    
    selected_idx = st.selectbox("Choose document:", range(len(doc_options)), 
                               format_func=lambda x: doc_options[x])
    
    if selected_idx is not None:
        selected_doc = extractions[selected_idx]
        display_document_extractions(selected_doc, char_type, info)
    
    # Feedback analysis
    if logs:
        st.subheader("🤖 LLM Feedback Analysis")
        display_feedback_analysis(logs, char_type)

def display_document_extractions(doc: Dict, char_type: str, info: Dict):
    """Display extractions from a specific document"""
    sections = doc.get('extracted_sections', [])
    
    if not sections:
        st.warning("No extractions found in this document.")
        return
    
    st.subheader(f"📋 Extracted {info['name']}")
    
    # Filter and group by type
    images = [s for s in sections if 'image' in s.get('type', '')]
    tables = [s for s in sections if 'table' in s.get('type', '')]
    text_sections = [s for s in sections if 'text' in s.get('type', '')]
    
    # Display images
    if images:
        st.markdown(f"### 🖼️ {info['name']} Images ({len(images)})")
        
        # Image grid
        cols_per_row = 2
        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(images):
                    with cols[j]:
                        img_section = images[idx]
                        display_image_section(img_section, char_type)
    
    # Display tables
    if tables:
        st.markdown(f"### 📊 {info['name']} Tables ({len(tables)})")
        for table_section in tables:
            display_table_section(table_section, char_type)
    
    # Display text
    if text_sections:
        st.markdown(f"### 📝 {info['name']} Text ({len(text_sections)})")
        for text_section in text_sections:
            display_text_section(text_section, char_type)

def display_image_section(section: Dict, char_type: str):
    """Display an image section with metadata"""
    confidence = section.get('confidence', 0)
    page = section.get('page', 'Unknown')
    
    # Confidence color
    if confidence >= 0.7:
        conf_color = "green"
    elif confidence >= 0.5:
        conf_color = "orange"
    else:
        conf_color = "red"
    
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <strong>Page {page}</strong>
            <span style="color: {conf_color}; font-weight: bold;">
                Confidence: {confidence:.2f}
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Display image
    data_uri = section.get('data_uri', '')
    if data_uri:
        st.image(data_uri, use_column_width=True)
    
    # Metadata
    metadata = section.get('metadata', {})
    if metadata:
        st.caption(f"Size: {metadata.get('width', '?')}x{metadata.get('height', '?')}px")
    
    # CV Analysis
    cv_analysis = section.get('cv_analysis', {})
    if cv_analysis.get('features_detected'):
        st.caption(f"CV Features: {', '.join(cv_analysis['features_detected'])}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_table_section(section: Dict, char_type: str):
    """Display a table section with analysis"""
    confidence = section.get('confidence', 0)
    page = section.get('page', 'Unknown')
    
    with st.expander(f"Table from Page {page} (Confidence: {confidence:.2f})", expanded=True):
        # Table content
        content = section.get('content', '')
        if content:
            st.markdown(content)
        
        # Table analysis
        table_analysis = section.get('table_analysis', {})
        if table_analysis:
            data_points = table_analysis.get('data_points_found', [])
            char_data = table_analysis.get('characteristic_specific_data', {})
            
            if data_points:
                st.caption(f"**Found**: {', '.join(data_points)}")
            
            if char_data:
                for key, value in char_data.items():
                    st.caption(f"**{key.replace('_', ' ').title()}**: {', '.join(map(str, value[:5]))}")

def display_text_section(section: Dict, char_type: str):
    """Display a text section with mentions"""
    confidence = section.get('confidence', 0)
    page = section.get('page', 'Unknown')
    
    with st.expander(f"Text from Page {page} (Confidence: {confidence:.2f})"):
        content = section.get('content', '')
        if content:
            st.text_area("Content", content, height=150, disabled=True)
        
        # Characteristic mentions
        mentions = section.get('characteristic_mentions', [])
        if mentions:
            st.caption("**Key Mentions:**")
            for mention in mentions[:3]:
                st.caption(f"• {mention}")

def display_feedback_analysis(logs: List[Dict], char_type: str):
    """Display feedback analysis for characteristic"""
    if not logs:
        st.info(f"No feedback logs found for {char_type}")
        return
    
    # Recent feedback summary
    recent_logs = logs[-5:]  # Last 5 feedback entries
    
    # Quality trends
    qualities = []
    timestamps = []
    
    for log in recent_logs:
        quality_scores = log.get('quality_scores', {})
        avg_quality = quality_scores.get('average', 0)
        if avg_quality > 0:
            qualities.append(avg_quality)
            timestamps.append(datetime.fromisoformat(log.get('timestamp', '')))
    
    if qualities:
        st.line_chart(dict(zip(timestamps, qualities)))
    
    # Latest feedback details
    if recent_logs:
        latest_log = recent_logs[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Latest Quality Scores")
            quality_scores = latest_log.get('quality_scores', {})
            
            for metric, score in quality_scores.items():
                if metric != 'average':
                    st.metric(metric.title(), f"{score}/5")
        
        with col2:
            st.markdown("#### 🔧 Recent Parameter Changes")
            changes = latest_log.get('parameter_changes', [])
            
            if changes:
                for change in changes[-3:]:  # Show last 3 changes
                    st.caption(f"• {change}")
            else:
                st.caption("No recent parameter changes")
        
        # LLM reasoning
        reasoning = latest_log.get('llm_reasoning', '')
        if reasoning:
            st.markdown("#### 💡 LLM Reasoning")
            st.info(reasoning)

def main():
    st.set_page_config(
        page_title="Window Characteristics Analyzer", 
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Window Characteristics Extraction Analyzer")
    st.markdown("*AI-powered extraction and analysis of specific window characteristics*")
    
    # Load data
    extractions = load_characteristic_extractions()
    feedback_logs = load_feedback_logs()
    
    # Sidebar navigation
    st.sidebar.title("🏗️ Navigation")
    
    view_options = ["Overview", "Anchors 🔩", "Glazing 🪟", "Impact Rating 🌪️", "Design Pressure 📊"]
    selected_view = st.sidebar.selectbox("Select View:", view_options)
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 System Status")
    
    total_docs = sum(len(docs) for docs in extractions.values())
    total_feedback = sum(len(logs) for logs in feedback_logs.values())
    
    st.sidebar.metric("Total Documents", total_docs)
    st.sidebar.metric("Feedback Entries", total_feedback)
    
    # Azure status
    azure_configured = all([
        os.getenv('AZURE_OPENAI_ENDPOINT'),
        os.getenv('AZURE_OPENAI_API_KEY'),
        os.getenv('AZURE_OPENAI_DEPLOYMENT')
    ])
    
    status_color = "🟢" if azure_configured else "🔴"
    st.sidebar.markdown(f"{status_color} Azure OpenAI: {'Configured' if azure_configured else 'Missing'}")
    
    # Main content
    if selected_view == "Overview":
        display_characteristic_overview(extractions)
    
    elif selected_view == "Anchors 🔩":
        display_characteristic_details('anchors', extractions['anchors'], feedback_logs['anchors'])
    
    elif selected_view == "Glazing 🪟":
        display_characteristic_details('glazing', extractions['glazing'], feedback_logs['glazing'])
    
    elif selected_view == "Impact Rating 🌪️":
        display_characteristic_details('impact_rating', extractions['impact_rating'], feedback_logs['impact_rating'])
    
    elif selected_view == "Design Pressure 📊":
        display_characteristic_details('design_pressure', extractions['design_pressure'], feedback_logs['design_pressure'])
    
    # Footer
    st.markdown("---")
    st.markdown("#### 🚀 Quick Commands")
    
    commands = [
        "python adaptive_agent.py --source doc.pdf --characteristic anchors",
        "python adaptive_agent.py --source doc.pdf --characteristic glazing", 
        "python adaptive_agent.py --source doc.pdf --characteristic impact_rating",
        "python adaptive_agent.py --source doc.pdf --characteristic design_pressure"
    ]
    
    for cmd in commands:
        st.code(cmd)

if __name__ == "__main__":
    main()