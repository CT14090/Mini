#!/usr/bin/env python3
# feedback_interface.py
"""
Complete Streamlit Interface for Azure OpenAI-First Construction Document Analysis
Enhanced interface with full pagination, filtering, and detailed view capabilities
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
import math

# Page config
st.set_page_config(
    page_title="Azure Construction Document Analyzer - Results",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_extraction_files():
    """Load all extraction files with error handling"""
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
            st.error(f"Error loading {file.name}: {e}")
    
    # Sort by timestamp (newest first)
    extraction_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return extraction_files

@st.cache_data
def load_feedback_logs():
    """Load feedback analysis logs"""
    try:
        with open("feedback_log.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading feedback log: {e}")
        return []

def display_extraction_overview(extractions):
    """Display comprehensive overview of all extractions"""
    if not extractions:
        st.info("No extraction files found. Process documents first!")
        st.markdown("""
        **To get started with Azure OpenAI-powered extraction:**
        
        ```bash
        # Test Azure connection
        python test_azure_connection.py
        
        # Process a document
        python adaptive_agent.py --source document.pdf --characteristic anchors
        
        # Process all characteristics
        python adaptive_agent.py --source document.pdf --all-characteristics
        ```
        """)
        return []
    
    st.header("Azure OpenAI Extraction Overview")
    
    # Enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Extractions", len(extractions))
    
    with col2:
        total_sections = sum(ex.get('total_sections', 0) for ex in extractions)
        st.metric("Total Items Found", total_sections)
    
    with col3:
        # Count Azure vs fallback extractions
        azure_extractions = sum(1 for ex in extractions if 'azure' in ex.get('processing_method', '').lower())
        st.metric("Azure Extractions", f"{azure_extractions}/{len(extractions)}")
    
    with col4:
        if extractions:
            avg_processing_time = sum(ex.get('processing_time', 0) for ex in extractions) / len(extractions)
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
    
    # Recent extractions table with enhanced info
    st.subheader("Recent Extractions")
    
    table_data = []
    for extraction in extractions[:20]:  # Show last 20
        doc_path = extraction.get('document_path', 'Unknown')
        doc_name = os.path.basename(doc_path)
        
        # Format timestamp
        timestamp = extraction.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%m-%d %H:%M')
            except:
                timestamp = timestamp[:16]
        
        # Get characteristic and method info
        target_char = extraction.get('target_characteristic', 'All')
        char_display = target_char.replace('_', ' ').title()
        
        processing_method = extraction.get('processing_method', 'unknown')
        method_icon = "🤖" if 'azure' in processing_method.lower() else "🔧"
        
        # Get Azure API usage
        azure_calls = extraction.get('extraction_summary', {}).get('azure_api_calls', 0)
        
        table_data.append({
            'Document': doc_name[:25] + ('...' if len(doc_name) > 25 else ''),
            'Characteristic': char_display,
            'Method': f"{method_icon} {'Azure' if azure_calls > 0 else 'Fallback'}",
            'Items': extraction.get('total_sections', 0),
            'API Calls': azure_calls,
            'Time': timestamp,
            'ID': extraction.get('document_id', 'Unknown')[:8]
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    return extractions

def display_extraction_details(extraction):
    """Display comprehensive extraction details"""
    st.header("Extraction Analysis")
    
    # Document information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Information")
        doc_path = extraction.get('document_path', 'Unknown')
        doc_name = os.path.basename(doc_path)
        
        st.text(f"Document: {doc_name}")
        st.text(f"Document ID: {extraction.get('document_id', 'Unknown')}")
        st.text(f"Processing Time: {extraction.get('processing_time', 0):.1f}s")
        
        target_char = extraction.get('target_characteristic')
        if target_char:
            st.text(f"Target: {target_char.replace('_', ' ').title()}")
            
        # Processing method details
        processing_method = extraction.get('processing_method', 'unknown')
        st.text(f"Method: {processing_method.replace('_', ' ').title()}")
    
    with col2:
        st.subheader("Extraction Results")
        summary = extraction.get('extraction_summary', {})
        
        st.text(f"Total Items: {summary.get('total_items', 0)}")
        st.text(f"Pages Processed: {summary.get('pages_processed', 0)}")
        st.text(f"Azure API Calls: {summary.get('azure_api_calls', 0)}")
        st.text(f"Average Confidence: {summary.get('avg_confidence', 0):.3f}")
        
        # Show extraction rate
        pages = summary.get('pages_processed', 1)
        items = summary.get('total_items', 0)
        if pages > 0:
            st.text(f"Extraction Rate: {items/pages:.1f} items/page")
    
    # Azure-specific metadata
    azure_metadata = extraction.get('azure_vision_metadata', {})
    if azure_metadata:
        st.subheader("Azure Vision Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text(f"Azure Enabled: {azure_metadata.get('azure_openai_enabled', False)}")
            st.text(f"Training Data Used: {azure_metadata.get('training_data_integrated', False)}")
            
        with col2:
            st.text(f"Vision Method: {azure_metadata.get('vision_analysis_method', 'Unknown')}")
            st.text(f"Cost Optimized: {azure_metadata.get('cost_optimized', False)}")
    
    # Page distribution visualization
    sections = extraction.get('extracted_sections', [])
    if sections:
        st.subheader("Content Distribution by Page")
        
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
    """Display extracted content with full pagination and filtering"""
    st.header("Extracted Content Browser")
    
    sections = extraction.get('extracted_sections', [])
    if not sections:
        st.warning("No content was extracted from this document.")
        
        # Show helpful troubleshooting info
        target_char = extraction.get('target_characteristic', 'unknown')
        azure_calls = extraction.get('extraction_summary', {}).get('azure_api_calls', 0)
        
        if azure_calls > 0:
            st.markdown(f"""
            **Azure OpenAI was used but found no {target_char.replace('_', ' ')} content.**
            
            Possible reasons:
            1. Document doesn't contain {target_char.replace('_', ' ')} information
            2. Training data doesn't match document content style
            3. System prompts need adjustment (automatic after 2-3 runs)
            
            **Next steps:**
            ```bash
            # Check training data alignment
            ls -la labeled_data/{target_char}/
            
            # Try with different characteristic
            python adaptive_agent.py --source document.pdf --characteristic design_pressure
            
            # System will auto-adjust - try running again
            ```
            """)
        else:
            st.markdown(f"""
            **Fallback method was used (Azure unavailable).**
            
            **To enable Azure OpenAI:**
            ```bash
            # Test Azure connection
            python test_azure_connection.py
            
            # Configure credentials in .env file
            ```
            """)
        return
    
    # Advanced filtering controls
    st.subheader("Filters and Display Options")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Content type filter
        content_types = sorted(set(section.get('type', 'unknown') for section in sections))
        selected_types = st.multiselect(
            "Content Types",
            content_types,
            default=content_types,
            help="Filter by characteristic type"
        )
    
    with col2:
        # Confidence filter
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Show only items above this confidence"
        )
    
    with col3:
        # Page filter
        pages = sorted(set(section.get('page', 0) for section in sections))
        if len(pages) > 1:
            selected_pages = st.multiselect(
                "Pages",
                pages,
                default=pages,
                help="Filter by document page"
            )
        else:
            selected_pages = pages
    
    with col4:
        # Items per page
        items_per_page_options = [10, 25, 50, 100, "All"]
        items_per_page = st.selectbox(
            "Items per page",
            items_per_page_options,
            index=4,  # Default to "All"
            help="Number of items to display"
        )
    
    # Apply filters
    filtered_sections = [
        section for section in sections
        if (section.get('type', 'unknown') in selected_types and
            section.get('confidence', 0) >= min_confidence and
            section.get('page', 0) in selected_pages)
    ]
    
    # Display filter results
    st.info(f"Showing {len(filtered_sections)} of {len(sections)} total items")
    
    if not filtered_sections:
        st.warning("No items match the current filters. Try adjusting the criteria.")
        return
    
    # Pagination setup
    if items_per_page == "All":
        display_sections = filtered_sections
        current_page = 1
        total_pages = 1
    else:
        items_per_page = int(items_per_page)
        total_pages = math.ceil(len(filtered_sections) / items_per_page)
        
        # Page navigation
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.get('current_page', 1) <= 1):
                    st.session_state.current_page = st.session_state.get('current_page', 1) - 1
                    st.rerun()
            
            with col2:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.get('current_page', 1),
                    key='page_input'
                )
                st.session_state.current_page = current_page
                st.caption(f"of {total_pages} pages")
            
            with col3:
                if st.button("Next ➡️", disabled=current_page >= total_pages):
                    st.session_state.current_page = current_page + 1
                    st.rerun()
        else:
            current_page = 1
        
        # Calculate displayed items
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_sections))
        display_sections = filtered_sections[start_idx:end_idx]
    
    # Display content items
    st.subheader(f"Content Items ({len(display_sections)} shown)")
    
    for i, section in enumerate(display_sections):
        type_display = section.get('type', 'unknown').replace('_', ' ').title()
        confidence = section.get('confidence', 0)
        page = section.get('page', 'Unknown')
        
        # Calculate global item number
        if items_per_page == "All":
            item_number = i + 1
        else:
            item_number = (current_page - 1) * items_per_page + i + 1
        
        # Color code by confidence
        if confidence >= 0.8:
            confidence_color = "🟢"
        elif confidence >= 0.6:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        with st.expander(
            f"{confidence_color} Item {item_number}: {type_display} (Page {page}, Confidence: {confidence:.2f})",
            expanded=(i < 2 and items_per_page != "All")  # Expand first 2 if paginated
        ):
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Content description
                content = section.get('content', 'No description available')
                st.markdown("**Description:**")
                
                if len(content) > 800:
                    st.text(content[:800] + "...")
                    if st.button(f"Show full description", key=f"expand_desc_{item_number}"):
                        st.text(content)
                else:
                    st.text(content)
                
                # Metadata section
                metadata = section.get('region_metadata', {})
                if metadata:
                    st.markdown("**Technical Details:**")
                    
                    detection_method = metadata.get('detection_method', 'Unknown')
                    extraction_method = metadata.get('extraction_method', 'Unknown')
                    area = metadata.get('area', 0)
                    
                    # Create metadata table
                    meta_data = {
                        'Detection Method': detection_method.replace('_', ' ').title(),
                        'Extraction Method': extraction_method.replace('_', ' ').title(),
                        'Region Area': f"{area:,} pixels" if area > 0 else "Unknown",
                        'Bounding Box': str(section.get('bbox', 'Not available'))
                    }
                    
                    meta_df = pd.DataFrame(list(meta_data.items()), columns=['Property', 'Value'])
                    st.dataframe(meta_df, hide_index=True)
                    
                    # Azure-specific information
                    azure_description = metadata.get('azure_description', '')
                    azure_reasoning = metadata.get('azure_reasoning', '')
                    
                    if azure_description:
                        st.markdown("**Azure Analysis:**")
                        st.info(f"Description: {azure_description}")
                    
                    if azure_reasoning:
                        st.success(f"Reasoning: {azure_reasoning}")
            
            with col2:
                # Image display
                st.markdown("**Extracted Image:**")
                
                if 'data_uri' in section and section['data_uri']:
                    try:
                        # Decode and display image
                        header, encoded = section['data_uri'].split(',', 1)
                        decoded = base64.b64decode(encoded)
                        image = Image.open(io.BytesIO(decoded))
                        
                        st.image(
                            image, 
                            caption=f"{type_display} - Page {page}",
                            use_column_width=True
                        )
                        
                        # Image info
                        st.caption(f"Image: {image.size[0]}×{image.size[1]} pixels")
                        
                    except Exception as e:
                        st.error(f"Image display error: {e}")
                        st.text("Image data corrupted or invalid format")
                else:
                    st.warning("No image data available")
                    st.text("This item was extracted without visual data")

def display_system_status():
    """Display comprehensive system status"""
    st.header("System Status & Configuration")
    
    # Core system files
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Core System Files")
        core_files = {
            "adaptive_agent.py": "Main processing engine",
            "characteristic_based_extractor.py": "Azure-first extraction logic",
            "llm_feedback.py": "Feedback and parameter tuning",
            "feedback_interface.py": "This interface",
            "diagnostic.py": "System diagnostics",
            "test_azure_connection.py": "Azure connection testing"
        }
        
        for file_path, description in core_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                st.success(f"✅ {file_path} ({file_size:,} bytes)")
                st.caption(description)
            else:
                st.error(f"❌ {file_path} - Missing")
    
    with col2:
        st.subheader("Training Data Status")
        labeled_path = pathlib.Path("labeled_data")
        
        if labeled_path.exists():
            categories = [d for d in labeled_path.iterdir() if d.is_dir()]
            total_images = 0
            
            training_status = []
            for cat_dir in categories:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                image_count = len(images)
                total_images += image_count
                
                if image_count >= 3:
                    status = "Excellent"
                    icon = "✅"
                elif image_count >= 1:
                    status = "Good"
                    icon = "✅"
                else:
                    status = "Empty"
                    icon = "❌"
                
                training_status.append({
                    'Category': cat_dir.name.replace('_', ' ').title(),
                    'Images': image_count,
                    'Status': f"{icon} {status}"
                })
            
            if training_status:
                df = pd.DataFrame(training_status)
                st.dataframe(df, hide_index=True)
                st.metric("Total Training Images", total_images)
            
        else:
            st.error("❌ No labeled_data directory found")
            st.info("Run: `python adaptive_agent.py --setup-labeled-data`")
    
    # Configuration status
    st.subheader("Azure OpenAI Configuration")
    
    # Check environment variables
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_DEPLOYMENT': os.getenv('AZURE_OPENAI_DEPLOYMENT')
    }
    
    config_status = []
    for var_name, var_value in required_vars.items():
        if var_value:
            # Mask sensitive values
            if 'KEY' in var_name:
                display_value = f"{var_value[:8]}... ({len(var_value)} chars)"
            else:
                display_value = var_value
                
            config_status.append({
                'Variable': var_name,
                'Status': '✅ Configured',
                'Value': display_value
            })
        else:
            config_status.append({
                'Variable': var_name,
                'Status': '❌ Missing',
                'Value': 'Not set'
            })
    
    config_df = pd.DataFrame(config_status)
    st.dataframe(config_df, hide_index=True)
    
    # System parameters
    if os.path.exists("learning_parameters.json"):
        st.subheader("Learning Parameters")
        try:
            with open("learning_parameters.json") as f:
                params = json.load(f)
            
            # Format parameters for display
            param_display = {}
            for key, value in params.items():
                if isinstance(value, float):
                    param_display[key.replace('_', ' ').title()] = f"{value:.3f}"
                elif isinstance(value, (int, str, bool)):
                    param_display[key.replace('_', ' ').title()] = str(value)
            
            st.json(param_display)
            
        except Exception as e:
            st.error(f"Error loading parameters: {e}")
    else:
        st.info("No learning parameters file (will be created automatically)")

def display_feedback_logs():
    """Display comprehensive feedback analysis logs"""
    st.header("Feedback Analysis History")
    
    logs = load_feedback_logs()
    
    if not logs:
        st.info("No feedback analysis logs found yet.")
        st.markdown("""
        Feedback logs are created when the system analyzes extraction quality.
        
        **Automatic feedback occurs after each extraction.**
        
        **Manual feedback analysis:**
        ```bash
        python llm_feedback.py --analyze-and-apply document_id
        ```
        """)
        return
    
    st.success(f"Found {len(logs)} feedback analysis entries")
    
    # Recent feedback summary
    recent_logs = logs[-10:]  # Last 10 entries
    
    for i, log in enumerate(reversed(recent_logs)):
        timestamp = log.get('timestamp', 'Unknown')
        doc_id = log.get('document_id', 'Unknown')
        analysis_method = log.get('analysis_method', 'Unknown')
        
        # Format timestamp
        try:
            if timestamp != 'Unknown':
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_time = 'Unknown'
        except:
            formatted_time = str(timestamp)[:19]
        
        with st.expander(f"Analysis {len(recent_logs)-i}: {doc_id[:8]} - {formatted_time}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Details:**")
                st.text(f"Document ID: {doc_id}")
                st.text(f"Method: {analysis_method}")
                st.text(f"Timestamp: {formatted_time}")
                
                # Vision analysis summary
                if 'vision_analysis_summary' in log:
                    vision_summary = log['vision_analysis_summary']
                    st.markdown("**Vision Analysis Results:**")
                    
                    items_analyzed = vision_summary.get('total_items_analyzed', 0)
                    vision_available = vision_summary.get('vision_available', False)
                    
                    st.text(f"Items Analyzed: {items_analyzed}")
                    st.text(f"Vision Available: {vision_available}")
                    
                    if vision_available:
                        accuracy = vision_summary.get('extraction_accuracy_rate', 0)
                        relevance = vision_summary.get('relevance_rate', 0)
                        construction = vision_summary.get('construction_authenticity_rate', 0)
                        
                        st.metric("Accuracy Rate", f"{accuracy:.1%}")
                        st.metric("Relevance Rate", f"{relevance:.1%}")
                        st.metric("Construction Rate", f"{construction:.1%}")
                    else:
                        estimated_acc = vision_summary.get('estimated_accuracy_rate', 0)
                        st.metric("Estimated Accuracy", f"{estimated_acc:.1%}")
            
            with col2:
                # Parameter recommendations
                if 'parameter_recommendations' in log:
                    recommendations = log['parameter_recommendations']
                    st.markdown("**Parameter Recommendations:**")
                    
                    priority = recommendations.get('priority', 'medium')
                    reasoning = recommendations.get('reasoning', 'No reasoning provided')
                    confidence = recommendations.get('confidence', 0)
                    
                    st.text(f"Priority: {priority.upper()}")
                    st.text(f"Confidence: {confidence:.1%}")
                    st.text(f"Reasoning: {reasoning}")
                    
                    adjustments = recommendations.get('adjustments', {})
                    if adjustments:
                        st.markdown("**Applied Adjustments:**")
                        adj_df = pd.DataFrame([
                            {'Parameter': k.replace('_', ' ').title(), 'New Value': str(v)}
                            for k, v in adjustments.items()
                        ])
                        st.dataframe(adj_df, hide_index=True)
                    else:
                        st.info("No parameter adjustments recommended")
                
                # Quality metrics
                if 'quality_metrics' in log:
                    quality = log['quality_metrics']
                    st.markdown("**Quality Metrics:**")
                    
                    overall_quality = quality.get('overall_quality_score', 0)
                    st.metric("Overall Quality", f"{overall_quality:.1%}")
                    
                    if quality.get('extraction_efficiency'):
                        st.text(f"Efficiency: {quality['extraction_efficiency']:.3f}")
                    if quality.get('confidence_quality'):
                        st.text(f"Confidence: {quality['confidence_quality']:.3f}")

def main():
    """Main interface application"""
    st.title("Azure OpenAI-First Construction Document Analyzer")
    st.markdown("Advanced document analysis results with Azure OpenAI Vision integration")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Add status indicator in sidebar
    extractions = load_extraction_files()
    if extractions:
        azure_count = sum(1 for ex in extractions if 'azure' in ex.get('processing_method', '').lower())
        st.sidebar.metric("Recent Extractions", len(extractions))
        st.sidebar.metric("Azure Powered", f"{azure_count}/{len(extractions)}")
    
    page = st.sidebar.selectbox(
        "Choose Analysis View",
        [
            "📊 Overview & Statistics", 
            "🔍 Detailed Content Browser", 
            "📚 System Status",
            "🤖 Feedback Analysis"
        ]
    )
    
    # Main content area
    if page == "📊 Overview & Statistics":
        display_extraction_overview(extractions)
    
    elif page == "🔍 Detailed Content Browser":
        if not extractions:
            st.info("No extraction files found. Process documents first!")
        else:
            # Enhanced document selector
            st.subheader("Select Extraction to Analyze")
            
            doc_options = []
            for extraction in extractions:
                doc_path = extraction.get('document_path', 'Unknown')
                doc_name = os.path.basename(doc_path)
                doc_id = extraction.get('document_id', 'Unknown')[:8]
                target_char = extraction.get('target_characteristic', 'All')
                char_display = target_char.replace('_', ' ').title()
                
                # Add processing info
                processing_method = extraction.get('processing_method', 'unknown')
                method_icon = "🤖" if 'azure' in processing_method.lower() else "🔧"
                
                item_count = extraction.get('total_sections', 0)
                
                doc_options.append(f"{method_icon} {doc_name} - {char_display} ({item_count} items) [{doc_id}]")
            
            selected_idx = st.selectbox(
                "Select Document Extraction",
                range(len(doc_options)),
                format_func=lambda x: doc_options[x],
                help="Choose an extraction to analyze in detail"
            )
            
            if selected_idx is not None:
                extraction = extractions[selected_idx]
                
                # Show extraction details and content
                display_extraction_details(extraction)
                st.divider()
                display_extracted_content(extraction)
    
    elif page == "📚 System Status":
        display_system_status()
    
    elif page == "🤖 Feedback Analysis":
        display_feedback_logs()
    
    # Sidebar footer with quick actions
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Quick Actions:**
    ```bash
    # Test Azure connection
    python test_azure_connection.py
    
    # Process document
    python adaptive_agent.py --source doc.pdf --characteristic anchors
    
    # System diagnostic
    python diagnostic.py --full
    ```
    
    **Need Help?**
    - Check Azure OpenAI configuration
    - Verify training data quality
    - Review system status above
    """)
    
    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

if __name__ == "__main__":
    main()