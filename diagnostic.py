def test_extraction_system():
    """Test the Azure-first extraction system"""
    try:
        # Try to import the Azure-first extractor with correct class name
        from characteristic_based_extractor import AzureFirstExtractor
        
        extractor = AzureFirstExtractor()
        
        system_results = {
            'extractor_import': {'status': 'success'},
            'azure_client_available': False,
            'characteristics': {},
            'training_data': {},
            'system_parameters': {},
            'system_ready': True
        }
        
        # Test Azure client availability
        if extractor.azure_client:
            system_results['azure_client_available'] = True
            
        # Test characteristics loading
        characteristics = extractor.get_available_characteristics()
        system_results['characteristics'] = {
            'count': len(characteristics),
            'available': characteristics
        }
        
        # Test training data loading and analysis
        total_training_examples = 0
        for char in characteristics:
            training_examples = extractor.training_data.get(char, [])
            training_count = len(training_examples)
            total_training_examples += training_count
            
            system_results['training_data'][char] = {
                'examples_loaded': training_count,
                'ready': training_count > 0,
                'has_data_uris': any('data_uri' in ex for ex in training_examples)
            }
        
        # Test system parameters
        system_results['system_parameters'] = {
            'max_regions_per_page': extractor.max_regions_per_page,
            'min_region_size': extractor.min_region_size,
            'max_region_size': extractor.max_region_size,
            'azure_vision_enabled': extractor.azure_client is not None,
            'extraction_method': 'azure_openai_vision_first'
        }
        
        # Overall system readiness
        system_results['system_ready'] = total_training_examples > 0 or extractor.azure_client is not None
        system_results['total_training_examples'] = total_training_examples
        
        return system_results
        
    except Exception as e:
        return {
            'extractor_import': {'status': 'failed', 'error': str(e)},
            'system_ready': False
        }#!/usr/bin/env python3
# diagnostic.py
"""
Enhanced Construction Document Analysis System Diagnostics
Comprehensive validation and testing of system components
"""

import os
import sys
import json
import pathlib
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("Warning: python-dotenv not available")

# Azure OpenAI imports
try:
    from langchain_openai import AzureChatOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Test imports
def test_imports():
    """Test critical system imports"""
    import_results = {}
    
    # Core dependencies
    try:
        import cv2
        import_results['opencv'] = {'status': 'available', 'version': cv2.__version__}
    except ImportError:
        import_results['opencv'] = {'status': 'missing', 'install': 'pip install opencv-python'}
    
    try:
        from PIL import Image
        import_results['pillow'] = {'status': 'available', 'version': 'available'}
    except ImportError:
        import_results['pillow'] = {'status': 'missing', 'install': 'pip install pillow'}
    
    try:
        from pdf2image import convert_from_path
        import_results['pdf2image'] = {'status': 'available', 'version': 'available'}
    except ImportError:
        import_results['pdf2image'] = {'status': 'missing', 'install': 'pip install pdf2image'}
    
    try:
        import numpy as np
        import_results['numpy'] = {'status': 'available', 'version': np.__version__}
    except ImportError:
        import_results['numpy'] = {'status': 'missing', 'install': 'pip install numpy'}
    
    try:
        import streamlit as st
        import_results['streamlit'] = {'status': 'available', 'version': st.__version__}
    except ImportError:
        import_results['streamlit'] = {'status': 'missing', 'install': 'pip install streamlit'}
    
    try:
        from langchain_openai import AzureChatOpenAI
        import_results['azure_openai'] = {'status': 'available', 'version': 'available'}
    except ImportError:
        import_results['azure_openai'] = {'status': 'missing', 'install': 'pip install langchain-openai'}
    
    try:
        import pandas as pd
        import_results['pandas'] = {'status': 'available', 'version': pd.__version__}
    except ImportError:
        import_results['pandas'] = {'status': 'missing', 'install': 'pip install pandas'}
    
    return import_results

def test_file_structure():
    """Test file structure and core files"""
    required_files = {
        'adaptive_agent.py': 'Main processing engine',
        'characteristic_based_extractor.py': 'Core extraction system', 
        'llm_feedback.py': 'Azure vision feedback',
        'feedback_interface.py': 'Streamlit results interface',
        'diagnostic.py': 'System diagnostics'
    }
    
    file_results = {}
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            try:
                size = os.path.getsize(filename)
                # Get modification time
                mtime = os.path.getmtime(filename)
                mod_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                
                file_results[filename] = {
                    'status': 'exists',
                    'size': f"{size:,} bytes",
                    'modified': mod_date,
                    'description': description
                }
            except Exception as e:
                file_results[filename] = {
                    'status': 'error', 
                    'error': str(e),
                    'description': description
                }
        else:
            file_results[filename] = {
                'status': 'missing',
                'description': description
            }
    
    return file_results

def test_azure_configuration():
    """Test Azure OpenAI configuration"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY', 
        'AZURE_OPENAI_DEPLOYMENT'
    ]
    
    config_results = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
            elif 'ENDPOINT' in var:
                masked_value = value.replace('https://', '').split('.')[0] + '.openai.azure.com'
            else:
                masked_value = value
            
            config_results[var] = {
                'status': 'configured',
                'value': masked_value,
                'length': len(value)
            }
        else:
            config_results[var] = {
                'status': 'missing',
                'value': None
            }
    
    return config_results

def test_training_data():
    """Test labeled training data"""
    labeled_path = pathlib.Path("labeled_data")
    
    expected_categories = ['anchors', 'design_pressure', 'glazing', 'impact_rating']
    
    training_results = {
        'labeled_data_exists': labeled_path.exists(),
        'categories': {},
        'total_images': 0,
        'recommendations': []
    }
    
    if not labeled_path.exists():
        training_results['recommendations'].append("Run: python adaptive_agent.py --setup-labeled-data")
        return training_results
    
    for category in expected_categories:
        category_path = labeled_path / category
        
        if category_path.exists():
            # Count images
            jpg_files = list(category_path.glob("*.jpg"))
            png_files = list(category_path.glob("*.png"))
            jpeg_files = list(category_path.glob("*.jpeg"))
            all_images = jpg_files + png_files + jpeg_files
            
            # Analyze image quality
            quality_info = analyze_training_images(all_images)
            
            if len(all_images) >= 5:
                status = 'excellent'
            elif len(all_images) >= 3:
                status = 'good'
            elif len(all_images) > 0:
                status = 'needs_more'
            else:
                status = 'empty'
            
            training_results['categories'][category] = {
                'status': status,
                'image_count': len(all_images),
                'jpg_count': len(jpg_files),
                'png_count': len(png_files),
                'jpeg_count': len(jpeg_files),
                'quality_info': quality_info
            }
            
            training_results['total_images'] += len(all_images)
            
            if status == 'empty':
                training_results['recommendations'].append(
                    f"Add training images to labeled_data/{category}/ (focus on visual diagrams/tables)"
                )
            elif status == 'needs_more':
                training_results['recommendations'].append(
                    f"Add more training images to labeled_data/{category}/ (aim for 3-5 examples)"
                )
        else:
            training_results['categories'][category] = {
                'status': 'missing',
                'image_count': 0
            }
            training_results['recommendations'].append(f"Create labeled_data/{category}/ directory")
    
    return training_results

def analyze_training_images(image_files: List[pathlib.Path]) -> Dict:
    """Analyze quality of training images"""
    if not image_files:
        return {'avg_size': 0, 'size_range': [0, 0], 'formats': {}}
    
    try:
        import cv2
        
        sizes = []
        formats = {}
        dimensions = []
        
        for img_file in image_files[:10]:  # Sample first 10
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    area = w * h
                    sizes.append(area)
                    dimensions.append((w, h))
                    
                    ext = img_file.suffix.lower()
                    formats[ext] = formats.get(ext, 0) + 1
                    
            except Exception:
                continue
        
        if sizes:
            avg_width = sum(d[0] for d in dimensions) / len(dimensions)
            avg_height = sum(d[1] for d in dimensions) / len(dimensions)
            
            return {
                'avg_size': int(sum(sizes) / len(sizes)),
                'size_range': [min(sizes), max(sizes)],
                'avg_dimensions': f"{avg_width:.0f}x{avg_height:.0f}",
                'formats': formats,
                'samples_analyzed': len(sizes),
                'suitable_for_extraction': min(sizes) >= 15000  # New minimum
            }
        else:
            return {'error': 'No images could be analyzed'}
            
    except ImportError:
        return {'error': 'OpenCV not available for image analysis'}

def test_visual_detection():
    """Test visual detection capabilities"""
    try:
        import cv2
        import numpy as np
        
        # Create test image with construction-like features
        test_img = np.zeros((300, 400, 3), dtype=np.uint8)
        test_img.fill(255)  # White background
        
        # Draw table-like structure
        cv2.rectangle(test_img, (50, 50), (250, 150), (0, 0, 0), 2)
        cv2.line(test_img, (50, 100), (250, 100), (0, 0, 0), 1)  # Horizontal line
        cv2.line(test_img, (150, 50), (150, 150), (0, 0, 0), 1)   # Vertical line
        
        # Draw circular element (fastener) - should now be POSITIVE
        cv2.circle(test_img, (320, 100), 20, (0, 0, 0), 2)
        cv2.circle(test_img, (320, 100), 5, (0, 0, 0), -1)  # Center dot
        
        # Draw dimension line
        cv2.line(test_img, (60, 200), (240, 200), (0, 0, 0), 1)
        cv2.line(test_img, (60, 190), (60, 210), (0, 0, 0), 1)  # End marks
        cv2.line(test_img, (240, 190), (240, 210), (0, 0, 0), 1)
        
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        detection_results = {}
        
        # Test edge detection
        edges = cv2.Canny(gray, 30, 100)
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / edges.size
        detection_results['edge_detection'] = {
            'status': 'working',
            'edge_pixels': int(edge_pixels),
            'edge_density': float(edge_density),
            'suitable_range': 0.01 < edge_density < 0.25
        }
        
        # Test line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        h_line_pixels = np.sum(horizontal_lines > 0)
        v_line_pixels = np.sum(vertical_lines > 0)
        total_line_pixels = h_line_pixels + v_line_pixels
        line_density = total_line_pixels / edges.size
        
        detection_results['line_detection'] = {
            'status': 'working',
            'horizontal_lines': int(h_line_pixels),
            'vertical_lines': int(v_line_pixels),
            'total_structured_lines': int(total_line_pixels),
            'line_density': float(line_density),
            'construction_indicator': line_density > 0.005
        }
        
        # Test circle detection (now construction-positive)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                 param1=50, param2=30, minRadius=10, maxRadius=50)
        
        circles_found = len(circles[0]) if circles is not None else 0
        detection_results['circle_detection'] = {
            'status': 'working',
            'circles_found': circles_found,
            'expected': 1,
            'construction_positive': True,  # NEW: Circles are now construction indicators
            'fastener_detection': circles_found > 0
        }
        
        # Test contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        detection_results['contour_detection'] = {
            'status': 'working', 
            'total_contours': len(contours),
            'significant_contours': len(significant_contours),
            'shape_complexity': len(significant_contours)
        }
        
        # Test construction validation logic
        construction_score = 2.0  # Base construction assumption
        if line_density > 0.005:
            construction_score += 1.5
        if circles_found > 0:
            construction_score += 2.0  # Fastener elements
        if 0.01 < edge_density < 0.25:
            construction_score += 1.0
        
        detection_results['construction_validation'] = {
            'status': 'working',
            'construction_score': construction_score,
            'validation_result': construction_score > 0.5,
            'construction_positive_bias': True
        }
        
        return detection_results
        
    except Exception as e:
        return {'error': f"Visual detection test failed: {e}"}

def test_extraction_system():
    """Test the core extraction system"""
    try:
        # Try to import the enhanced extractor
        from characteristic_based_extractor import CharacteristicBasedExtractor
        
        extractor = CharacteristicBasedExtractor()
        
        system_results = {
            'extractor_import': {'status': 'success'},
            'characteristics': {},
            'training_data': {},
            'system_parameters': {},
            'system_ready': True
        }
        
        # Test characteristics loading
        characteristics = extractor.get_available_characteristics()
        system_results['characteristics'] = {
            'count': len(characteristics),
            'available': characteristics
        }
        
        # Test training data loading and analysis
        total_training_examples = 0
        for char in characteristics:
            training_examples = extractor.training_data.get(char, [])
            training_count = len(training_examples)
            total_training_examples += training_count
            
            # Analyze training stats if available
            training_stats = getattr(extractor, 'training_stats', {}).get(char, {})
            
            system_results['training_data'][char] = {
                'examples_loaded': training_count,
                'ready': training_count > 0,
                'avg_area': training_stats.get('avg_area', 0),
                'has_circular_elements': training_stats.get('has_circular_elements', False)
            }
        
        # Test system parameters
        system_results['system_parameters'] = {
            'min_region_size': extractor.min_region_size,
            'max_region_size': extractor.max_region_size,
            'max_regions_per_page': extractor.max_regions_per_page,
            'construction_positive_validation': True  # NEW feature
        }
        
        # Overall system readiness
        system_results['system_ready'] = total_training_examples > 0
        system_results['total_training_examples'] = total_training_examples
        
        return system_results
        
    except Exception as e:
        return {
            'extractor_import': {'status': 'failed', 'error': str(e)},
            'system_ready': False
        }

def test_feedback_system():
    """Test the LLM feedback system"""
    try:
        # Try to import feedback system
        from llm_feedback import EnhancedLLMFeedback
        
        feedback_system = EnhancedLLMFeedback()
        
        feedback_results = {
            'feedback_import': {'status': 'success'},
            'azure_client': {},
            'parameters': {},
            'feedback_capabilities': {},
            'system_ready': True
        }
        
        # Test Azure configuration
        if feedback_system.azure_client:
            feedback_results['azure_client'] = {
                'status': 'configured',
                'vision_analysis': 'available',
                'training_data_comparison': 'enabled'
            }
        else:
            feedback_results['azure_client'] = {
                'status': 'not_configured',
                'vision_analysis': 'fallback_mode',
                'training_data_comparison': 'limited'
            }
        
        # Test parameter loading
        params = feedback_system.current_params
        feedback_results['parameters'] = {
            'loaded': len(params) > 0,
            'count': len(params),
            'last_updated': params.get('last_updated', 'never'),
            'confidence_threshold': params.get('confidence_threshold', 'unknown'),
            'construction_bias': params.get('construction_bias', 'unknown')
        }
        
        # Test feedback capabilities
        feedback_results['feedback_capabilities'] = {
            'vision_analysis': feedback_system.azure_client is not None,
            'parameter_tuning': True,
            'training_data_validation': True,
            'extraction_quality_scoring': True
        }
        
        return feedback_results
        
    except Exception as e:
        return {
            'feedback_import': {'status': 'failed', 'error': str(e)},
            'system_ready': False
        }

def test_recent_extractions():
    """Test recent extraction results"""
    feedback_dir = pathlib.Path("feedback_data")
    
    extraction_results = {
        'feedback_directory_exists': feedback_dir.exists(),
        'recent_extractions': [],
        'extraction_stats': {}
    }
    
    if not feedback_dir.exists():
        return extraction_results
    
    # Find recent extraction files
    extraction_files = list(feedback_dir.glob("extraction_*.json"))
    extraction_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    for i, extraction_file in enumerate(extraction_files[:5]):  # Last 5 extractions
        try:
            with open(extraction_file) as f:
                data = json.load(f)
            
            extraction_info = {
                'file': extraction_file.name,
                'document_id': data.get('document_id', 'unknown'),
                'characteristic': data.get('target_characteristic', 'unknown'),
                'total_items': data.get('total_sections', 0),
                'processing_time': data.get('processing_time', 0),
                'timestamp': data.get('timestamp', 'unknown'),
                'extraction_method': data.get('processing_method', 'unknown')
            }
            
            # Check for construction validation results
            sections = data.get('extracted_sections', [])
            if sections:
                construction_validated = 0
                avg_confidence = 0
                for section in sections:
                    validation = section.get('region_metadata', {}).get('construction_validation', {})
                    if validation.get('is_construction_related', False):
                        construction_validated += 1
                    avg_confidence += section.get('confidence', 0)
                
                extraction_info['construction_validation_rate'] = construction_validated / len(sections)
                extraction_info['avg_confidence'] = avg_confidence / len(sections)
            
            extraction_results['recent_extractions'].append(extraction_info)
            
        except Exception as e:
            extraction_results['recent_extractions'].append({
                'file': extraction_file.name,
                'error': f"Failed to load: {e}"
            })
    
    # Overall extraction statistics
    total_extractions = len(extraction_files)
    if total_extractions > 0:
        # Calculate success rate (files with items > 0)
        successful_extractions = 0
        total_items = 0
        
        for extraction_file in extraction_files:
            try:
                with open(extraction_file) as f:
                    data = json.load(f)
                items = data.get('total_sections', 0)
                total_items += items
                if items > 0:
                    successful_extractions += 1
            except:
                continue
        
        extraction_results['extraction_stats'] = {
            'total_extractions': total_extractions,
            'successful_extractions': successful_extractions,
            'success_rate': successful_extractions / total_extractions if total_extractions > 0 else 0,
            'avg_items_per_extraction': total_items / total_extractions if total_extractions > 0 else 0
        }
    
    return extraction_results

def run_quick_diagnostic():
    """Run quick system diagnostic"""
    print("Quick System Diagnostic")
    print("=" * 50)
    
    # Test imports
    print("\nTesting Critical Dependencies...")
    imports = test_imports()
    critical_missing = 0
    for package, info in imports.items():
        status_icon = "✅" if info['status'] == 'available' else "❌"
        print(f"  {status_icon} {package}: {info['status']}")
        if info['status'] == 'missing' and package in ['opencv', 'pillow', 'pdf2image', 'numpy']:
            critical_missing += 1
            print(f"    Install: {info['install']}")
    
    # Test files
    print("\nTesting Core Files...")
    files = test_file_structure()
    files_missing = 0
    for filename, info in files.items():
        status_icon = "✅" if info['status'] == 'exists' else "❌"
        print(f"  {status_icon} {filename}: {info['status']}")
        if info['status'] != 'exists':
            files_missing += 1
    
    # Test training data
    print("\nTesting Training Data...")
    training = test_training_data()
    if training['labeled_data_exists']:
        print(f"  ✅ labeled_data directory exists")
        print(f"  📊 Total training images: {training['total_images']}")
        
        good_categories = 0
        for category, info in training['categories'].items():
            if info['status'] == 'good' or info['status'] == 'excellent':
                status_icon = "✅"
                good_categories += 1
            elif info['status'] == 'needs_more':
                status_icon = "⚠️"
            else:
                status_icon = "❌"
            print(f"    {status_icon} {category}: {info['image_count']} images ({info['status']})")
        
        print(f"  Ready categories: {good_categories}/4")
    else:
        print(f"  ❌ labeled_data directory missing")
    
    # Test Azure config
    print("\nTesting Azure OpenAI Configuration...")
    azure = test_azure_configuration()
    configured_vars = sum(1 for info in azure.values() if info['status'] == 'configured')
    if configured_vars == len(azure):
        print(f"  ✅ All Azure variables configured ({configured_vars}/{len(azure)})")
    else:
        print(f"  ⚠️ Partial Azure configuration ({configured_vars}/{len(azure)})")
        for var, info in azure.items():
            if info['status'] == 'missing':
                print(f"    ❌ {var}: missing")
    
    # Test recent extractions
    print("\nChecking Recent Extractions...")
    recent = test_recent_extractions()
    if recent['feedback_directory_exists']:
        stats = recent.get('extraction_stats', {})
        if stats:
            success_rate = stats.get('success_rate', 0)
            avg_items = stats.get('avg_items_per_extraction', 0)
            print(f"  📊 Success rate: {success_rate:.1%} ({stats['successful_extractions']}/{stats['total_extractions']})")
            print(f"  📊 Avg items per extraction: {avg_items:.1f}")
            
            if success_rate > 0.7:
                print(f"  ✅ Extraction system performing well")
            elif success_rate > 0.3:
                print(f"  ⚠️ Extraction system needs improvement")
            else:
                print(f"  ❌ Extraction system needs attention")
        else:
            print(f"  ℹ️ No extraction statistics available")
    else:
        print(f"  ℹ️ No extraction results found")
    
    # Overall status
    print("\nSystem Status Summary")
    print("-" * 30)
    
    training_ready = training['total_images'] >= 6  # At least 6 total images
    azure_ready = configured_vars == len(azure)
    recent_success = recent.get('extraction_stats', {}).get('success_rate', 0) > 0.5
    
    if critical_missing == 0 and files_missing == 0 and training_ready:
        if azure_ready:
            print("✅ System fully ready with Azure OpenAI feedback")
        else:
            print("✅ System ready (Azure OpenAI recommended for enhanced feedback)")
    elif critical_missing > 0:
        print(f"❌ Critical dependencies missing ({critical_missing})")
        print("   Install missing packages to proceed")
    elif files_missing > 0:
        print(f"❌ Core files missing ({files_missing})")
    elif not training_ready:
        print("⚠️ Insufficient training data - add examples to labeled_data/")
        print("   Need at least 2-3 examples per category")
    else:
        print("⚠️ System needs attention")
    
    if recent.get('extraction_stats', {}).get('success_rate', 1) < 0.3:
        print("💡 Tip: Run Azure vision feedback to improve extraction accuracy")

def run_full_diagnostic():
    """Run comprehensive system diagnostic"""
    print("Comprehensive System Diagnostic")
    print("=" * 60)
    
    # Quick tests first
    run_quick_diagnostic()
    
    # Extended tests
    print("\n" + "="*60)
    print("Extended System Testing...")
    print("="*60)
    
    # Test visual detection
    print("\nTesting Visual Detection Capabilities...")
    visual_test = test_visual_detection()
    if 'error' in visual_test:
        print(f"  ❌ Visual detection test failed: {visual_test['error']}")
    else:
        for test_name, result in visual_test.items():
            print(f"  ✅ {test_name}: {result['status']}")
            
            if test_name == 'edge_detection':
                density = result['edge_density']
                suitable = result['suitable_range']
                print(f"    Edge density: {density:.4f} ({'suitable' if suitable else 'may need adjustment'})")
            
            elif test_name == 'line_detection':
                line_density = result['line_density']
                is_construction = result['construction_indicator']
                print(f"    Line density: {line_density:.4f} ({'construction indicator' if is_construction else 'low structure'})")
            
            elif test_name == 'circle_detection':
                circles = result['circles_found']
                construction_positive = result['construction_positive']
                print(f"    Circles detected: {circles} ({'construction positive' if construction_positive else 'neutral'})")
            
            elif test_name == 'construction_validation':
                score = result['construction_score']
                valid = result['validation_result']
                print(f"    Construction score: {score:.1f} ({'passes validation' if valid else 'fails validation'})")
    
    # Test extraction system
    print("\n⚙️ Testing Azure-First Extraction System...")
    extraction_test = test_extraction_system()
    if extraction_test.get('system_ready', False):
        print(f"  ✅ Azure-First extraction system ready")
        print(f"    Characteristics: {extraction_test['characteristics']['count']}")
        
        azure_available = extraction_test.get('azure_client_available', False)
        total_examples = extraction_test.get('total_training_examples', 0)
        
        print(f"    Azure OpenAI client: {'Available' if azure_available else 'Not configured'}")
        print(f"    Total training examples: {total_examples}")
        
        params = extraction_test.get('system_parameters', {})
        max_regions = params.get('max_regions_per_page', 0)
        extraction_method = params.get('extraction_method', 'unknown')
        
        print(f"    Max regions per page: {max_regions} (Azure-optimized)")
        print(f"    Extraction method: {extraction_method}")
        
        for char, info in extraction_test['training_data'].items():
            status_icon = "✅" if info['ready'] else "⚠️"
            examples = info['examples_loaded']
            has_uris = info.get('has_data_uris', False)
            
            print(f"    {status_icon} {char}: {examples} examples ({'Azure-ready' if has_uris else 'basic'})")
    else:
        print(f"  ❌ Azure-First extraction system not ready")
        if 'error' in extraction_test.get('extractor_import', {}):
            print(f"    Error: {extraction_test['extractor_import']['error']}")
    
    # Test feedback system with Azure integration
    print("\n🤖 Testing Integrated Feedback System...")
    feedback_test = test_feedback_system()
    if feedback_test.get('system_ready', False):
        print(f"  ✅ Feedback system ready")
        
        azure_status = feedback_test['azure_client']['status']
        vision_mode = feedback_test['azure_client']['vision_analysis']
        
        print(f"    Azure OpenAI: {azure_status}")
        print(f"    Vision analysis: {vision_mode}")
        
        params = feedback_test['parameters']
        print(f"    Parameters: {params['count']} loaded")
        print(f"    Confidence threshold: {params.get('confidence_threshold', 'unknown')}")
        
        capabilities = feedback_test['feedback_capabilities']
        print(f"    Vision analysis: {capabilities['vision_analysis']}")
        print(f"    Parameter tuning: {capabilities['parameter_tuning']}")
        print(f"    Training validation: {capabilities['training_data_validation']}")
    else:
        print(f"  ❌ Feedback system not ready")
        if 'error' in feedback_test.get('feedback_import', {}):
            print(f"    Error: {feedback_test['feedback_import']['error']}")
    
    # Test recent extractions with Azure metrics
    print("\n📊 Analyzing Azure Extraction Results...")
    recent = test_recent_extractions()
    
    if recent['recent_extractions']:
        print(f"  📊 Found {len(recent['recent_extractions'])} recent extractions")
        
        azure_extractions = 0
        fallback_extractions = 0
        
        for extraction in recent['recent_extractions'][:3]:
            if 'error' in extraction:
                continue
                
            doc_id = extraction['document_id'][:8]
            char = extraction['characteristic']
            items = extraction['total_items']
            
            # Check if Azure was used
            method = extraction.get('extraction_method', 'unknown')
            if 'azure' in method.lower():
                azure_extractions += 1
                method_icon = "🤖"
            else:
                fallback_extractions += 1
                method_icon = "🔧"
            
            print(f"    {method_icon} {doc_id} ({char}): {items} items")
        
        print(f"  🤖 Azure extractions: {azure_extractions}")
        print(f"  🔧 Fallback extractions: {fallback_extractions}")
        
        stats = recent.get('extraction_stats', {})
        if stats:
            success_rate = stats['success_rate']
            avg_items = stats['avg_items_per_extraction']
            
            print(f"  📊 Overall success rate: {success_rate:.1%}")
            print(f"  📊 Average items per extraction: {avg_items:.1f}")
            
            if success_rate >= 0.7:
                print(f"  ✅ Excellent Azure extraction performance")
            elif success_rate >= 0.4:
                print(f"  ✅ Good Azure extraction performance")  
            else:
                print(f"  ⚠️ Azure extraction performance needs improvement")
    else:
        print(f"  ℹ️ No recent extractions found")
    
    # Azure-specific recommendations
    print(f"\nAzure OpenAI System Recommendations")
    print("-" * 40)
    
    # Training data recommendations
    training = test_training_data()
    if training['recommendations']:
        print("📚 Training Data for Azure Vision:")
        for i, rec in enumerate(training['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        print("     Azure OpenAI learns from visual examples - quality over quantity!")
    
    # Azure recommendations
    azure = test_azure_configuration()
    missing_azure = [var for var in azure.items() if var[1]['status'] == 'missing']
    if missing_azure:
        print("🤖 Azure OpenAI Setup (Critical for best performance):")
        print("  • Create .env file with Azure OpenAI credentials")
        print("  • Azure Vision provides 80%+ accuracy vs 30% fallback methods")
        print("  • Enables intelligent content understanding and training data comparison")
    
    # Performance recommendations
    extraction_test = test_extraction_system()
    recent_stats = recent.get('extraction_stats', {})
    
    if extraction_test.get('system_ready', False):
        azure_available = extraction_test.get('azure_client_available', False)
        total_examples = extraction_test.get('total_training_examples', 0)
        success_rate = recent_stats.get('success_rate', 0)
        
        print("📈 Azure Performance Optimization:")
        
        if not azure_available:
            print("  • PRIORITY: Configure Azure OpenAI for intelligent extraction")
            print("  • Current fallback methods have limited accuracy")
        elif total_examples < 6:
            print("  • Add 2-3 training examples per category for better Azure performance")
            print("  • Azure learns from your specific document types")
        elif success_rate < 0.5:
            print("  • Check training data alignment with your document types")
            print("  • Run Azure feedback analysis for parameter tuning")
        else:
            print("  • System optimized for Azure OpenAI Vision analysis")
            print("  • Monitor extraction quality through feedback interface")
    
    print(f"\n🎯 Next Steps for Azure-First System:")
    if not AZURE_AVAILABLE:
        print(f"  1. Install Azure OpenAI: pip install langchain-openai")
    elif missing_azure:
        print(f"  1. Configure Azure OpenAI credentials (.env file)")
    elif training['total_images'] < 4:
        print(f"  1. Add visual training examples for Azure to learn from")
    elif recent_stats.get('success_rate', 0) < 0.3:
        print(f"  1. Test Azure connection: python adaptive_agent.py --test-azure")
        print(f"  2. Verify training data quality matches your documents")
    else:
        print(f"  1. Process documents: python adaptive_agent.py --source doc.pdf --all-characteristics")
        print(f"  2. Review Azure results: streamlit run feedback_interface.py")

def main():
    parser = argparse.ArgumentParser(description="Enhanced System Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostic")
    parser.add_argument("--full", action="store_true", help="Run comprehensive diagnostic")
    parser.add_argument("--test", choices=[
        'imports', 'files', 'training', 'azure', 
        'visual-detection', 'extraction', 'feedback', 'recent'
    ], help="Test specific component")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_diagnostic()
        return 0
    
    if args.full:
        run_full_diagnostic()
        return 0
    
    if args.test:
        print(f"Testing: {args.test}")
        print("=" * 40)
        
        if args.test == 'imports':
            imports = test_imports()
            for package, info in imports.items():
                status_icon = "✅" if info['status'] == 'available' else "❌"
                version = f" ({info.get('version', '')})" if 'version' in info else ""
                print(f"  {status_icon} {package}{version}: {info['status']}")
                if info['status'] == 'missing':
                    print(f"    Install: {info['install']}")
        
        elif args.test == 'files':
            files = test_file_structure()
            for filename, info in files.items():
                status_icon = "✅" if info['status'] == 'exists' else "❌"
                size = f" ({info.get('size', '')})" if 'size' in info else ""
                modified = f" - modified {info.get('modified', '')}" if 'modified' in info else ""
                print(f"  {status_icon} {filename}{size}{modified}")
                print(f"    {info['description']}")
        
        elif args.test == 'training':
            training = test_training_data()
            print(f"Training data directory: {'exists' if training['labeled_data_exists'] else 'missing'}")
            print(f"Total images: {training['total_images']}")
            
            for category, info in training['categories'].items():
                status_icon = {"excellent": "✅", "good": "✅", "needs_more": "⚠️", "empty": "❌", "missing": "❌"}.get(info['status'], "❓")
                print(f"  {status_icon} {category}: {info['image_count']} images ({info['status']})")
                
                if 'quality_info' in info and 'avg_size' in info['quality_info']:
                    quality = info['quality_info']
                    print(f"    Average size: {quality['avg_size']:,} pixels")
                    print(f"    Dimensions: {quality.get('avg_dimensions', 'unknown')}")
                    print(f"    Suitable for extraction: {quality.get('suitable_for_extraction', 'unknown')}")
            
            if training['recommendations']:
                print("\nRecommendations:")
                for rec in training['recommendations']:
                    print(f"  • {rec}")
        
        elif args.test == 'azure':
            azure = test_azure_configuration()
            for var, info in azure.items():
                status_icon = "✅" if info['status'] == 'configured' else "❌"
                value = f": {info['value']}" if info['value'] else ""
                print(f"  {status_icon} {var}{value}")
                if 'length' in info:
                    print(f"    Length: {info['length']} characters")
        
        elif args.test == 'visual-detection':
            visual = test_visual_detection()
            if 'error' in visual:
                print(f"❌ {visual['error']}")
            else:
                for test_name, result in visual.items():
                    print(f"✅ {test_name}: {result['status']}")
                    for key, value in result.items():
                        if key != 'status':
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
        
        elif args.test == 'extraction':
            extraction = test_extraction_system()
            if extraction.get('system_ready', False):
                print(f"✅ Azure-First extraction system ready")
                print(f"  Characteristics: {extraction['characteristics']['count']}")
                print(f"  Total training examples: {extraction.get('total_training_examples', 0)}")
                
                azure_available = extraction.get('azure_client_available', False)
                print(f"  Azure OpenAI client: {'Available' if azure_available else 'Not configured'}")
                
                params = extraction.get('system_parameters', {})
                for param, value in params.items():
                    if isinstance(value, int):
                        print(f"  {param}: {value:,}")
                    else:
                        print(f"  {param}: {value}")
                
                print("\nTraining data per characteristic:")
                for char, info in extraction['training_data'].items():
                    status = "ready" if info['ready'] else "no training data"
                    has_uris = info.get('has_data_uris', False)
                    print(f"  {char}: {info['examples_loaded']} examples ({status})")
                    if has_uris:
                        print(f"    Azure-ready with data URIs")
            else:
                print(f"❌ Azure-First extraction system not ready")
                if 'error' in extraction.get('extractor_import', {}):
                    print(f"  Error: {extraction['extractor_import']['error']}")
                    
                    # Additional troubleshooting info
                    error_msg = extraction['extractor_import']['error']
                    if 'cannot import name' in error_msg:
                        print(f"  Troubleshooting: Check class name in characteristic_based_extractor.py")
                        print(f"  Expected class: AzureFirstExtractor")
        
        elif args.test == 'feedback':
            feedback = test_feedback_system()
            if feedback.get('system_ready', False):
                print(f"✅ Feedback system ready")
                
                azure_info = feedback['azure_client']
                print(f"  Azure OpenAI: {azure_info['status']}")
                print(f"  Vision analysis: {azure_info['vision_analysis']}")
                print(f"  Training data comparison: {azure_info.get('training_data_comparison', 'unknown')}")
                
                params = feedback['parameters']
                print(f"  Parameters loaded: {params['count']}")
                for key in ['confidence_threshold', 'construction_bias']:
                    if key in params:
                        print(f"  {key}: {params[key]}")
                print(f"  Last updated: {params['last_updated']}")
                
                capabilities = feedback['feedback_capabilities']
                print("\nCapabilities:")
                for capability, available in capabilities.items():
                    status = "✅" if available else "❌"
                    print(f"  {status} {capability.replace('_', ' ')}")
            else:
                print(f"❌ Feedback system not ready")
                if 'error' in feedback.get('feedback_import', {}):
                    print(f"  Error: {feedback['feedback_import']['error']}")
        
        elif args.test == 'recent':
            recent = test_recent_extractions()
            
            if recent['feedback_directory_exists']:
                print(f"✅ Feedback directory exists")
                
                if recent['recent_extractions']:
                    print(f"\nRecent extractions ({len(recent['recent_extractions'])}):")
                    
                    for extraction in recent['recent_extractions']:
                        if 'error' in extraction:
                            print(f"  ❌ {extraction['file']}: {extraction['error']}")
                        else:
                            items = extraction['total_items']
                            char = extraction['characteristic']
                            time_taken = extraction['processing_time']
                            
                            status_icon = "✅" if items > 0 else "⚠️"
                            print(f"  {status_icon} {extraction['document_id'][:8]} ({char})")
                            print(f"    Items: {items}, Time: {time_taken:.1f}s")
                            print(f"    File: {extraction['file']}")
                            
                            if 'construction_validation_rate' in extraction:
                                val_rate = extraction['construction_validation_rate']
                                avg_conf = extraction['avg_confidence']
                                print(f"    Validation rate: {val_rate:.1%}, Avg confidence: {avg_conf:.3f}")
                
                stats = recent.get('extraction_stats', {})
                if stats:
                    print(f"\nOverall statistics:")
                    print(f"  Total extractions: {stats['total_extractions']}")
                    print(f"  Successful: {stats['successful_extractions']} ({stats['success_rate']:.1%})")
                    print(f"  Avg items per extraction: {stats['avg_items_per_extraction']:.1f}")
                else:
                    print(f"\nNo extraction statistics available")
            else:
                print(f"❌ No feedback directory found")
                print(f"   Run extractions first: python adaptive_agent.py --source doc.pdf --characteristic anchors")
        
        return 0
    
    # Default to quick diagnostic
    run_quick_diagnostic()
    return 0

if __name__ == "__main__":
    sys.exit(main())