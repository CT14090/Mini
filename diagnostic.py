#!/usr/bin/env python3
# diagnostic.py
"""
Enhanced Diagnostic Tool for Visual Content Extraction System
Comprehensive testing and validation of visual detection capabilities
"""

import json
import os
import pathlib
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Test imports
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV_AVAILABLE = True
except ImportError:
    pass

PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass

CHARACTERISTIC_EXTRACTOR_AVAILABLE = False
try:
    from characteristic_based_extractor import CharacteristicBasedExtractor
    CHARACTERISTIC_EXTRACTOR_AVAILABLE = True
except ImportError:
    pass

LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

class EnhancedDiagnostic:
    """Enhanced diagnostic tool for visual content extraction system"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_comprehensive_diagnostic(self) -> Dict:
        """Run complete diagnostic suite"""
        print("🔧 ENHANCED VISUAL EXTRACTION DIAGNOSTIC")
        print("="*60)
        
        tests = [
            ("Dependencies", self.test_dependencies),
            ("Core Files", self.test_core_files),
            ("Configuration", self.test_configuration),
            ("Visual Detection", self.test_visual_detection_capabilities),
            ("Training Data", self.test_training_data_quality),
            ("Extraction Pipeline", self.test_extraction_pipeline),
            ("Parameter Learning", self.test_parameter_learning),
            ("Azure OpenAI", self.test_azure_openai),
            ("Recent Extractions", self.test_recent_extractions),
            ("System Performance", self.test_system_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✅ PASS" if result.get('status') == 'pass' else "❌ FAIL"
                print(f"   {status} {result.get('message', 'No message')}")
                
                if result.get('details'):
                    for detail in result['details'][:3]:  # Show top 3 details
                        print(f"     • {detail}")
                        
            except Exception as e:
                self.results[test_name] = {'status': 'fail', 'message': f'Test error: {e}'}
                print(f"   ❌ FAIL Test error: {e}")
        
        return self._generate_diagnostic_summary()
    
    def test_dependencies(self) -> Dict:
        """Test all required dependencies"""
        issues = []
        
        if not PDF2IMAGE_AVAILABLE:
            issues.append("pdf2image missing - install with: pip install pdf2image")
        
        if not CV_AVAILABLE:
            issues.append("OpenCV missing - install with: pip install opencv-python")
        
        if not CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            issues.append("characteristic_based_extractor.py not found or has errors")
        
        # Test OpenCV capabilities
        if CV_AVAILABLE:
            try:
                # Test core OpenCV functions used in visual detection
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                test_img.fill(255)
                
                # Test edge detection
                gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 30, 100)
                
                # Test morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                
                # Test contour detection
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Test histogram calculation
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                
            except Exception as e:
                issues.append(f"OpenCV functionality test failed: {e}")
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Dependencies check: {len(issues)} issues found" if issues else "All dependencies available",
            'details': issues,
            'available_modules': {
                'pdf2image': PDF2IMAGE_AVAILABLE,
                'opencv': CV_AVAILABLE,
                'extractor': CHARACTERISTIC_EXTRACTOR_AVAILABLE,
                'langchain': LANGCHAIN_AVAILABLE
            }
        }
    
    def test_core_files(self) -> Dict:
        """Test presence and basic integrity of core files"""
        required_files = {
            'adaptive_agent.py': 'Main processing script',
            'characteristic_based_extractor.py': 'Visual content extraction engine',
            'llm_feedback.py': 'Vision feedback system',
            'feedback_interface.py': 'Results viewer interface'
        }
        
        issues = []
        found_files = {}
        
        for filename, description in required_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = len(content.split('\n'))
                        
                    found_files[filename] = {
                        'exists': True,
                        'size_kb': round(len(content) / 1024, 1),
                        'lines': lines
                    }
                    
                    # Basic integrity checks
                    if filename == 'characteristic_based_extractor.py':
                        if 'detect_visual_content_regions' not in content:
                            issues.append(f"{filename} missing visual detection methods")
                        if '_detect_table_regions' not in content:
                            issues.append(f"{filename} missing table detection")
                        if '_detect_diagram_regions' not in content:
                            issues.append(f"{filename} missing diagram detection")
                    
                    elif filename == 'adaptive_agent.py':
                        if 'EnhancedDocumentProcessor' not in content:
                            issues.append(f"{filename} missing enhanced processor class")
                    
                except Exception as e:
                    issues.append(f"Error reading {filename}: {e}")
                    found_files[filename] = {'exists': True, 'error': str(e)}
            else:
                issues.append(f"Missing required file: {filename} ({description})")
                found_files[filename] = {'exists': False}
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Core files: {len(issues)} issues found" if issues else "All core files present and valid",
            'details': issues,
            'file_info': found_files
        }
    
    def test_configuration(self) -> Dict:
        """Test system configuration"""
        issues = []
        config_info = {}
        
        # Load environment variables like the main system does
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except:
            pass
        
        # Check Azure OpenAI configuration
        required_azure_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_azure = [var for var in required_azure_vars if not os.getenv(var)]
        
        if missing_azure:
            issues.append(f"Azure OpenAI missing variables: {missing_azure}")
            config_info['azure_openai'] = 'incomplete'
        else:
            config_info['azure_openai'] = 'configured'
        
        # Check learning parameters
        param_file = "learning_parameters.json"
        if os.path.exists(param_file):
            try:
                with open(param_file) as f:
                    params = json.load(f)
                config_info['learning_parameters'] = params
                
                # Validate parameter ranges
                if 'confidence_threshold' in params:
                    ct = params['confidence_threshold']
                    if not (0.1 <= ct <= 0.9):
                        issues.append(f"confidence_threshold out of range: {ct}")
                
                if 'min_region_size' in params:
                    mrs = params['min_region_size']
                    if not (1000 <= mrs <= 50000):
                        issues.append(f"min_region_size out of range: {mrs}")
                        
            except Exception as e:
                issues.append(f"Error loading learning parameters: {e}")
        else:
            config_info['learning_parameters'] = 'will_be_created'
        
        # Check directories
        required_dirs = ['feedback_data', 'labeled_data']
        for dir_name in required_dirs:
            if pathlib.Path(dir_name).exists():
                config_info[f'{dir_name}_dir'] = 'exists'
            else:
                config_info[f'{dir_name}_dir'] = 'missing'
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Configuration: {len(issues)} issues found" if issues else "Configuration looks good",
            'details': issues,
            'config_info': config_info
        }
    
    def test_visual_detection_capabilities(self) -> Dict:
        """Test visual detection algorithms"""
        if not CV_AVAILABLE:
            return {
                'status': 'fail',
                'message': "OpenCV not available for visual detection testing",
                'details': ['Install OpenCV with: pip install opencv-python']
            }
        
        issues = []
        detection_results = {}
        
        try:
            # Create test images with different content types
            test_scenarios = self._create_visual_test_scenarios()
            
            for scenario_name, test_image in test_scenarios.items():
                print(f"     Testing {scenario_name} detection...")
                
                results = self._test_detection_on_image(test_image, scenario_name)
                detection_results[scenario_name] = results
                
                if not results.get('detected', False):
                    issues.append(f"{scenario_name} detection failed")
        
        except Exception as e:
            issues.append(f"Visual detection test error: {e}")
        
        # Test extractor if available
        if CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            try:
                extractor = CharacteristicBasedExtractor()
                detection_results['extractor_initialized'] = True
                
                # Test visual feature extraction
                test_img = np.zeros((200, 300, 3), dtype=np.uint8)
                test_img.fill(255)
                cv2.rectangle(test_img, (50, 50), (250, 150), (0, 0, 0), 2)
                
                features = extractor._extract_visual_features(test_img)
                detection_results['feature_extraction'] = {
                    'features_extracted': len(features),
                    'sample_features': list(features.keys())[:5]
                }
                
            except Exception as e:
                issues.append(f"Extractor testing failed: {e}")
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Visual detection: {len(issues)} issues found" if issues else "Visual detection capabilities working",
            'details': issues,
            'detection_results': detection_results
        }
    
    def _create_visual_test_scenarios(self) -> Dict[str, np.ndarray]:
        """Create test images for different visual content types"""
        scenarios = {}
        
        # Table-like structure
        table_img = np.zeros((200, 300, 3), dtype=np.uint8)
        table_img.fill(255)
        
        # Draw table grid
        for i in range(4):
            y = 50 + i * 30
            cv2.line(table_img, (50, y), (250, y), (0, 0, 0), 1)
        for i in range(4):
            x = 50 + i * 50
            cv2.line(table_img, (x, 50), (x, 140), (0, 0, 0), 1)
        
        scenarios['table_structure'] = table_img
        
        # Diagram-like structure
        diagram_img = np.zeros((200, 300, 3), dtype=np.uint8)
        diagram_img.fill(255)
        
        # Draw shapes and lines
        cv2.circle(diagram_img, (150, 100), 40, (0, 0, 0), 2)
        cv2.rectangle(diagram_img, (80, 60), (120, 140), (0, 0, 0), 2)
        cv2.line(diagram_img, (120, 100), (110, 100), (0, 0, 0), 2)
        cv2.arrowedLine(diagram_img, (190, 100), (240, 100), (0, 0, 0), 2)
        
        scenarios['diagram_structure'] = diagram_img
        
        # Content block
        content_img = np.zeros((200, 300, 3), dtype=np.uint8)
        content_img.fill(255)
        
        # Draw mixed content
        cv2.rectangle(content_img, (30, 30), (270, 170), (0, 0, 0), 1)
        cv2.putText(content_img, "TITLE", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(content_img, (50, 80), (250, 80), (0, 0, 0), 1)
        cv2.rectangle(content_img, (60, 100), (240, 150), (0, 0, 0), 2)
        
        scenarios['content_block'] = content_img
        
        return scenarios
    
    def _test_detection_on_image(self, test_img: np.ndarray, scenario_name: str) -> Dict:
        """Test detection methods on a specific image"""
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        results = {'detected': False, 'details': {}}
        
        # Test table detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        h_pixels = np.sum(horizontal_lines > 0)
        v_pixels = np.sum(vertical_lines > 0)
        
        results['details']['table_detection'] = {
            'horizontal_pixels': int(h_pixels),
            'vertical_pixels': int(v_pixels),
            'likely_table': h_pixels > 100 and v_pixels > 100
        }
        
        # Test edge detection
        edges = cv2.Canny(gray, 30, 100)
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / edges.size
        
        results['details']['edge_detection'] = {
            'edge_pixels': int(edge_pixels),
            'edge_density': float(edge_density),
            'has_structure': edge_density > 0.02
        }
        
        # Test contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        results['details']['contour_detection'] = {
            'total_contours': len(contours),
            'significant_contours': len(significant_contours),
            'has_objects': len(significant_contours) > 0
        }
        
        # Determine if detection worked for this scenario
        if scenario_name == 'table_structure':
            results['detected'] = results['details']['table_detection']['likely_table']
        elif scenario_name == 'diagram_structure':
            # More lenient criteria for diagram detection
            has_structure = results['details']['edge_detection']['has_structure']
            has_objects = results['details']['contour_detection']['has_objects']
            # Also check if we have reasonable edge content
            reasonable_edges = results['details']['edge_detection']['edge_pixels'] > 50
            results['detected'] = has_structure and (has_objects or reasonable_edges)
        elif scenario_name == 'content_block':
            results['detected'] = results['details']['contour_detection']['significant_contours'] > 0
        
        return results
    
    def test_training_data_quality(self) -> Dict:
        """Test training data quality and completeness"""
        labeled_path = pathlib.Path("labeled_data")
        issues = []
        training_info = {}
        
        if not labeled_path.exists():
            return {
                'status': 'fail',
                'message': "No labeled_data directory found",
                'details': ['Run: python adaptive_agent.py --setup-labeled-data'],
                'training_info': {'directories_found': 0}
            }
        
        expected_categories = ['anchors', 'design_pressure', 'glazing', 'impact_rating']
        
        for category in expected_categories:
            category_path = labeled_path / category
            category_info = {'exists': category_path.exists()}
            
            if category_path.exists():
                # Count images
                images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
                category_info['image_count'] = len(images)
                
                if len(images) == 0:
                    issues.append(f"No training images in {category}/")
                elif len(images) < 3:
                    issues.append(f"Only {len(images)} images in {category}/ (recommend 3+)")
                
                # Analyze image quality if OpenCV available
                if CV_AVAILABLE and images:
                    try:
                        quality_stats = self._analyze_training_image_quality(images[:3])
                        category_info['quality_analysis'] = quality_stats
                        
                        if quality_stats.get('avg_resolution', 0) < 50000:  # 50K pixels
                            issues.append(f"{category}/ images may be too small/low quality")
                    except Exception as e:
                        category_info['quality_analysis_error'] = str(e)
            else:
                issues.append(f"Missing category directory: {category}/")
            
            training_info[category] = category_info
        
        # Fix: Handle cases where category_info might be an int instead of dict
        total_images = 0
        categories_with_images = 0
        
        for category, info in training_info.items():
            if isinstance(info, dict) and 'image_count' in info:
                total_images += info['image_count']
                if info['image_count'] > 0:
                    categories_with_images += 1
        
        training_info['total_images'] = total_images
        training_info['categories_with_images'] = categories_with_images
        
        if total_images < 8:
            issues.append(f"Total training images ({total_images}) is quite low - recommend 12+")
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Training data: {len(issues)} issues found" if issues else f"Training data looks good ({total_images} images)",
            'details': issues,
            'training_info': training_info
        }
    
    def _analyze_training_image_quality(self, image_paths: List[pathlib.Path]) -> Dict:
        """Analyze quality of training images"""
        quality_stats = {
            'images_analyzed': 0,
            'resolutions': [],
            'avg_resolution': 0,
            'has_visual_content': 0
        }
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    resolution = h * w
                    quality_stats['resolutions'].append(resolution)
                    
                    # Quick check for visual content (not just text)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 30, 100)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    if edge_density > 0.02:  # Has some visual structure
                        quality_stats['has_visual_content'] += 1
                    
                    quality_stats['images_analyzed'] += 1
                    
            except Exception:
                continue
        
        if quality_stats['resolutions']:
            quality_stats['avg_resolution'] = sum(quality_stats['resolutions']) / len(quality_stats['resolutions'])
        
        return quality_stats
    
    def test_extraction_pipeline(self) -> Dict:
        """Test the extraction pipeline with synthetic data"""
        if not CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            return {
                'status': 'fail',
                'message': "Extraction pipeline not available",
                'details': ['characteristic_based_extractor.py not found or has errors']
            }
        
        issues = []
        pipeline_results = {}
        
        try:
            extractor = CharacteristicBasedExtractor()
            pipeline_results['extractor_created'] = True
            
            # Test characteristics loading
            characteristics = extractor.get_available_characteristics()
            pipeline_results['characteristics_loaded'] = len(characteristics)
            
            if len(characteristics) == 0:
                issues.append("No characteristics loaded")
            
            # Test with synthetic image
            if CV_AVAILABLE:
                # Create test PIL image
                test_array = np.zeros((400, 600, 3), dtype=np.uint8)
                test_array.fill(255)
                
                # Add some visual content
                cv2.rectangle(test_array, (100, 100), (500, 300), (0, 0, 0), 2)
                cv2.line(test_array, (100, 200), (500, 200), (0, 0, 0), 1)
                
                test_pil = Image.fromarray(cv2.cvtColor(test_array, cv2.COLOR_BGR2RGB))
                
                # Test extraction
                if characteristics:
                    test_char = characteristics[0]
                    extracted = extractor.extract_characteristic_content(test_pil, test_char, 1, debug=False)
                    
                    pipeline_results['test_extraction'] = {
                        'characteristic_tested': test_char,
                        'items_extracted': len(extracted),
                        'extraction_successful': len(extracted) >= 0  # Should at least not crash
                    }
                    
                    if len(extracted) > 5:
                        issues.append(f"Extraction may be too permissive ({len(extracted)} items from simple test image)")
                else:
                    issues.append("No characteristics available for testing")
            
        except Exception as e:
            issues.append(f"Pipeline test error: {e}")
            pipeline_results['error'] = str(e)
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Extraction pipeline: {len(issues)} issues found" if issues else "Pipeline working correctly",
            'details': issues,
            'pipeline_results': pipeline_results
        }
    
    def test_parameter_learning(self) -> Dict:
        """Test parameter learning and adaptation"""
        param_file = "learning_parameters.json"
        issues = []
        param_info = {}
        
        # Check if parameters file exists and is valid
        if os.path.exists(param_file):
            try:
                with open(param_file) as f:
                    params = json.load(f)
                param_info['file_exists'] = True
                param_info['parameters'] = params
                
                # Check parameter validity
                expected_params = ['confidence_threshold', 'min_region_size', 'similarity_threshold']
                
                for param in expected_params:
                    if param not in params:
                        issues.append(f"Missing parameter: {param}")
                    else:
                        value = params[param]
                        if param == 'confidence_threshold' and not (0.1 <= value <= 0.9):
                            issues.append(f"confidence_threshold out of reasonable range: {value}")
                        elif param == 'min_region_size' and not (1000 <= value <= 50000):
                            issues.append(f"min_region_size out of reasonable range: {value}")
                        elif param == 'similarity_threshold' and not (0.1 <= value <= 0.9):
                            issues.append(f"similarity_threshold out of reasonable range: {value}")
                
                # Check metadata
                if '_metadata' in params:
                    metadata = params['_metadata']
                    param_info['last_updated'] = metadata.get('last_updated', 'unknown')
                    param_info['updated_by'] = metadata.get('updated_by', 'unknown')
                
            except Exception as e:
                issues.append(f"Error loading parameters file: {e}")
                param_info['file_error'] = str(e)
        else:
            param_info['file_exists'] = False
            param_info['will_be_created'] = 'Parameters will be created automatically on first run'
        
        # Check feedback log
        feedback_log = "feedback_log.json"
        if os.path.exists(feedback_log):
            try:
                with open(feedback_log) as f:
                    logs = json.load(f)
                param_info['feedback_logs'] = len(logs)
                
                if logs:
                    recent_log = logs[-1]
                    param_info['last_feedback'] = {
                        'timestamp': recent_log.get('timestamp', 'unknown'),
                        'method': recent_log.get('analysis_method', 'unknown')
                    }
            except Exception as e:
                issues.append(f"Error reading feedback log: {e}")
        else:
            param_info['feedback_logs'] = 0
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Parameter learning: {len(issues)} issues found" if issues else "Parameter learning system ready",
            'details': issues,
            'param_info': param_info
        }
    
    def test_azure_openai(self) -> Dict:
        """Test Azure OpenAI connection and capabilities"""
        if not LANGCHAIN_AVAILABLE:
            return {
                'status': 'fail',
                'message': "LangChain not available for Azure OpenAI testing",
                'details': ['Install with: pip install langchain-openai']
            }
        
        # Load environment variables like the main system does
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except:
            pass
        
        issues = []
        azure_info = {}
        
        # Check environment variables
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return {
                'status': 'fail',
                'message': "Azure OpenAI not configured",
                'details': [f"Missing environment variables: {missing_vars}"],
                'azure_info': {'configured': False, 'missing_vars': missing_vars}
            }
        
        azure_info['configured'] = True
        azure_info['endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT', '').replace('https://', '').split('.')[0]
        azure_info['deployment'] = os.getenv('AZURE_OPENAI_DEPLOYMENT', '')
        
        # Test connection
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version="2024-02-15-preview",
                temperature=0.1,
                max_tokens=100
            )
            
            # Simple test
            test_message = [HumanMessage(content="Test connection - respond with 'OK'")]
            response = llm.invoke(test_message)
            
            azure_info['connection_test'] = 'successful'
            azure_info['response_length'] = len(response.content)
            
            if 'ok' not in response.content.lower():
                issues.append("Azure OpenAI response seems unusual")
            
        except Exception as e:
            issues.append(f"Azure OpenAI connection failed: {e}")
            azure_info['connection_test'] = 'failed'
            azure_info['error'] = str(e)[:200]
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Azure OpenAI: {len(issues)} issues found" if issues else "Azure OpenAI working correctly",
            'details': issues,
            'azure_info': azure_info
        }
    
    def test_recent_extractions(self) -> Dict:
        """Test recent extraction results"""
        feedback_dir = pathlib.Path("feedback_data")
        issues = []
        extraction_info = {}
        
        if not feedback_dir.exists():
            return {
                'status': 'pass',  # Not an error if no extractions yet
                'message': "No extractions found (run some extractions first)",
                'details': ['Run: python adaptive_agent.py --source document.pdf --characteristic anchors'],
                'extraction_info': {'extractions_found': 0}
            }
        
        # Find recent extraction files
        extraction_files = list(feedback_dir.glob("extraction_*.json"))
        extraction_info['extractions_found'] = len(extraction_files)
        
        if len(extraction_files) == 0:
            extraction_info['no_files'] = True
            return {
                'status': 'pass',
                'message': "No extraction files found yet",
                'extraction_info': extraction_info
            }
        
        # Analyze recent extractions
        recent_extractions = []
        for file_path in extraction_files[-5:]:  # Last 5
            try:
                with open(file_path) as f:
                    data = json.load(f)
                recent_extractions.append(data)
            except Exception as e:
                issues.append(f"Error reading {file_path.name}: {e}")
        
        if recent_extractions:
            # Analyze extraction quality
            total_items = sum(ex.get('total_sections', 0) for ex in recent_extractions)
            avg_items = total_items / len(recent_extractions)
            
            extraction_info['recent_extractions_analyzed'] = len(recent_extractions)
            extraction_info['total_items_extracted'] = total_items
            extraction_info['avg_items_per_extraction'] = round(avg_items, 1)
            
            # Check for potential issues
            zero_extraction_count = sum(1 for ex in recent_extractions if ex.get('total_sections', 0) == 0)
            if zero_extraction_count > 0:
                issues.append(f"{zero_extraction_count}/{len(recent_extractions)} recent extractions found no items")
            
            high_extraction_count = sum(1 for ex in recent_extractions if ex.get('total_sections', 0) > 30)
            if high_extraction_count > 0:
                issues.append(f"{high_extraction_count}/{len(recent_extractions)} recent extractions found >30 items (may be too permissive)")
            
            # Check processing times
            processing_times = [ex.get('processing_time', 0) for ex in recent_extractions]
            avg_time = sum(processing_times) / len(processing_times)
            extraction_info['avg_processing_time'] = round(avg_time, 1)
            
            if avg_time > 180:  # 3 minutes
                issues.append(f"Average processing time is high ({avg_time:.1f}s)")
            
            # Check for enhanced features
            enhanced_count = sum(1 for ex in recent_extractions 
                               if ex.get('processing_method') == 'enhanced_visual_content_extraction')
            extraction_info['enhanced_extractions'] = enhanced_count
            
            if enhanced_count < len(recent_extractions):
                issues.append(f"Some extractions not using enhanced visual detection ({enhanced_count}/{len(recent_extractions)})")
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"Recent extractions: {len(issues)} issues found" if issues else f"Recent extractions look good (analyzed {len(recent_extractions)})",
            'details': issues,
            'extraction_info': extraction_info
        }
    
    def test_system_performance(self) -> Dict:
        """Test overall system performance"""
        issues = []
        performance_info = {}
        
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            performance_info['free_disk_gb'] = round(free_gb, 1)
            
            if free_gb < 1:
                issues.append(f"Low disk space: {free_gb:.1f} GB free")
        except Exception:
            pass
        
        # Check memory if psutil available
        try:
            import psutil
            memory = psutil.virtual_memory()
            performance_info['memory_available_gb'] = round(memory.available / (1024**3), 1)
            performance_info['memory_percent_used'] = memory.percent
            
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
        except ImportError:
            performance_info['memory_info'] = 'psutil not available'
        
        # Check file sizes
        large_files = []
        for file_path in ['feedback_log.json', 'learning_parameters.json']:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024**2)
                if size_mb > 10:  # 10MB
                    large_files.append(f"{file_path}: {size_mb:.1f}MB")
        
        if large_files:
            issues.append(f"Large files found: {', '.join(large_files)}")
        
        # Check feedback_data directory size
        feedback_dir = pathlib.Path("feedback_data")
        if feedback_dir.exists():
            total_size = sum(f.stat().st_size for f in feedback_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024**2)
            performance_info['feedback_data_size_mb'] = round(size_mb, 1)
            
            if size_mb > 500:  # 500MB
                issues.append(f"feedback_data directory is large: {size_mb:.1f}MB")
        
        # Performance recommendations
        recommendations = []
        if len(issues) == 0:
            recommendations.append("System performance looks good")
        else:
            recommendations.append("Consider cleaning up large files periodically")
            recommendations.append("Monitor memory usage during large document processing")
        
        performance_info['recommendations'] = recommendations
        
        status = 'pass' if not issues else 'fail'
        return {
            'status': status,
            'message': f"System performance: {len(issues)} issues found" if issues else "System performance good",
            'details': issues,
            'performance_info': performance_info
        }
    
    def _generate_diagnostic_summary(self) -> Dict:
        """Generate comprehensive diagnostic summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result.get('status') == 'pass')
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"🎯 ENHANCED DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")
        print(f"Tests run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        print(f"Duration: {duration:.1f}s")
        
        # Show critical issues
        critical_issues = []
        for test_name, result in self.results.items():
            if result.get('status') == 'fail' and test_name in ['Dependencies', 'Core Files', 'Visual Detection']:
                critical_issues.append(f"{test_name}: {result.get('message', 'Failed')}")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"  • {issue}")
        
        # Show recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if passed_tests == total_tests:
            print("  • System is ready for enhanced visual content extraction")
            print("  • Add training images to labeled_data/ directories for better accuracy")
            print("  • Configure Azure OpenAI for vision feedback (optional)")
        else:
            if self.results.get('Dependencies', {}).get('status') == 'fail':
                print("  • Install missing dependencies first")
            if self.results.get('Training Data', {}).get('status') == 'fail':
                print("  • Set up training data: python adaptive_agent.py --setup-labeled-data")
            if self.results.get('Azure OpenAI', {}).get('status') == 'fail':
                print("  • Configure Azure OpenAI in .env file (optional but recommended)")
            if self.results.get('Visual Detection', {}).get('status') == 'fail':
                print("  • Check OpenCV installation and functionality")
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests/total_tests,
                'duration_seconds': duration,
                'critical_issues': critical_issues
            },
            'detailed_results': self.results,
            'timestamp': end_time.isoformat(),
            'diagnostic_version': '2.0_enhanced'
        }

def main():
    parser = argparse.ArgumentParser(description="Enhanced Diagnostic Tool for Visual Content Extraction")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostic (core tests only)")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive diagnostic")
    parser.add_argument("--test", choices=['dependencies', 'training-data', 'azure', 'visual-detection', 'pipeline'], 
                       help="Run specific test only")
    parser.add_argument("--save-report", action="store_true", help="Save diagnostic report to file")
    
    args = parser.parse_args()
    
    diagnostic = EnhancedDiagnostic()
    
    if args.test:
        # Run specific test only
        test_map = {
            'dependencies': diagnostic.test_dependencies,
            'training-data': diagnostic.test_training_data_quality,
            'azure': diagnostic.test_azure_openai,
            'visual-detection': diagnostic.test_visual_detection_capabilities,
            'pipeline': diagnostic.test_extraction_pipeline
        }
        
        test_func = test_map.get(args.test)
        if test_func:
            print(f"🧪 Running {args.test.replace('-', ' ').title()} Test...")
            result = test_func()
            
            status = "✅ PASS" if result.get('status') == 'pass' else "❌ FAIL"
            print(f"\n{status} {result.get('message', 'No message')}")
            
            if result.get('details'):
                print("\nDetails:")
                for detail in result['details']:
                    print(f"  • {detail}")
        
        return 0
    
    elif args.quick:
        # Quick diagnostic - core tests only
        print("🧪 QUICK DIAGNOSTIC")
        print("="*40)
        
        quick_tests = [
            ("Dependencies", diagnostic.test_dependencies),
            ("Core Files", diagnostic.test_core_files),
            ("Training Data", diagnostic.test_training_data_quality),
            ("Visual Detection", diagnostic.test_visual_detection_capabilities)
        ]
        
        for test_name, test_func in quick_tests:
            print(f"\n🧪 Testing {test_name}...")
            result = test_func()
            
            status = "✅ PASS" if result.get('status') == 'pass' else "❌ FAIL"
            print(f"   {status} {result.get('message', 'No message')}")
    
    else:
        # Full comprehensive diagnostic
        results = diagnostic.run_comprehensive_diagnostic()
        
        if args.save_report:
            # Save diagnostic report
            report_dir = pathlib.Path("diagnostic_output")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"diagnostic_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Diagnostic report saved: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())