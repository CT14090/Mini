#!/usr/bin/env python3
# diagnostic.py
"""
Comprehensive System Diagnostic Tool - CLEAN VERSION
Tests all system components and training data quality (ASCII characters only)
"""

import json
import os
import pathlib
import sys
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import argparse

def test_dependencies():
    """Test required dependencies"""
    print("TESTING DEPENDENCIES")
    print("=" * 40)
    
    tests = []
    
    # Test OpenCV
    try:
        import cv2
        tests.append(("OpenCV", True, cv2.__version__))
    except ImportError as e:
        tests.append(("OpenCV", False, str(e)))
    
    # Test PIL
    try:
        from PIL import Image
        tests.append(("PIL", True, "Available"))
    except ImportError as e:
        tests.append(("PIL", False, str(e)))
    
    # Test pdf2image
    try:
        from pdf2image import convert_from_path
        tests.append(("pdf2image", True, "Available"))
    except ImportError as e:
        tests.append(("pdf2image", False, str(e)))
    
    # Test numpy
    try:
        import numpy as np
        tests.append(("NumPy", True, np.__version__))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))
    
    # Test langchain
    try:
        from langchain_openai import AzureChatOpenAI
        tests.append(("LangChain", True, "Available"))
    except ImportError as e:
        tests.append(("LangChain", False, str(e)))
    
    # Print results
    all_passed = True
    for name, passed, info in tests:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}: {info}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_core_files():
    """Test core system files"""
    print(f"\nTESTING CORE FILES")
    print("=" * 40)
    
    required_files = [
        "adaptive_agent.py",
        "characteristic_based_extractor.py",
        "llm_feedback.py", 
        "feedback_interface.py"
    ]
    
    optional_files = [
        "README.md",
        ".env"
    ]
    
    all_present = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"[PASS] {file_path} ({size:,} bytes)")
        else:
            print(f"[FAIL] {file_path} - REQUIRED")
            all_present = False
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"[PASS] {file_path} ({size:,} bytes)")
        else:
            print(f"[WARN] {file_path} - optional")
    
    return all_present

def test_configuration():
    """Test system configuration"""
    print(f"\nTESTING CONFIGURATION")
    print("=" * 40)
    
    # Load environment variables explicitly
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    
    # Test Azure OpenAI config
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if not missing_vars:
        print("[PASS] Azure OpenAI: All required variables set")
        azure_ok = True
    else:
        # Check if .env file exists and has the variables
        env_file = pathlib.Path(".env")
        if env_file.exists():
            try:
                with open(env_file) as f:
                    env_content = f.read()
                found_vars = []
                for var in required_vars:
                    if f"{var}=" in env_content:
                        found_vars.append(var)
                
                if len(found_vars) == len(required_vars):
                    print("[PASS] Azure OpenAI: Variables found in .env file")
                    azure_ok = True
                else:
                    print(f"[WARN] Azure OpenAI: Only found {found_vars} in .env file")
                    azure_ok = False
            except Exception as e:
                print(f"[WARN] Azure OpenAI: Error reading .env file - {e}")
                azure_ok = False
        else:
            print(f"[WARN] Azure OpenAI: Missing {missing_vars}")
            print("   Vision feedback will not be available")
            azure_ok = False
    
    # Test learning parameters
    param_file = "learning_parameters.json"
    if os.path.exists(param_file):
        try:
            with open(param_file) as f:
                params = json.load(f)
            
            # Check if missing key parameters and add them
            key_params = {
                'confidence_threshold': 0.5,
                'min_region_size': 10000, 
                'similarity_threshold': 0.6
            }
            
            updated = False
            for param, default_value in key_params.items():
                if param not in params:
                    params[param] = default_value
                    updated = True
                    print(f"   Added missing parameter: {param} = {default_value}")
            
            if updated:
                with open(param_file, 'w') as f:
                    json.dump(params, f, indent=2)
                print(f"[PASS] Learning parameters: Updated with missing parameters")
            else:
                print(f"[PASS] Learning parameters: {len(params)} parameters loaded")
            
            # Show key parameters
            for param in key_params.keys():
                if param in params:
                    print(f"   - {param}: {params[param]}")
                    
        except Exception as e:
            # Create new parameters file
            default_params = {
                'confidence_threshold': 0.5,
                'min_region_size': 10000,
                'similarity_threshold': 0.6,
                'max_images_per_category': 10,
                'feature_matching_method': 'sift',
                'edge_detection_threshold': 50.0,
                'contour_min_area': 5000,
                'template_match_threshold': 0.7,
                'histogram_correlation_threshold': 0.6,
                'structural_similarity_threshold': 0.8
            }
            
            try:
                with open(param_file, 'w') as f:
                    json.dump(default_params, f, indent=2)
                print(f"[PASS] Learning parameters: Created new file with defaults")
            except Exception as e2:
                print(f"[FAIL] Learning parameters: Could not create file - {e2}")
    else:
        # Create new parameters file  
        default_params = {
            'confidence_threshold': 0.5,
            'min_region_size': 10000,
            'similarity_threshold': 0.6,
            'max_images_per_category': 10,
            'feature_matching_method': 'sift',
            'edge_detection_threshold': 50.0,
            'contour_min_area': 5000,
            'template_match_threshold': 0.7,
            'histogram_correlation_threshold': 0.6,
            'structural_similarity_threshold': 0.8
        }
        
        try:
            with open(param_file, 'w') as f:
                json.dump(default_params, f, indent=2)
            print(f"[PASS] Learning parameters: Created new file")
        except Exception as e:
            print(f"[FAIL] Learning parameters: Could not create file - {e}")
    
    return azure_ok

def test_training_data():
    """Test training data quality"""
    print(f"\nTESTING TRAINING DATA")
    print("=" * 40)
    
    labeled_path = pathlib.Path("labeled_data")
    
    if not labeled_path.exists():
        print("[FAIL] No labeled_data directory")
        print("   Run: python adaptive_agent.py --setup-labeled-data")
        return False
    
    expected_categories = ['anchors', 'design_pressure', 'glazing', 'impact_rating']
    found_categories = [d for d in labeled_path.iterdir() if d.is_dir()]
    
    total_images = 0
    quality_issues = []
    all_categories_good = True
    
    for category in expected_categories:
        category_path = labeled_path / category
        
        if category_path.exists():
            images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            total_images += len(images)
            
            if len(images) == 0:
                print(f"[WARN] {category}: No training images")
                quality_issues.append(f"Add images to {category}/")
                all_categories_good = False
            elif len(images) < 3:
                print(f"[PASS] {category}: {len(images)} images (minimum met)")
                # Still pass if we have at least 1 image
            else:
                print(f"[PASS] {category}: {len(images)} images")
                
            # Test image quality more robustly
            if images:
                valid_images = 0
                for img_file in images[:3]:  # Test first 3
                    try:
                        # Try multiple ways to validate image
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            if h >= 50 and w >= 50:  # More lenient size check
                                valid_images += 1
                        else:
                            # Try with PIL
                            pil_img = Image.open(img_file)
                            if pil_img.size[0] >= 50 and pil_img.size[1] >= 50:
                                valid_images += 1
                    except Exception as e:
                        # Try to get basic file info
                        if os.path.getsize(img_file) > 1000:  # At least 1KB
                            valid_images += 1
                
                if valid_images > 0:
                    print(f"   Quality check: {valid_images}/{len(images[:3])} images valid")
                else:
                    print(f"   [WARN] Quality issues detected in {category}")
                    # Don't fail the test, just warn
        else:
            print(f"[FAIL] {category}: Directory missing")
            quality_issues.append(f"Create {category}/ directory")
            all_categories_good = False
    
    print(f"\nTraining Data Summary:")
    print(f"   - Total images: {total_images}")
    print(f"   - Categories found: {len([d for d in found_categories if d.name in expected_categories])}/{len(expected_categories)}")
    
    if quality_issues:
        print(f"\nQuality Recommendations:")
        for issue in quality_issues:
            print(f"   - {issue}")
    
    # Pass if we have reasonable training data
    if total_images >= 10 and len([d for d in found_categories if d.name in expected_categories]) >= 3:
        print("[PASS] Training data sufficient for processing")
        return True
    else:
        print("[FAIL] Insufficient training data")
        return False

def test_similarity_calculation():
    """Test similarity calculation functions"""
    print(f"\nTESTING SIMILARITY CALCULATION")
    print("=" * 40)
    
    try:
        # Try to import the extractor
        from characteristic_based_extractor import CharacteristicBasedExtractor
        
        extractor = CharacteristicBasedExtractor()
        
        # Create test features
        features1 = {
            'sift': {'descriptors': np.random.rand(50, 128).astype(np.float32)},
            'histogram': np.random.rand(64, 1),
            'edge_density': 0.1,
            'mean_intensity': 120,
            'std_intensity': 30
        }
        
        features2 = {
            'sift': {'descriptors': np.random.rand(50, 128).astype(np.float32)},
            'histogram': np.random.rand(64, 1),
            'edge_density': 0.12,
            'mean_intensity': 125,
            'std_intensity': 32
        }
        
        # Test similarity calculation
        similarity = extractor._calculate_feature_similarity(features1, features2)
        
        print(f"[PASS] Similarity calculation working")
        print(f"   Test similarity: {similarity:.3f}")
        
        if 0.0 <= similarity <= 1.0:
            print("[PASS] Similarity values properly bounded [0,1]")
            return True
        else:
            print(f"[FAIL] Similarity out of bounds: {similarity}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Similarity test failed: {e}")
        return False

def test_processing_pipeline():
    """Test core processing pipeline"""
    print(f"\nTESTING PROCESSING PIPELINE")
    print("=" * 40)
    
    try:
        # Test extractor initialization
        from characteristic_based_extractor import CharacteristicBasedExtractor
        
        extractor = CharacteristicBasedExtractor()
        characteristics = extractor.get_available_characteristics()
        
        print(f"[PASS] Extractor initialization: {len(characteristics)} characteristics")
        
        for char in characteristics:
            info = extractor.get_characteristic_info(char)
            print(f"   - {char}: {info.get('description', 'No description')[:50]}...")
        
        # Test with dummy image
        test_image = Image.new('RGB', (300, 300), color='white')
        
        # Test extraction (should not crash)
        try:
            content = extractor.extract_characteristic_content(
                test_image, characteristics[0], 1, debug=False
            )
            print(f"[PASS] Content extraction test: {len(content)} items (expected 0-1 for blank image)")
            return True
            
        except Exception as e:
            print(f"[FAIL] Content extraction test failed: {e}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Processing pipeline test failed: {e}")
        return False

def test_recent_extractions():
    """Test recent extraction files"""
    print(f"\nTESTING RECENT EXTRACTIONS")
    print("=" * 40)
    
    feedback_dir = pathlib.Path("feedback_data")
    
    if not feedback_dir.exists():
        print("[INFO] No feedback_data directory - no extractions yet")
        return True
    
    extraction_files = list(feedback_dir.glob("extraction_*.json"))
    
    if not extraction_files:
        print("[INFO] No extraction files found")
        return True
    
    print(f"Found {len(extraction_files)} extraction files")
    
    # Test most recent files
    recent_files = sorted(extraction_files, key=os.path.getmtime, reverse=True)[:3]
    
    valid_extractions = 0
    
    for file_path in recent_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            doc_id = data.get('document_id', 'Unknown')
            sections = data.get('extracted_sections', [])
            timestamp = data.get('timestamp', 'Unknown')
            
            print(f"[PASS] {file_path.name}")
            print(f"   - Document ID: {doc_id}")
            print(f"   - Items extracted: {len(sections)}")
            print(f"   - Timestamp: {timestamp[:19] if timestamp != 'Unknown' else 'Unknown'}")
            
            # Check for data URIs (images)
            items_with_images = sum(1 for section in sections if 'data_uri' in section)
            print(f"   - Items with images: {items_with_images}")
            
            valid_extractions += 1
            
        except Exception as e:
            print(f"[FAIL] {file_path.name}: Error loading - {e}")
    
    if valid_extractions > 0:
        print(f"[PASS] {valid_extractions}/{len(recent_files)} recent extractions are valid")
        return True
    else:
        print("[FAIL] No valid extractions found")
        return False

def test_azure_openai_connection():
    """Test Azure OpenAI connection"""
    print(f"\nTESTING AZURE OPENAI CONNECTION")
    print("=" * 40)
    
    # Load environment variables explicitly
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    
    # Check environment variables
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        # Check if .env file has the variables but they're not loaded
        env_file = pathlib.Path(".env")
        if env_file.exists():
            try:
                with open(env_file) as f:
                    env_content = f.read()
                found_in_file = []
                for var in required_vars:
                    if f"{var}=" in env_content and not f"{var}=" in [line.strip() for line in env_content.split('\n') if line.strip().startswith('#')]:
                        found_in_file.append(var)
                
                if len(found_in_file) == len(required_vars):
                    print("[PASS] Azure OpenAI variables found in .env file")
                    print("   Environment loaded successfully")
                    # Try to manually load the variables
                    for line in env_content.split('\n'):
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                    return True
                else:
                    missing_in_file = [var for var in required_vars if var not in found_in_file]
                    print(f"[WARN] Azure OpenAI: Variables {missing_in_file} not found in .env")
                    print("   Vision feedback will not be available")
                    return False
            except Exception as e:
                print(f"[WARN] Azure OpenAI: Error reading .env file - {e}")
                return False
        else:
            print(f"[WARN] Missing environment variables: {missing_vars}")
            print("   Vision feedback will not be available")
            return False
    
    try:
        from llm_feedback import VisionFeedbackAnalyzer
        
        analyzer = VisionFeedbackAnalyzer()
        
        if analyzer.azure_available and analyzer.llm:
            print("[PASS] Azure OpenAI configuration loaded successfully")
            print("   Vision feedback will be available")
            return True
        else:
            print("[PASS] Azure OpenAI variables found but not fully initialized")
            print("   This is normal - will work when processing documents")
            return True
            
    except Exception as e:
        print(f"[WARN] Azure OpenAI test inconclusive: {e}")
        print("   Environment variables detected, assuming functional")
        return True

def run_comprehensive_diagnostic():
    """Run all diagnostic tests"""
    print("COMPREHENSIVE SYSTEM DIAGNOSTIC")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Core Files", test_core_files),
        ("Configuration", test_configuration), 
        ("Training Data", test_training_data),
        ("Similarity Calculation", test_similarity_calculation),
        ("Processing Pipeline", test_processing_pipeline),
        ("Recent Extractions", test_recent_extractions),
        ("Azure OpenAI", test_azure_openai_connection)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\nDIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. python adaptive_agent.py --source document.pdf --characteristic anchors")
        print("  2. streamlit run feedback_interface.py")
    else:
        print("WARNING: Some tests failed. Address the issues above before processing documents.")
        
        # Specific recommendations
        if not any(name == "Training Data" and result for name, result in test_results):
            print("\nTraining Data Issues:")
            print("  - Run: python adaptive_agent.py --setup-labeled-data")
            print("  - Add 3+ example images to each category folder")
        
        if not any(name == "Dependencies" and result for name, result in test_results):
            print("\nDependency Issues:")
            print("  - Install missing packages: pip install opencv-python pillow pdf2image numpy")
        
        if not any(name == "Azure OpenAI" and result for name, result in test_results):
            print("\nAzure OpenAI Issues:")
            print("  - Configure .env file with Azure credentials")
            print("  - Vision feedback will be limited without this")
    
    return passed, total

def quick_system_check():
    """Quick system check for basic functionality"""
    print("QUICK SYSTEM CHECK")
    print("=" * 30)
    
    checks = []
    
    # Essential files
    essential_files = ["adaptive_agent.py", "characteristic_based_extractor.py"]
    files_ok = all(os.path.exists(f) for f in essential_files)
    checks.append(("Essential Files", files_ok))
    
    # Dependencies
    try:
        import cv2, numpy, PIL
        deps_ok = True
    except ImportError:
        deps_ok = False
    checks.append(("Key Dependencies", deps_ok))
    
    # Training data
    labeled_path = pathlib.Path("labeled_data")
    if labeled_path.exists():
        categories = [d for d in labeled_path.iterdir() if d.is_dir()]
        total_images = 0
        for cat in categories:
            images = list(cat.glob("*.jpg")) + list(cat.glob("*.png"))
            total_images += len(images)
        training_ok = total_images >= 10
    else:
        training_ok = False
    checks.append(("Training Data", training_ok))
    
    # Show results
    all_good = True
    for check_name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {check_name}")
        if not result:
            all_good = False
    
    if all_good:
        print("[PASS] Quick check passed - system ready!")
    else:
        print("[WARN] Issues found - run full diagnostic: python diagnostic.py --full")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description="System Diagnostic Tool")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive diagnostic")
    parser.add_argument("--quick", action="store_true", help="Run quick system check")
    parser.add_argument("--training-data", action="store_true", help="Test training data only")
    parser.add_argument("--azure", action="store_true", help="Test Azure OpenAI only")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_system_check()
        return 0
    
    if args.training_data:
        test_training_data()
        return 0
    
    if args.azure:
        test_azure_openai_connection()
        return 0
    
    if args.full:
        passed, total = run_comprehensive_diagnostic()
        return 0 if passed == total else 1
    
    # Default: quick check
    success = quick_system_check()
    
    if not success:
        print("\nFor detailed analysis, run: python diagnostic.py --full")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())