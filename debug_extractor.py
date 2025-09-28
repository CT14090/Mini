#!/usr/bin/env python3
# debug_extractor.py
"""
Debug script to check what's available in characteristic_based_extractor.py
"""

import sys
import os

def debug_extractor_module():
    """Debug the extractor module to see what's available"""
    print("Debugging characteristic_based_extractor.py...")
    print("=" * 50)
    
    # Check if file exists
    file_path = "characteristic_based_extractor.py"
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} does not exist")
        return
    
    print(f"✅ File {file_path} exists")
    
    # Try to import the module
    try:
        import characteristic_based_extractor
        print("✅ Module imported successfully")
        
        # Check what's available in the module
        available_items = dir(characteristic_based_extractor)
        print(f"\nAvailable items in module:")
        
        classes = []
        functions = []
        variables = []
        
        for item in available_items:
            if not item.startswith('_'):  # Skip private items
                obj = getattr(characteristic_based_extractor, item)
                if isinstance(obj, type):
                    classes.append(item)
                elif callable(obj):
                    functions.append(item)
                else:
                    variables.append(item)
        
        if classes:
            print(f"\nClasses found:")
            for cls in classes:
                print(f"  - {cls}")
        
        if functions:
            print(f"\nFunctions found:")
            for func in functions:
                print(f"  - {func}")
        
        if variables:
            print(f"\nVariables found:")
            for var in variables:
                print(f"  - {var}")
        
        # Specifically check for expected classes
        expected_classes = ['AzureFirstExtractor', 'CharacteristicBasedExtractor']
        print(f"\nChecking for expected classes:")
        for expected in expected_classes:
            if expected in classes:
                print(f"  ✅ {expected} - FOUND")
            else:
                print(f"  ❌ {expected} - NOT FOUND")
        
        # Try specific imports
        print(f"\nTesting specific imports:")
        
        try:
            from characteristic_based_extractor import AzureFirstExtractor
            print("  ✅ AzureFirstExtractor import successful")
        except ImportError as e:
            print(f"  ❌ AzureFirstExtractor import failed: {e}")
        
        try:
            from characteristic_based_extractor import CharacteristicBasedExtractor
            print("  ✅ CharacteristicBasedExtractor import successful")
        except ImportError as e:
            print(f"  ❌ CharacteristicBasedExtractor import failed: {e}")
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        
        # Try to read the file and check for syntax errors
        print(f"\nChecking file for syntax issues...")
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, file_path, 'exec')
            print("  ✅ File syntax is valid")
            
            # Check for class definitions
            lines = content.split('\n')
            class_definitions = []
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    class_name = line.strip().split()[1].split('(')[0].rstrip(':')
                    class_definitions.append((i+1, class_name))
            
            if class_definitions:
                print(f"\nClass definitions found in file:")
                for line_num, class_name in class_definitions:
                    print(f"  Line {line_num}: {class_name}")
            else:
                print(f"\n❌ No class definitions found in file")
                
        except SyntaxError as se:
            print(f"  ❌ Syntax error in file: {se}")
        except Exception as fe:
            print(f"  ❌ Error reading file: {fe}")

if __name__ == "__main__":
    debug_extractor_module()