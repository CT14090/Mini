#!/usr/bin/env python3
# fix_similarity_precise.py
"""
More precise fix for similarity calculation with proper indentation
"""

import os
import re

def fix_similarity_method():
    """Fix the similarity calculation method with proper indentation"""
    
    # Read the backup file
    backup_file = "characteristic_based_extractor_backup_20250920_230943.py"
    if not os.path.exists(backup_file):
        print("Backup file not found!")
        return False
    
    with open(backup_file, 'r') as f:
        content = f.read()
    
    # Find and replace the _calculate_similarity_to_characteristic method
    pattern = r'(    def _calculate_similarity_to_characteristic\(self, region_features: Dict, characteristic: str\) -> float:.*?)(    def|\Z)'
    
    new_method = '''    def _calculate_similarity_to_characteristic(self, region_features: Dict, characteristic: str) -> float:
        """Calculate similarity using VISUAL template matching"""
        if characteristic not in self.training_data or not region_features:
            return 0.0
        
        training_examples = self.training_data[characteristic]
        if not training_examples:
            return 0.0
        
        # Get the current CV region for template matching
        cv_region = getattr(self, '_current_cv_region', None)
        if cv_region is None:
            return 0.0
        
        similarities = []
        
        for training_example in training_examples:
            training_image = training_example.get('image')
            if training_image is None:
                continue
            
            # Use template matching
            template_sim = self._template_matching_similarity(cv_region, training_image)
            similarities.append(template_sim)
        
        return max(similarities) if similarities else 0.0
    
    def _template_matching_similarity(self, region_image, training_image) -> float:
        """Use OpenCV template matching for visual similarity"""
        try:
            # Resize both images to same size
            target_size = (200, 200)
            region_resized = cv2.resize(region_image, target_size)
            training_resized = cv2.resize(training_image, target_size)
            
            # Convert to grayscale
            region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
            training_gray = cv2.cvtColor(training_resized, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(region_gray, training_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # Ensure [0,1] range
            return max(0.0, min(1.0, max_val))
            
        except Exception:
            return 0.0

    '''
    
    # Replace the method using regex
    new_content = re.sub(pattern, new_method + r'\2', content, flags=re.DOTALL)
    
    # Also fix the classify method to set _current_cv_region
    classify_pattern = r'(    def _classify_region_for_characteristic\(self, region_data: Dict, characteristic: str, debug: bool = False\) -> Optional\[Dict\]:.*?)(        # Extract features from region)'
    
    classify_replacement = r'''\1        # Store CV region for template matching
        self._current_cv_region = region_data.get('cv_image')
        
\2'''
    
    new_content = re.sub(classify_pattern, classify_replacement, new_content, flags=re.DOTALL)
    
    # Also update the threshold in classify method
    new_content = new_content.replace(
        'confidence_threshold = 0.85  # Raised from 0.7 to 0.85',
        'confidence_threshold = 0.3  # Lower threshold for template matching'
    )
    
    # Write the fixed content
    with open("characteristic_based_extractor.py", 'w') as f:
        f.write(new_content)
    
    print("Fixed similarity calculation with proper indentation")
    return True

if __name__ == "__main__":
    print("APPLYING PRECISE SIMILARITY FIX")
    print("=" * 40)
    
    success = fix_similarity_method()
    
    if success:
        print("SUCCESS: Applied visual similarity fix")
        print("Test with: python adaptive_agent.py --source document.pdf --characteristic glazing --debug")
    else:
        print("FAILED: Could not apply fix")