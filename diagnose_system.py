#!/usr/bin/env python3
"""
System Diagnosis Script for Window Characteristic Extraction
Checks Azure OpenAI, reference data, and system integration
"""
import os
import pathlib
import json
from dotenv import load_dotenv

def diagnose_system():
    """Run complete system diagnosis"""
    print("🔍 WINDOW CHARACTERISTIC SYSTEM DIAGNOSIS")
    print("="*60)
    
    # 1. Environment Variables Check
    print("\n1. ENVIRONMENT VARIABLES")
    print("-" * 30)
    
    load_dotenv(override=True)
    
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_DEPLOYMENT': os.getenv('AZURE_OPENAI_DEPLOYMENT')
    }
    
    env_ok = True
    for var, value in required_vars.items():
        if value and len(str(value).strip()) > 0:
            if 'KEY' in var:
                print(f"✅ {var}: {str(value)[:8]}...{str(value)[-4:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            env_ok = False
    
    if not env_ok:
        print("\n💡 To fix: Create/update .env file with your Azure OpenAI credentials")
    
    # 2. Azure OpenAI Connection Test
    print("\n2. AZURE OPENAI CONNECTION")
    print("-" * 30)
    
    if env_ok:
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = AzureChatOpenAI(
                azure_endpoint=required_vars['AZURE_OPENAI_ENDPOINT'],
                api_key=required_vars['AZURE_OPENAI_API_KEY'],
                azure_deployment=required_vars['AZURE_OPENAI_DEPLOYMENT'],
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=10
            )
            
            response = llm.invoke([HumanMessage(content="Reply 'OK' if working.")])
            
            if response and response.content:
                print(f"✅ Connection successful: {response.content}")
            else:
                print("❌ No response received")
                
        except ImportError:
            print("❌ LangChain not installed")
            print("💡 Fix: pip install langchain-openai langchain-core")
        except Exception as e:
            print(f"❌ Connection failed: {str(e)[:100]}")
    else:
        print("⏭️  Skipping (environment variables not set)")
    
    # 3. Reference Data Check
    print("\n3. REFERENCE DATA")
    print("-" * 30)
    
    labeled_data_path = pathlib.Path("labeled_data")
    if not labeled_data_path.exists():
        print("❌ labeled_data directory not found")
        print("💡 Fix: python adaptive_agent.py --setup-reference-data")
        return
    
    characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
    total_images = 0
    total_descriptions = 0
    
    for char in characteristics:
        char_path = labeled_data_path / char
        if char_path.exists():
            # Count images
            image_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_count += len(list(char_path.glob(ext)))
            
            # Check descriptions
            desc_file = char_path / "descriptions.json"
            desc_count = 0
            if desc_file.exists():
                try:
                    with open(desc_file) as f:
                        descriptions = json.load(f)
                        desc_count = len(descriptions)
                except:
                    pass
            
            total_images += image_count
            total_descriptions += desc_count
            
            status = "✅" if image_count > 0 else "⚠️"
            print(f"{status} {char}: {image_count} images, {desc_count} descriptions")
        else:
            print(f"❌ {char}: Directory not found")
    
    if total_images == 0:
        print("\n⚠️  WARNING: No reference images found!")
        print("💡 Image matching will not work without reference data")
        print("💡 Add JPG/PNG files to labeled_data/ subdirectories")
    
    # 4. Extraction Data Check
    print("\n4. EXTRACTION DATA")
    print("-" * 30)
    
    feedback_dir = pathlib.Path("feedback_data")
    if feedback_dir.exists():
        extraction_files = list(feedback_dir.glob("*_extraction_*.json"))
        feedback_files = []
        
        for char in characteristics:
            char_feedback = pathlib.Path(f"feedback_log_{char}.json")
            if char_feedback.exists():
                feedback_files.append(char_feedback)
        
        print(f"✅ Extraction files: {len(extraction_files)}")
        print(f"✅ Feedback files: {len(feedback_files)}")
        
        if extraction_files:
            print("Recent extractions:")
            for f in extraction_files[-3:]:
                print(f"   • {f.name}")
    else:
        print("❌ No extraction data found")
        print("💡 Run: python adaptive_agent.py --source document.pdf")
    
    # 5. Dependencies Check
    print("\n5. SYSTEM DEPENDENCIES")
    print("-" * 30)
    
    deps = {
        'pdf2image': 'PDF conversion',
        'cv2': 'Image processing', 
        'docling': 'Text/table extraction',
        'langchain_openai': 'Azure OpenAI integration',
        'streamlit': 'Web interface'
    }
    
    for package, purpose in deps.items():
        try:
            __import__(package)
            if package == 'cv2':
                import cv2
                print(f"✅ opencv-python (cv2): {purpose} - v{cv2.__version__}")
            else:
                print(f"✅ {package}: {purpose}")
        except ImportError:
            if package == 'cv2':
                print(f"❌ opencv-python: Missing - {purpose}")
                print(f"   Fix: pip install opencv-python")
            elif package == 'langchain_openai':
                print(f"❌ langchain-openai: Missing - {purpose}")
                print(f"   Fix: pip install langchain-openai langchain-core")
            else:
                print(f"❌ {package}: Missing - {purpose}")
                print(f"   Fix: pip install {package}")
    
    # 6. System Integration Test
    print("\n6. SYSTEM INTEGRATION")
    print("-" * 30)
    
    issues = []
    if not env_ok:
        issues.append("Azure OpenAI configuration incomplete")
    if total_images == 0:
        issues.append("No reference images for matching")
    if not pathlib.Path("feedback_data").exists():
        issues.append("No extraction data - run processing first")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ All systems appear to be working")
    
    # 7. Next Steps
    print("\n7. RECOMMENDED ACTIONS")
    print("-" * 30)
    
    if not env_ok:
        print("1. Configure Azure OpenAI in .env file")
    
    if total_images == 0:
        print("2. Add reference images:")
        print("   python adaptive_agent.py --setup-reference-data")
        print("   # Then add JPG/PNG files to labeled_data/ folders")
    
    if not pathlib.Path("feedback_data").exists():
        print("3. Process a document:")
        print("   python adaptive_agent.py --source document.pdf")
    
    print("4. View results:")
    print("   streamlit run feedback_interface.py")
    
    print("\n" + "="*60)
    print("📊 DIAGNOSIS COMPLETE")
    
    return env_ok and total_images > 0

if __name__ == "__main__":
    success = diagnose_system()
    if success:
        print("\n🎉 System ready for production use!")
    else:
        print("\n🔧 System needs configuration - follow recommended actions above")