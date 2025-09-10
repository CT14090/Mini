#!/usr/bin/env python3
"""
Setup script for Window Characteristics Extraction Agent
Handles installation and environment setup
"""
import subprocess
import sys
import os
import pathlib
from typing import List, Tuple

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        print(f"🔧 {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False, str(e)

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Need Python 3.8+")
        return False

def install_dependencies() -> bool:
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "streamlit>=1.28.0"
    ]
    
    # Try to install core dependencies first
    for dep in core_deps:
        success, _ = run_command([sys.executable, "-m", "pip", "install", dep], 
                               f"Installing {dep}")
        if not success:
            return False
    
    # Computer vision dependencies (optional but recommended)
    cv_deps = [
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0", 
        "numpy>=1.24.0"
    ]
    
    print("🖼️ Installing computer vision dependencies...")
    cv_success = True
    for dep in cv_deps:
        success, _ = run_command([sys.executable, "-m", "pip", "install", dep],
                               f"Installing {dep}")
        if not success:
            cv_success = False
    
    if not cv_success:
        print("⚠️ Computer vision dependencies failed - feature detection will be limited")
    
    # Docling (document processing)
    print("📄 Installing document processing...")
    docling_success, _ = run_command([sys.executable, "-m", "pip", "install", "docling>=1.0.0"],
                                   "Installing docling")
    
    if not docling_success:
        print("❌ Docling installation failed - trying alternative approach...")
        # Try installing docling dependencies separately
        docling_deps = [
            "pypdf>=3.0.0",
            "pymupdf>=1.23.0",
            "python-magic>=0.4.27"
        ]
        
        for dep in docling_deps:
            run_command([sys.executable, "-m", "pip", "install", dep],
                       f"Installing {dep}")
    
    # LangChain for LLM feedback
    print("🤖 Installing LLM dependencies...")
    llm_deps = [
        "langchain>=0.2.0",
        "langchain-core>=0.2.0", 
        "langchain-openai>=0.1.0"
    ]
    
    llm_success = True
    for dep in llm_deps:
        success, _ = run_command([sys.executable, "-m", "pip", "install", dep],
                               f"Installing {dep}")
        if not success:
            llm_success = False
    
    if not llm_success:
        print("⚠️ LLM dependencies failed - feedback will use fallback mode")
    
    return True

def setup_environment():
    """Setup environment files and directories"""
    print("🏗️ Setting up environment...")
    
    # Create directories
    dirs_to_create = [
        "data/input_pdfs",
        "data/outputs", 
        "feedback_data"
    ]
    
    for dir_path in dirs_to_create:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        env_content = """# Azure OpenAI Configuration for LLM Feedback
# Get these values from your Azure OpenAI resource

AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Optional: API Version (defaults to 2024-02-01)
# AZURE_OPENAI_API_VERSION=2024-02-01
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
        print("📝 Please edit .env with your Azure OpenAI credentials")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test if core components are working"""
    print("\n🧪 Testing installation...")
    
    # Test imports
    test_results = {}
    
    # Test docling
    try:
        from docling.document_converter import DocumentConverter
        test_results['docling'] = True
        print("✅ Docling import successful")
    except Exception as e:
        test_results['docling'] = False
        print(f"❌ Docling import failed: {e}")
    
    # Test computer vision
    try:
        import cv2
        import numpy as np
        from PIL import Image
        test_results['cv'] = True
        print("✅ Computer vision imports successful")
    except Exception as e:
        test_results['cv'] = False
        print(f"⚠️ Computer vision imports failed: {e}")
    
    # Test langchain
    try:
        from langchain_openai import AzureChatOpenAI
        test_results['langchain'] = True
        print("✅ LangChain imports successful")
    except Exception as e:
        test_results['langchain'] = False
        print(f"⚠️ LangChain imports failed: {e}")
    
    # Test streamlit
    try:
        import streamlit
        test_results['streamlit'] = True
        print("✅ Streamlit import successful")
    except Exception as e:
        test_results['streamlit'] = False
        print(f"❌ Streamlit import failed: {e}")
    
    return test_results

def create_fallback_mode():
    """Create a fallback version that works with minimal dependencies"""
    print("🔄 Creating fallback configuration...")
    
    fallback_config = {
        "docling_available": False,
        "cv_available": False,
        "langchain_available": False,
        "use_fallback_processing": True,
        "fallback_pdf_processor": "pypdf2",
        "fallback_image_processing": False
    }
    
    with open("fallback_config.json", "w") as f:
        import json
        json.dump(fallback_config, f, indent=2)
    
    print("✅ Fallback configuration created")

def main():
    """Main setup function"""
    print("🎯 Window Characteristics Extraction Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️ Some dependencies failed to install")
        create_fallback_mode()
    
    # Setup environment
    setup_environment()
    
    # Test installation
    test_results = test_installation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)
    
    success_count = sum(test_results.values())
    total_count = len(test_results)
    
    print(f"✅ {success_count}/{total_count} components installed successfully")
    
    if test_results.get('docling', False):
        print("🎉 Ready for full document processing!")
    else:
        print("⚠️ Limited functionality - some PDF processing may not work")
    
    if test_results.get('langchain', False):
        print("🤖 LLM feedback available (configure Azure OpenAI in .env)")
    else:
        print("🔧 LLM feedback will use fallback mode")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Edit .env file with your Azure OpenAI credentials")
    print("2. Test with: python adaptive_agent.py --test-params --characteristic anchors --source dummy")
    print("3. Process document: python adaptive_agent.py --source document.pdf --characteristic anchors")
    print("4. View results: streamlit run feedback_interface.py")

if __name__ == "__main__":
    main()