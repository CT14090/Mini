#!/usr/bin/env python3
# test_azure_connection.py
"""
Dedicated Azure OpenAI connection testing script
"""

import os
from dotenv import load_dotenv

def test_azure_connection():
    """Test Azure OpenAI connection step by step"""
    print("Azure OpenAI Connection Test")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    print("1. Checking environment variables...")
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    
    print(f"   AZURE_OPENAI_ENDPOINT: {'✓' if endpoint else '❌ Missing'}")
    if endpoint:
        print(f"     Value: {endpoint[:50]}{'...' if len(endpoint) > 50 else ''}")
    
    print(f"   AZURE_OPENAI_API_KEY: {'✓' if api_key else '❌ Missing'}")
    if api_key:
        print(f"     Length: {len(api_key)} characters")
    
    print(f"   AZURE_OPENAI_DEPLOYMENT: {'✓' if deployment else '❌ Missing'}")
    if deployment:
        print(f"     Value: {deployment}")
    
    if not all([endpoint, api_key, deployment]):
        print("\n❌ Missing required environment variables")
        print("Create a .env file with:")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("AZURE_OPENAI_API_KEY=your-api-key")
        print("AZURE_OPENAI_DEPLOYMENT=your-deployment-name")
        return False
    
    # Test import
    print("\n2. Testing imports...")
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage
        print("   ✓ Imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Install with: pip install langchain-openai")
        return False
    
    # Test client creation
    print("\n3. Creating Azure OpenAI client...")
    try:
        client = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version="2024-02-01",
            temperature=0.1,
            max_tokens=100
        )
        print("   ✓ Client created successfully")
    except Exception as e:
        print(f"   ❌ Client creation failed: {e}")
        return False
    
    # Test basic text chat
    print("\n4. Testing basic text chat...")
    try:
        test_message = HumanMessage(content="Hello, please respond with 'Connection test successful'")
        response = client.invoke([test_message])
        print(f"   ✓ Text chat successful")
        print(f"   Response: {response.content}")
    except Exception as e:
        print(f"   ❌ Text chat failed: {e}")
        print(f"   This might indicate deployment or authentication issues")
        return False
    
    # Test vision capability
    print("\n5. Testing vision capability...")
    try:
        # Create a simple 1x1 pixel test image
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        vision_message = HumanMessage(content=[
            {"type": "text", "text": "What do you see in this image? Just respond with 'Vision test successful'."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}}
        ])
        
        vision_response = client.invoke([vision_message])
        print(f"   ✓ Vision capability confirmed")
        print(f"   Response: {vision_response.content}")
        
    except Exception as e:
        print(f"   ❌ Vision test failed: {e}")
        print(f"   Your deployment might not support vision capabilities")
        print(f"   Check if you're using a GPT-4 Vision deployment")
        return False
    
    print("\n✅ All tests passed! Azure OpenAI is ready for document analysis.")
    return True

def test_extractor_with_azure():
    """Test the extractor with Azure enabled"""
    print("\n" + "=" * 40)
    print("Testing Extractor with Azure")
    print("=" * 40)
    
    try:
        from characteristic_based_extractor import AzureFirstExtractor
        
        print("Creating extractor...")
        extractor = AzureFirstExtractor()
        
        if extractor.azure_client:
            print("✅ Extractor created with Azure OpenAI client")
            print(f"   Training data loaded: {sum(len(data) for data in extractor.training_data.values())} examples")
        else:
            print("❌ Extractor created but no Azure client")
            print("   Will use fallback methods")
            
        return extractor.azure_client is not None
        
    except Exception as e:
        print(f"❌ Extractor test failed: {e}")
        return False

if __name__ == "__main__":
    azure_ok = test_azure_connection()
    
    if azure_ok:
        extractor_ok = test_extractor_with_azure()
        
        if extractor_ok:
            print("\n🎉 System ready for Azure-powered document analysis!")
            print("\nNext steps:")
            print("python adaptive_agent.py --source document.pdf --characteristic anchors")
        else:
            print("\n⚠️ Azure connection works but extractor needs attention")
    else:
        print("\n❌ Fix Azure connection first")