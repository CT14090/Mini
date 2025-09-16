#!/usr/bin/env python3
"""
Azure OpenAI Configuration Checker and Test Script
"""
import os
from dotenv import load_dotenv

def check_azure_config():
    """Check Azure OpenAI configuration"""
    print("🔧 Checking Azure OpenAI Configuration...")
    
    # Load environment variables
    load_dotenv(override=True)
    
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'), 
        'AZURE_OPENAI_DEPLOYMENT': os.getenv('AZURE_OPENAI_DEPLOYMENT')
    }
    
    print("\nEnvironment Variables:")
    for var, value in required_vars.items():
        if value:
            if 'KEY' in var:
                print(f"✅ {var}: {value[:10]}...{value[-4:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    missing = [k for k, v in required_vars.items() if not v]
    
    if missing:
        print(f"\n❌ Missing variables: {missing}")
        print("\nTo fix, add these to your .env file:")
        for var in missing:
            print(f"{var}=your_value_here")
        return False
    
    # Test connection
    print("\n🧪 Testing Azure OpenAI connection...")
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = AzureChatOpenAI(
            azure_endpoint=required_vars['AZURE_OPENAI_ENDPOINT'],
            api_key=required_vars['AZURE_OPENAI_API_KEY'],
            azure_deployment=required_vars['AZURE_OPENAI_DEPLOYMENT'],
            api_version="2024-02-01",
            temperature=0.1,
            max_tokens=50
        )
        
        response = llm.invoke([HumanMessage(content="Reply 'OK' if you can read this.")])
        
        if response and response.content:
            print(f"✅ Connection successful! Response: {response.content}")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    check_azure_config()