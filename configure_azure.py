#!/usr/bin/env python3
"""
Azure OpenAI Configuration Helper
Interactive setup for Azure OpenAI credentials
"""
import os
import pathlib

def configure_azure_openai():
    """Interactive Azure OpenAI configuration"""
    print("🔧 Azure OpenAI Configuration Helper")
    print("=" * 50)
    
    # Check if .env exists
    env_file = pathlib.Path(".env")
    existing_config = {}
    
    if env_file.exists():
        print("✓ Found existing .env file")
        try:
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        existing_config[key.strip()] = value.strip()
        except Exception as e:
            print(f"⚠ Error reading .env file: {e}")
    else:
        print("ℹ️  No .env file found - creating new one")
    
    print("\n📝 Azure OpenAI Configuration")
    print("-" * 30)
    
    # Get configuration values
    configs = {}
    
    # Azure OpenAI Endpoint
    current_endpoint = existing_config.get('AZURE_OPENAI_ENDPOINT', '')
    print(f"\n1. Azure OpenAI Endpoint")
    print(f"   Example: https://your-resource-name.openai.azure.com/")
    if current_endpoint:
        print(f"   Current: {current_endpoint}")
    
    endpoint = input("   Enter endpoint (or press Enter to keep current): ").strip()
    if endpoint:
        configs['AZURE_OPENAI_ENDPOINT'] = endpoint
    elif current_endpoint:
        configs['AZURE_OPENAI_ENDPOINT'] = current_endpoint
    else:
        print("   ❌ Endpoint is required!")
        return False
    
    # Azure OpenAI API Key
    current_key = existing_config.get('AZURE_OPENAI_API_KEY', '')
    print(f"\n2. Azure OpenAI API Key")
    print(f"   Example: 1234567890abcdef1234567890abcdef")
    if current_key:
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "****"
        print(f"   Current: {masked_key}")
    
    api_key = input("   Enter API key (or press Enter to keep current): ").strip()
    if api_key:
        configs['AZURE_OPENAI_API_KEY'] = api_key
    elif current_key:
        configs['AZURE_OPENAI_API_KEY'] = current_key
    else:
        print("   ❌ API key is required!")
        return False
    
    # Azure OpenAI Deployment
    current_deployment = existing_config.get('AZURE_OPENAI_DEPLOYMENT', '')
    print(f"\n3. Azure OpenAI Deployment Name")
    print(f"   Example: gpt-4, gpt-35-turbo, your-deployment-name")
    if current_deployment:
        print(f"   Current: {current_deployment}")
    
    deployment = input("   Enter deployment name (or press Enter to keep current): ").strip()
    if deployment:
        configs['AZURE_OPENAI_DEPLOYMENT'] = deployment
    elif current_deployment:
        configs['AZURE_OPENAI_DEPLOYMENT'] = current_deployment
    else:
        print("   ❌ Deployment name is required!")
        return False
    
    # Write configuration
    print(f"\n💾 Writing configuration to .env file...")
    
    try:
        # Preserve other environment variables
        env_content = []
        
        # Add Azure OpenAI configs
        env_content.append("# Azure OpenAI Configuration for Enhanced Window Extraction")
        env_content.append(f"AZURE_OPENAI_ENDPOINT={configs['AZURE_OPENAI_ENDPOINT']}")
        env_content.append(f"AZURE_OPENAI_API_KEY={configs['AZURE_OPENAI_API_KEY']}")
        env_content.append(f"AZURE_OPENAI_DEPLOYMENT={configs['AZURE_OPENAI_DEPLOYMENT']}")
        env_content.append("")
        
        # Add other existing configs (if any)
        for key, value in existing_config.items():
            if not key.startswith('AZURE_OPENAI_'):
                env_content.append(f"{key}={value}")
        
        with open(".env", "w") as f:
            f.write('\n'.join(env_content))
        
        print("✅ Configuration saved to .env file")
        
        # Test the configuration
        print(f"\n🧪 Testing configuration...")
        
        # Set environment variables for testing
        for key, value in configs.items():
            os.environ[key] = value
        
        # Try to import and test
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, "-c",
                "from enhanced_llm_feedback import test_enhanced_connection; test_enhanced_connection()"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Configuration test successful!")
                print("\n🎉 Azure OpenAI is now configured!")
                print("Next steps:")
                print("1. python setup_enhanced.py  # Setup enhanced system")
                print("2. python enhanced_adaptive_agent.py --source document.pdf --characteristic anchors")
                return True
            else:
                print("⚠ Configuration test had issues:")
                print(result.stdout)
                print(result.stderr)
                print("\nConfiguration saved but test failed. Please check your credentials.")
                return True  # Still saved, just test failed
        
        except Exception as e:
            print(f"⚠ Could not test configuration: {e}")
            print("Configuration saved. You can test manually with:")
            print("python enhanced_llm_feedback.py --test-connection")
            return True
    
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def show_current_config():
    """Show current Azure OpenAI configuration"""
    print("🔍 Current Azure OpenAI Configuration")
    print("=" * 40)
    
    env_file = pathlib.Path(".env")
    if not env_file.exists():
        print("❌ No .env file found")
        return
    
    try:
        config = {}
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        
        print("Required variables:")
        for var in required_vars:
            value = config.get(var, '')
            if value:
                if 'API_KEY' in var:
                    # Mask API key for security
                    display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "****"
                else:
                    display_value = value
                print(f"  ✅ {var}: {display_value}")
            else:
                print(f"  ❌ {var}: (not set)")
        
        missing = [var for var in required_vars if not config.get(var)]
        if missing:
            print(f"\n⚠ Missing variables: {missing}")
            print("Run: python configure_azure.py --setup")
        else:
            print(f"\n✅ All required variables are set")
            print("Test with: python enhanced_llm_feedback.py --test-connection")
    
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure OpenAI Configuration Helper")
    parser.add_argument("--setup", action="store_true", help="Interactive setup")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.setup:
        configure_azure_openai()
    elif args.show:
        show_current_config()
    else:
        print("Azure OpenAI Configuration Helper")
        print("Usage:")
        print("  python configure_azure.py --setup   # Interactive setup")
        print("  python configure_azure.py --show    # Show current config")

if __name__ == "__main__":
    main()