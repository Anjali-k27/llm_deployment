#!/usr/bin/env python3
"""
Setup script for the Quantized LLM Chat Application
"""

# lets you run any system level commands (lik pip install or any other python file)
import subprocess  

# It is helpful for getting python interpreter access 
import sys

import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Quantized LLM Chat Application...")
    
    # Check if Python is available
    try:
        python_version = subprocess.check_output([sys.executable, "--version"], text=True).strip()
        print(f"✅ Found Python: {python_version}")
    except Exception as e:
        print(f"❌ Python not found: {e}")
        sys.exit(1)
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)
    
    # Run quantization
    if not run_command(f"{sys.executable} quantize.py", "Downloading and quantizing model"):
        print("⚠️ Model quantization failed, but you can try running it manually later.")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("   1. To run locally: streamlit run app.py")
    print("   2. To run with Docker: docker-compose up --build")
    print("   3. Access the app at: http://localhost:8501")

if __name__ == "__main__":
    main()
