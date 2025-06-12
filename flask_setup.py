#!/usr/bin/env python3
"""
Setup script for Political Evolution Web Application

This script sets up the directory structure and files needed
for the Flask web application.

Author: Phil Moyer (phil@moyer.ai)
Date: June 2025
"""

import os
from pathlib import Path

def setup_web_app():
    """Set up the web application directory structure"""
    
    print("Setting up Political Evolution Web Application...")
    
    # Create directory structure
    directories = [
        'templates',
        'static',
        'static/css',
        'static/js',
        'web_output'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Check for required files
    required_files = [
        'political_evolution_ga.py',
        'political_cli.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Warning: The following required files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nMake sure these files are in the same directory as the web app.")
    else:
        print("✓ All required files found")
    
    print("\n📋 Setup Instructions:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Save the Flask app code as 'app.py'")
    print("3. Save the HTML template in 'templates/index.html'")
    print("4. Run the app: python app.py")
    print("5. Open browser to: http://localhost:5000")
    
    print("\n🔧 Optional: Set environment variables:")
    print("   export FLASK_ENV=development")
    print("   export FLASK_DEBUG=1")
    
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    setup_web_app()
