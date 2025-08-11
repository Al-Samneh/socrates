#!/usr/bin/env python3
"""
Setup script for Socrates AI project.

This script helps set up the project environment and verify everything is working.
"""

import os
import sys
import importlib.util

def check_python_version():
    """Check if Python version is adequate."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['openai', 'json', 'typing']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'json':
                import json
            elif package == 'typing':
                import typing
            elif package == 'openai':
                import openai
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_structure():
    """Verify project structure is correct."""
    required_dirs = ['src', 'src/socrates_ai', 'data', 'logs', 'config', 'scripts']
    required_files = [
        'src/socrates_ai/__init__.py',
        'src/socrates_ai/characters.py',
        'src/socrates_ai/dialogue_generator.py',
        'src/socrates_ai/question_generator.py',
        'config/sample_topics.txt',
        'requirements.txt'
    ]
    
    print("\n🏗️  Checking project structure...")
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
            return False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_imports():
    """Test if modules can be imported correctly."""
    print("\n🔍 Testing imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        from socrates_ai.characters import get_character_names
        print("✅ characters module imports successfully")
        print(f"   📊 {len(get_character_names())} characters available")
    except Exception as e:
        print(f"❌ Error importing characters: {e}")
        return False
    
    try:
        from socrates_ai.dialogue_generator import SocraticDialogueGenerator
        print("✅ dialogue_generator module imports successfully")
    except Exception as e:
        print(f"❌ Error importing dialogue_generator: {e}")
        return False
    
    try:
        from socrates_ai.question_generator import SocraticQuestionGenerator
        print("✅ question_generator module imports successfully")
    except Exception as e:
        print(f"❌ Error importing question_generator: {e}")
        return False
    
    return True

def main():
    """Main setup verification function."""
    print("🎭 Socrates AI Project Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check Python version
    all_good &= check_python_version()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    all_good &= check_dependencies()
    
    # Check project structure
    all_good &= check_project_structure()
    
    # Test imports
    all_good &= test_imports()
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("\n✨ You're ready to generate philosophical dialogues!")
        print("\n🚀 Quick start:")
        print("   python scripts/generate_dialogues.py")
        print("   python scripts/generate_questions.py")
    else:
        print("❌ Setup verification FAILED!")
        print("\n🔧 Please fix the issues above and run this script again.")
    
    return all_good

if __name__ == "__main__":
    main()
