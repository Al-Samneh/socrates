#!/usr/bin/env python3
"""
Main script for generating Socratic dialogues with historical characters.

Usage:
    python scripts/generate_dialogues.py
"""

import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from socrates_ai.dialogue_generator import SocraticDialogueGenerator


def main():
    """Main function to orchestrate the dialogue generation process."""
    print("🎭 Socratic Historical Dialogue Generator")
    print("=" * 50)
    
    # Get user input
    api_key = input("Enter your OpenAI API key: ")
    if not api_key.strip():
        print("❌ API key is required!")
        return
    
    try:
        num_dialogues = int(input("Enter the number of dialogue topics to generate: "))
        if num_dialogues <= 0:
            print("❌ Number of dialogues must be positive!")
            return
    except ValueError:
        print("❌ Please enter a valid number!")
        return
    
    # Initialize generator
    generator = SocraticDialogueGenerator(api_key)
    
    # Generate dataset
    success = generator.generate_dataset(num_dialogues)
    
    if success:
        print("\n🎉 Generation completed successfully!")
        print("📂 Check the 'data/' directory for output files")
        print("📜 Check the 'logs/' directory for detailed logs")
    else:
        print("\n❌ Generation failed! Check logs for details.")


if __name__ == "__main__":
    main()
