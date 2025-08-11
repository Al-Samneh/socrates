#!/usr/bin/env python3
"""
Main script for generating Socratic questions and responses.

Usage:
    python scripts/generate_questions.py
"""

import sys
import os
import ast

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from socrates_ai.question_generator import SocraticQuestionGenerator


def load_topics_from_file(filename: str = "config/sample_topics.txt") -> list:
    """
    Load topics from configuration file.
    
    Args:
        filename: Path to the topics file
        
    Returns:
        List of topics
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            topics_str = f.read()
            topics = ast.literal_eval(topics_str)
        return topics
    except Exception as e:
        print(f"❌ Error loading topics from {filename}: {e}")
        return []


def main():
    """Main function to orchestrate the question generation process."""
    print("🚀 Socratic Question & Response Generator")
    print("=" * 50)
    
    # Get user input
    api_key = input("Enter your OpenAI API key: ")
    if not api_key.strip():
        print("❌ API key is required!")
        return
    
    try:
        num_pairs = int(input("Enter the number of question-response pairs to generate: "))
        if num_pairs <= 0:
            print("❌ Number of pairs must be positive!")
            return
    except ValueError:
        print("❌ Please enter a valid number!")
        return
    
    # Load topics
    print("📚 Loading topics from configuration...")
    topics = load_topics_from_file()
    if not topics:
        print("❌ No topics loaded! Please check config/sample_topics.txt")
        return
    
    print(f"📖 Loaded {len(topics)} topics")
    
    # Initialize generator
    generator = SocraticQuestionGenerator(api_key)
    
    # Generate dataset
    success = generator.generate_dataset(topics, num_pairs)
    
    if success:
        print("\n🎉 Generation completed successfully!")
        print("📂 Check the 'data/' directory for output files")
        print("📜 Check the 'logs/' directory for detailed logs")
    else:
        print("\n❌ Generation failed! Check logs for details.")


if __name__ == "__main__":
    main()
