"""
Dataset Transformation Script

This script transforms the existing socratic_dialogues_dataset.json to:
1. Remove TOPIC_PROCESSED entries  
2. Add 'name' and 'question' columns to each dialogue exchange
3. Create a clean, structured dataset

Usage:
    python scripts/transform_dataset.py
"""

import json
import os
import re
import sys
from datetime import datetime

# Add proper path handling for different execution contexts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

def debug_print(message):
    """Print debug information."""
    print(f"🔍 DEBUG: {message}")

def create_backup(original_file):
    """Create a backup of the original dataset file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{original_file}.backup_{timestamp}"
        
        debug_print(f"Creating backup from {original_file} to {backup_file}")
        
        if not os.path.exists(original_file):
            raise FileNotFoundError(f"Original file not found: {original_file}")
        
        with open(original_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Created backup: {backup_file}")
        return backup_file
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        raise

def extract_character_name_from_topic_marker(topic_output):
    """Extract character name from TOPIC_PROCESSED output message."""
    try:
        debug_print(f"Extracting character from: {topic_output}")
        # Pattern: "Topic 'question' completed with Character Name"
        match = re.search(r"completed with (.+)$", topic_output)
        if match:
            character = match.group(1).strip()
            debug_print(f"Found character: {character}")
            return character
        debug_print("No character found, returning Unknown")
        return "Unknown"
    except Exception as e:
        debug_print(f"Error extracting character: {e}")
        return "Unknown"

def extract_question_from_topic_marker(topic_input):
    """Extract question from TOPIC_PROCESSED input."""
    try:
        debug_print(f"Extracting question from: {topic_input}")
        question = topic_input.replace("TOPIC_PROCESSED:", "").strip()
        debug_print(f"Found question: {question}")
        return question
    except Exception as e:
        debug_print(f"Error extracting question: {e}")
        return "Unknown"

def transform_dataset(input_file, output_file):
    """Transform the dataset to remove TOPIC_PROCESSED entries and add metadata columns."""
    try:
        debug_print(f"Starting transformation from {input_file} to {output_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load the original dataset
        debug_print("Loading original dataset...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_dataset = data.get('dataset', [])
        print(f"📊 Original dataset has {len(original_dataset)} entries")
        
        # Process the dataset
        transformed_dataset = []
        current_character = "Unknown"
        current_question = "Unknown"
        topic_processed_count = 0
        
        debug_print("Processing entries...")
        for i, entry in enumerate(original_dataset):
            input_text = entry.get('input', '')
            output_text = entry.get('output', '')
            
            debug_print(f"Processing entry {i+1}/{len(original_dataset)}")
            
            # Check if this is a TOPIC_PROCESSED entry
            if input_text.startswith("TOPIC_PROCESSED:"):
                topic_processed_count += 1
                debug_print(f"Found TOPIC_PROCESSED entry #{topic_processed_count}")
                
                # Extract metadata for the following entries
                current_question = extract_question_from_topic_marker(input_text)
                current_character = extract_character_name_from_topic_marker(output_text)
                print(f"🔄 Found topic: '{current_question}' with {current_character}")
                # Skip this entry (don't add to transformed dataset)
                continue
            
            # This is a regular dialogue entry - add metadata and include it
            transformed_entry = {
                "input": input_text,
                "output": output_text,
                "name": current_character,
                "question": current_question
            }
            
            transformed_dataset.append(transformed_entry)
        
        print(f"✨ Transformed dataset has {len(transformed_dataset)} entries")
        print(f"🗑️  Removed {topic_processed_count} TOPIC_PROCESSED entries")
        
        # Create the new dataset structure
        debug_print("Creating transformed data structure...")
        transformed_data = {
            "dataset": transformed_dataset,
            "metadata": {
                "total_items": len(transformed_dataset),
                "transformation_completed": datetime.now().isoformat(),
                "original_file": input_file,
                "transformation_type": "added_name_and_question_columns",
                "format": "character_input_socrates_output_with_metadata",
                "removed_entries": topic_processed_count,
                "columns": {
                    "input": "What the historical character says",
                    "output": "What Socrates responds", 
                    "name": "Name of the historical character",
                    "question": "The philosophical topic/question being discussed"
                }
            }
        }
        
        # Save the transformed dataset
        debug_print(f"Saving transformed dataset to {output_file}")
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=4, ensure_ascii=False)
        
        print(f"💾 Saved transformed dataset to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_sample_entries(file_path, num_samples=3):
    """Display sample entries from the transformed dataset."""
    try:
        debug_print(f"Displaying samples from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = data.get('dataset', [])
        if not dataset:
            print("❌ No data found in dataset")
            return
        
        print(f"\n📋 Sample entries from transformed dataset:")
        print("=" * 80)
        
        for i in range(min(num_samples, len(dataset))):
            entry = dataset[i]
            print(f"\nEntry {i + 1}:")
            print(f"Question: {entry.get('question', 'N/A')}")
            print(f"Character: {entry.get('name', 'N/A')}")
            print(f"Input: {entry.get('input', 'N/A')[:100]}...")
            print(f"Output: {entry.get('output', 'N/A')[:100]}...")
            print("-" * 40)
        
        print(f"\n📊 Total entries: {len(dataset)}")
        
    except Exception as e:
        print(f"❌ Error displaying samples: {e}")

def main():
    """Main transformation function."""
    print("🔄 Socratic Dataset Transformation Tool")
    print("=" * 50)
    
    # Debug current working directory
    debug_print(f"Current working directory: {os.getcwd()}")
    debug_print(f"Script directory: {script_dir}")
    debug_print(f"Project root: {project_root}")
    
    input_file = "data/socratic_dialogues_dataset.json"
    output_file = "data/socratic_dialogues_dataset_transformed.json"
    
    # Check if files exist
    debug_print(f"Checking if input file exists: {input_file}")
    debug_print(f"File exists: {os.path.exists(input_file)}")
    debug_print(f"File absolute path: {os.path.abspath(input_file)}")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print(f"Current directory: {os.getcwd()}")
        print("Available files in data directory:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                print(f"  - {file}")
        else:
            print("  data/ directory not found")
        return
    
    print(f"📂 Input file: {input_file}")
    print(f"📂 Output file: {output_file}")
    
    try:
        # Create backup
        print("\n🔒 Creating backup...")
        backup_file = create_backup(input_file)
        
        # Transform the dataset
        print("\n🔄 Transforming dataset...")
        success = transform_dataset(input_file, output_file)
        
        if success:
            print("\n✅ Transformation completed successfully!")
            
            # Display sample entries
            display_sample_entries(output_file)
            
            print(f"\n📁 Files created:")
            print(f"   - Backup: {backup_file}")
            print(f"   - Transformed: {output_file}")
            
            # Ask user if they want to replace the original
            print(f"\n❓ Replace original file with transformed version?")
            response = input("   Type 'yes' to replace, anything else to keep both: ").lower().strip()
            
            if response == 'yes':
                os.rename(output_file, input_file)
                print(f"✅ Original file replaced with transformed version")
                print(f"📁 Backup preserved as: {backup_file}")
            else:
                print(f"✅ Both files preserved:")
                print(f"   - Original: {input_file}")
                print(f"   - Transformed: {output_file}")
                
        else:
            print("\n❌ Transformation failed!")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()