"""
Socratic Historical Dialogue Generator

This module generates conversational dialogues between Socrates and various 
historical figures using OpenAI's API. The dialogues capture the authentic 
personalities and speaking styles of historical characters while maintaining 
the Socratic method of philosophical inquiry.

Author: AI Assistant
Created: 2024
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from openai import OpenAI
from .characters import HISTORICAL_CHARACTERS, get_character_names, get_random_character


def log_message(message: str, log_file: str = "logs/dialogue_generation.log") -> None:
    """
    Log message to both console and file with timestamp.
    
    Args:
        message: The message to log
        log_file: Path to the log file (default: "logs/dialogue_generation.log")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Also save to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")


def save_incremental_data(data: Dict[str, Any], filename: str, backup_count: int = 3) -> bool:
    """
    Save data with backup rotation to prevent data loss.
    
    Args:
        data: The data dictionary to save
        filename: Path to the target file
        backup_count: Number of backup files to maintain (default: 3)
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create backup of existing file
        if os.path.exists(filename):
            for i in range(backup_count-1, 0, -1):
                old_backup = f"{filename}.backup{i}"
                new_backup = f"{filename}.backup{i+1}"
                if os.path.exists(old_backup):
                    if os.path.exists(new_backup):
                        os.remove(new_backup)
                    os.rename(old_backup, new_backup)
            
            # Create backup1 from current file
            backup1 = f"{filename}.backup1"
            if os.path.exists(backup1):
                os.remove(backup1)
            os.rename(filename, backup1)
        
        # Save new data
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        log_message(f"✅ Saved {len(data.get('dataset', []))} items to {filename}")
        return True
    except Exception as e:
        log_message(f"Error saving to {filename}: {str(e)}")
        return False


def load_existing_data(filename: str) -> List[Dict[str, str]]:
    """
    Load existing dataset from file if it exists.
    
    Args:
        filename: Path to the dataset file
        
    Returns:
        List of dialogue items, empty list if file doesn't exist or can't be loaded
    """
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            log_message(f"📁 Loaded existing dataset with {len(data.get('dataset', []))} items")
            return data.get('dataset', [])
        except Exception as e:
            log_message(f"Error loading existing file {filename}: {str(e)}")
    return []


def generate_dialogue_topics(
    api_key: str, 
    model: str, 
    num_topics: int = 50, 
    topics_file: str = "data/dialogue_topics.json"
) -> List[str]:
    """
    Generate philosophical topics suitable for Socratic dialogues with historical characters.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model to use for generation
        num_topics: Number of topics to generate (default: 50)
        topics_file: File to store generated topics (default: "data/dialogue_topics.json")
        
    Returns:
        List of philosophical topic strings
    """
    client = OpenAI(api_key=api_key)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(topics_file), exist_ok=True)
    
    # Load existing topics if any
    existing_topics = []
    if os.path.exists(topics_file):
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_topics = existing_data.get("topics", [])
            log_message(f"📁 Loaded {len(existing_topics)} existing topics")
        except:
            log_message("Error loading existing topics file")
    
    if len(existing_topics) >= num_topics:
        log_message(f"✅ Already have {len(existing_topics)} topics, using existing ones")
        return existing_topics[:num_topics]
    
    topics = existing_topics.copy()
    needed_topics = num_topics - len(existing_topics)
    
    prompt_template = """
    Generate {num} philosophical discussion topics that would make for engaging dialogues between Socrates and various historical philosophers.
    
    The topics should be:
    - Fundamental philosophical questions about life, ethics, knowledge, reality
    - Suitable for dramatic, engaging conversations between great thinkers
    - Topics where different philosophical traditions would have interesting disagreements
    - Examples: "The nature of true happiness", "Whether virtue can be taught", "The relationship between power and justice"
    
    Output as a JSON object with key "topics" and value as a list of strings.
    Make the topics thought-provoking but accessible, suitable for philosophical dialogue.
    """
    
    full_prompt = prompt_template.format(num=needed_topics)
    
    try:
        log_message(f"🔄 Generating {needed_topics} new dialogue topics...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": full_prompt}],
            timeout=30
        )
        
        elapsed = time.time() - start_time
        log_message(f"✅ API call completed in {elapsed:.1f}s")
        
        response_content = response.choices[0].message.content
        generated = json.loads(response_content)
        new_topics = generated["topics"]
        topics.extend(new_topics)
        
        # Save topics
        topics_data = {
            "topics": topics,
            "timestamp": datetime.now().isoformat()
        }
        with open(topics_file, "w", encoding="utf-8") as f:
            json.dump(topics_data, f, indent=4, ensure_ascii=False)
        
        log_message(f"✅ Generated {len(new_topics)} new topics. Total: {len(topics)}")
        
    except Exception as e:
        log_message(f"Error generating topics: {str(e)}")
        if len(existing_topics) > 0:
            log_message("Using existing topics only")
            return existing_topics
        else:
            return []
    
    return topics[:num_topics]


def parse_dialogue_into_exchanges(dialogue_text: str, character_name: str) -> List[Dict[str, str]]:
    """
    Parse a full dialogue into individual exchange pairs (character input -> Socrates output).
    
    Args:
        dialogue_text: The complete dialogue text
        character_name: Name of the historical character
        
    Returns:
        List of exchange pairs with 'input' (character) and 'output' (Socrates) keys
    """
    exchanges = []
    lines = [line.strip() for line in dialogue_text.split('\n') if line.strip()]
    
    character_statement = None
    socrates_statement = None
    
    for line in lines:
        if line.upper().startswith(f"{character_name.upper()}:"):
            # If we have a pending Socrates response, save the previous exchange
            if character_statement and socrates_statement:
                exchanges.append({
                    "input": character_statement,
                    "output": socrates_statement
                })
            
            # Start new exchange with character statement
            character_statement = line[len(character_name)+1:].strip()
            socrates_statement = None
            
        elif line.upper().startswith("SOCRATES:"):
            # Socrates response to current character statement
            socrates_response = line[9:].strip()  # Remove "SOCRATES:"
            
            if character_statement:
                # Normal exchange: character -> Socrates
                socrates_statement = socrates_response
            else:
                # Socrates talking to himself or opening - treat as both input and output
                exchanges.append({
                    "input": socrates_response,
                    "output": socrates_response
                })
    
    # Don't forget the last exchange if it exists
    if character_statement and socrates_statement:
        exchanges.append({
            "input": character_statement,
            "output": socrates_statement
        })
    
    return exchanges


def generate_socratic_dialogues(
    api_key: str, 
    model: str, 
    topics: List[str], 
    dataset_file: str = "data/socratic_dialogues_dataset.json"
) -> List[Dict[str, str]]:
    """
    Generate conversational dialogues between Socrates and historical characters.
    Each dialogue is broken into individual exchange pairs (character input -> Socrates output).
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model to use for generation
        topics: List of philosophical topics for dialogues
        dataset_file: File to store generated dialogues (default: "data/socratic_dialogues_dataset.json")
        
    Returns:
        List of exchange pairs with 'input' (character statement) and 'output' (Socrates response) keys
    """
    client = OpenAI(api_key=api_key)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    
    # Load existing dataset
    dataset = load_existing_data(dataset_file)
    
    # For the new format, we need to track which topics have been processed differently
    # We'll use a simple approach: if we have any exchanges for a topic, we skip it
    processed_topic_markers = set()
    for item in dataset:
        # Look for our topic marker in the input or create one
        if "TOPIC_PROCESSED:" in item.get("input", ""):
            topic = item["input"].replace("TOPIC_PROCESSED:", "").strip()
            processed_topic_markers.add(topic)
    
    remaining_topics = [t for t in topics if t not in processed_topic_markers]
    
    if not remaining_topics:
        log_message("All topics already have dialogues!")
        return dataset
    
    log_message(f"Need to process {len(remaining_topics)} new topics (skipping {len(processed_topic_markers)} existing)")
    
    dialogue_prompt_template = """
    You are creating a philosophical dialogue between {character_name} and Socrates on the topic: "{topic}"
    
    Character Profile for {character_name}:
    - Description: {character_description}
    - Speaking Style: {speaking_style}
    - Key Traits: {key_traits}
    
    Create a natural, engaging dialogue that:
    1. Shows both philosophers' authentic personalities and speaking styles
    2. Is conversational and accessible, not overly academic
    3. Demonstrates their different philosophical approaches to the topic
    4. Includes substantive back-and-forth exchange of ideas
    5. Captures the essence of how these historical figures might actually converse
    6. Flows naturally like a real conversation
    
    The dialogue should be 6-8 exchanges total (alternating speakers).
    Start with {character_name} introducing the topic or asking an opening question.
    
    Format your response as a JSON object with a single key "dialogue" containing the complete conversation.
    Use clear speaker labels: "{character_name_upper}:" and "SOCRATES:" for each statement.
    
    Example format:
    {{
        "dialogue": "{character_name_upper}: [opening statement about the topic]\\n\\nSOCRATES: [response with characteristic questioning]\\n\\n{character_name_upper}: [continued discussion]\\n\\nSOCRATES: [further exploration]\\n\\n[continue alternating...]"
    }}
    """
    
    for i, topic in enumerate(remaining_topics, 1):
        # Randomly select a historical character for this dialogue
        character_name = get_random_character()
        character_info = HISTORICAL_CHARACTERS[character_name]
        character_name_upper = character_name.upper()
        
        log_message(f"[{i}/{len(remaining_topics)}] Topic: {topic[:50]}... with {character_name}")
        
        full_prompt = dialogue_prompt_template.format(
            character_name=character_name,
            character_name_upper=character_name_upper,
            topic=topic,
            character_description=character_info["description"],
            speaking_style=character_info["speaking_style"],
            key_traits=character_info["key_traits"]
        )
        
        try:
            start_time = time.time()
            log_message(f"Making API call...")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": full_prompt}],
                timeout=60  # Longer timeout for dialogue generation
            )
            
            elapsed = time.time() - start_time
            log_message(f"API call completed in {elapsed:.1f}s")
            
            response_content = response.choices[0].message.content
            log_message(f" Response preview: {response_content[:100]}...")
            
            generated = json.loads(response_content)
            dialogue_text = generated["dialogue"]
            
            # Parse the dialogue into individual exchanges
            exchanges = parse_dialogue_into_exchanges(dialogue_text, character_name)
            
            if exchanges:
                # Add all exchanges to the dataset
                dataset.extend(exchanges)
                log_message(f"Added {len(exchanges)} exchanges from dialogue. Total dataset size: {len(dataset)}")
                
                # Add a topic marker to track that this topic has been processed
                topic_marker = {"input": f"TOPIC_PROCESSED:{topic}", "output": f"Topic '{topic}' completed with {character_name}"}
                dataset.append(topic_marker)
            else:
                log_message(f"Warning: No valid exchanges parsed from dialogue for topic '{topic}'")
            
            # Save progress every 2 dialogues (since we now generate more items per dialogue)
            if i % 2 == 0 or i == len(remaining_topics):
                output_data = {
                    "dataset": dataset,
                    "metadata": {
                        "total_items": len(dataset),
                        "processed": len(dataset) - len(load_existing_data(dataset_file)),
                        "timestamp": datetime.now().isoformat(),
                        "characters_used": list(set([item.get("character", "mixed") for item in dataset])),
                        "dialogue_type": "socratic_exchange_pairs",
                        "format": "character_input_socrates_output_pairs"
                    }
                }
                save_incremental_data(output_data, dataset_file)
                log_message(f"Progress saved: {i}/{len(remaining_topics)} dialogues processed")
            
        except json.JSONDecodeError as e:
            log_message(f"   JSON parsing error for topic: {topic[:30]}... - {str(e)}")
            if 'response' in locals():
                log_message(f"      Raw response: {response.choices[0].message.content}")
        except Exception as e:
            log_message(f"   API error for topic: {topic[:30]}... - {str(e)}")
            log_message(f"      Error type: {type(e).__name__}")
            
            # Wait a bit before continuing on error
            time.sleep(2)
    
    log_message(f"Dialogue generation complete! Total dataset: {len(dataset)} items")
    return dataset


class SocraticDialogueGenerator:
    """
    Main class for generating Socratic dialogues with historical characters.
    """
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        """
        Initialize the dialogue generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: 'gpt-4o-mini')
        """
        self.api_key = api_key
        self.model = model
        self.log_file = "logs/dialogue_generation.log"
        
        # Clear log file for new session
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        log_message("Starting Socratic Historical Dialogue Generator", self.log_file)
        log_message(f"Available historical characters: {', '.join(get_character_names())}", self.log_file)
        log_message(f"Using model: {self.model}", self.log_file)
    
    def generate_dataset(self, num_dialogues: int) -> bool:
        """
        Generate a complete dataset of Socratic dialogues.
        
        Args:
            num_dialogues: Number of dialogue topics to generate
            
        Returns:
            True if generation was successful, False otherwise
        """
        log_message(f"Target: {num_dialogues} dialogue pairs", self.log_file)
        
        # Generate topics for dialogues
        log_message("=" * 60, self.log_file)
        log_message("PHASE 1: GENERATING DIALOGUE TOPICS", self.log_file)
        log_message("=" * 60, self.log_file)
        topics = generate_dialogue_topics(self.api_key, self.model, num_dialogues)
        
        if not topics:
            log_message("No topics were generated successfully. Exiting.", self.log_file)
            return False
        
        # Generate dialogues for each topic
        log_message("=" * 60, self.log_file)
        log_message("PHASE 2: GENERATING SOCRATIC DIALOGUES", self.log_file)
        log_message("=" * 60, self.log_file)
        dataset = generate_socratic_dialogues(self.api_key, self.model, topics)
        
        if not dataset:
            log_message("No dialogues were generated successfully. Exiting.", self.log_file)
            return False
        
        # Final save
        log_message("=" * 60, self.log_file)
        log_message("FINAL SAVE", self.log_file)
        log_message("=" * 60, self.log_file)
        output_json = {
            "dataset": dataset,
            "metadata": {
                "total_items": len(dataset),
                "generation_completed": datetime.now().isoformat(),
                "model_used": self.model,
                "target_dialogues": num_dialogues,
                "dialogue_type": "socratic_exchange_pairs",
                "available_characters": get_character_names(),
                "format": "character_input_socrates_output_pairs"
            }
        }
        
        if save_incremental_data(output_json, "data/socratic_dialogues_dataset.json"):
            log_message("Dialogue dataset generation completed successfully!", self.log_file)
            log_message(f"Dataset size: {len(dataset)} dialogue pairs", self.log_file)
            log_message(f"Saved to: data/socratic_dialogues_dataset.json", self.log_file)
            log_message(f"Full log saved to: {self.log_file}", self.log_file)
            log_message(f"Characters featured: {', '.join(get_character_names())}", self.log_file)
            return True
        else:
            log_message("Final save failed!", self.log_file)
            return False
