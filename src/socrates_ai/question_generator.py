"""
Socratic Question Generator

This module generates philosophical questions and Socratic responses for training
conversational AI models in the Socratic method.

Author: AI Assistant
Created: 2024
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

from openai import OpenAI


def log_message(message: str, log_file: str = "logs/question_generation.log") -> None:
    """
    Log message to both console and file with timestamp.
    
    Args:
        message: The message to log
        log_file: Path to the log file
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
        backup_count: Number of backup files to maintain
        
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
        log_message(f"❌ Error saving to {filename}: {str(e)}")
        return False


def load_existing_data(filename: str) -> List[Dict[str, str]]:
    """
    Load existing dataset if it exists.
    
    Args:
        filename: Path to the dataset file
        
    Returns:
        List of question-response items
    """
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            log_message(f"📁 Loaded existing dataset with {len(data.get('dataset', []))} items")
            return data.get('dataset', [])
        except Exception as e:
            log_message(f"❌ Error loading existing file {filename}: {str(e)}")
    return []


def generate_questions(
    api_key: str, 
    model: str, 
    topics: List[str], 
    num_questions_per_topic: int = 1, 
    questions_file: str = "data/questions_progress.json"
) -> List[str]:
    """
    Generate questions based on topics.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model to use
        topics: List of topics to generate questions for
        num_questions_per_topic: Number of questions per topic
        questions_file: File to store question progress
        
    Returns:
        List of generated questions
    """
    client = OpenAI(api_key=api_key)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(questions_file), exist_ok=True)
    
    # Load existing questions if any
    existing_questions = []
    if os.path.exists(questions_file):
        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_questions = existing_data.get("questions", [])
            log_message(f"📁 Loaded {len(existing_questions)} existing questions")
        except:
            log_message("❌ Error loading existing questions file")
    
    questions = existing_questions.copy()
    
    prompt_template = """
    You are a philosophy question generator. Your task is to create original questions centered on philosophy, inspired by the given topic.
    Generate {num} original questions for the topic: {topic}.
    Output as a JSON object with key "questions" and value as a list of strings.
    Ensure questions are varied and philosophical, e.g., "What is the essence of [concept]?" or "How does [concept] relate to human nature?"
    """
    
    log_message(f"🔄 Starting question generation for {len(topics)} topics...")
    
    for i, topic in enumerate(topics, 1):
        log_message(f"📝 [{i}/{len(topics)}] Processing topic: {topic}")
        num = num_questions_per_topic
        full_prompt = prompt_template.format(num=num, topic=topic)
        
        try:
            start_time = time.time()
            log_message(f"   🌐 Making API call...")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": full_prompt}],
                timeout=30
            )
            
            elapsed = time.time() - start_time
            log_message(f"   ✅ API call completed in {elapsed:.1f}s")
            
            response_content = response.choices[0].message.content
            log_message(f"   📤 Response preview: {response_content[:100]}...")
            
            generated = json.loads(response_content)
            new_questions = generated["questions"]
            questions.extend(new_questions)
            
            log_message(f"   ✅ Added {len(new_questions)} questions. Total: {len(questions)}")
            
            # Save progress every 5 topics
            if i % 5 == 0 or i == len(topics):
                progress_data = {
                    "questions": questions,
                    "topics_processed": i,
                    "total_topics": len(topics),
                    "timestamp": datetime.now().isoformat()
                }
                with open(questions_file, "w", encoding="utf-8") as f:
                    json.dump(progress_data, f, indent=4, ensure_ascii=False)
                log_message(f"💾 Progress saved: {i}/{len(topics)} topics processed")
            
        except json.JSONDecodeError as e:
            log_message(f"   ❌ JSON parsing error for topic '{topic}': {str(e)}")
            if 'response' in locals():
                log_message(f"      Raw response: {response.choices[0].message.content}")
        except Exception as e:
            log_message(f"   ❌ API error for topic '{topic}': {str(e)}")
            log_message(f"      Error type: {type(e).__name__}")
            
            # Wait a bit before continuing on error
            time.sleep(2)
    
    log_message(f"🎯 Question generation complete! Total: {len(questions)} questions")
    return questions


def generate_socratic_responses(
    api_key: str, 
    model: str, 
    questions: List[str], 
    dataset_file: str = "data/generated_dataset.json"
) -> List[Dict[str, str]]:
    """
    Generate Socratic responses for given questions.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model to use
        questions: List of questions to respond to
        dataset_file: File to store the dataset
        
    Returns:
        List of question-response pairs
    """
    client = OpenAI(api_key=api_key)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    
    # Load existing dataset
    dataset = load_existing_data(dataset_file)
    processed_questions = set(item["input"] for item in dataset)
    
    # Filter out already processed questions
    remaining_questions = [q for q in questions if q not in processed_questions]
    
    if not remaining_questions:
        log_message("✅ All questions already have responses!")
        return dataset
    
    log_message(f"🧠 Need to process {len(remaining_questions)} new questions (skipping {len(processed_questions)} existing)")
    
    socratic_prompt_template = """
    You are Socrates and a master of the Socratic method. Your task is to generate high-quality training data for a smaller AI model that is being fine-tuned to act as a Socratic dialogue partner.
    For the given user question: "{question}"
    Generate a single Socratic response.
    Output MUST be a single JSON object with key "output" and value as the <socratic_response> string.
    The <socratic_response> must adhere to these strict principles:
    1. Balanced Insight and Inquiry: It should provide a thoughtful answer or definition to the question, but also be inquisitive by asking clarifying or counter-questions to guide the user's thinking.
    2. Intellectual Humility: It should acknowledge the complexity of the topic.
    3. Deconstructive: It should break down the user's question into its core concepts.
    4. No Artifacts: The response must NOT include instructional tags like [INST].
    Example for illustration (do not use exact): For "What is courage?", output: {{"output": "Courage, my friend, seems to be the virtue that enables one to face danger wisely, not without fear but in spite of it. Yet, is this fully accurate? If a person acts bravely out of ignorance rather than knowledge, can we call that true courage? Let us explore: what role does wisdom play in distinguishing courage from mere boldness?"}}
    """
    
    for i, question in enumerate(remaining_questions, 1):
        log_message(f"🤔 [{i}/{len(remaining_questions)}] Processing: {question[:50]}...")
        full_prompt = socratic_prompt_template.format(question=question)
        
        try:
            start_time = time.time()
            log_message(f"   🌐 Making API call...")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": full_prompt}],
                timeout=30
            )
            
            elapsed = time.time() - start_time
            log_message(f"   ✅ API call completed in {elapsed:.1f}s")
            
            response_content = response.choices[0].message.content
            log_message(f"   📤 Response preview: {response_content[:100]}...")
            
            generated = json.loads(response_content)
            new_item = {"input": question, "output": generated["output"]}
            dataset.append(new_item)
            
            log_message(f"   ✅ Response added. Total dataset size: {len(dataset)}")
            
            # Save progress every 5 responses
            if i % 5 == 0 or i == len(remaining_questions):
                output_data = {
                    "dataset": dataset,
                    "metadata": {
                        "total_items": len(dataset),
                        "processed": len(dataset) - len(load_existing_data(dataset_file)),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                save_incremental_data(output_data, dataset_file)
                log_message(f"💾 Progress saved: {i}/{len(remaining_questions)} responses processed")
            
        except json.JSONDecodeError as e:
            log_message(f"   ❌ JSON parsing error for question: {question[:30]}... - {str(e)}")
            if 'response' in locals():
                log_message(f"      Raw response: {response.choices[0].message.content}")
        except Exception as e:
            log_message(f"   ❌ API error for question: {question[:30]}... - {str(e)}")
            log_message(f"      Error type: {type(e).__name__}")
            
            # Wait a bit before continuing on error
            time.sleep(2)
    
    log_message(f"🎯 Response generation complete! Total dataset: {len(dataset)} items")
    return dataset


class SocraticQuestionGenerator:
    """
    Main class for generating Socratic questions and responses.
    """
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        """
        Initialize the question generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key
        self.model = model
        self.log_file = "logs/question_generation.log"
        
        # Clear log file for new session
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        log_message("🚀 Starting Socratic Dataset Generator with Incremental Saving", self.log_file)
        log_message(f"🤖 Using model: {self.model}", self.log_file)
    
    def generate_dataset(self, topics: List[str], num_pairs: int) -> bool:
        """
        Generate a complete dataset of questions and Socratic responses.
        
        Args:
            topics: List of topics to generate questions from
            num_pairs: Number of question-response pairs to generate
            
        Returns:
            True if generation was successful, False otherwise
        """
        log_message(f"🎯 Target: {num_pairs} question-response pairs", self.log_file)
        log_message(f"📖 Loaded {len(topics)} topics", self.log_file)
        
        # Determine how many questions per topic
        if num_pairs <= len(topics):
            selected_topics = topics[:num_pairs]
            num_per_topic = 1
        else:
            selected_topics = (topics * (num_pairs // len(topics) + 1))[:num_pairs]
            num_per_topic = 1
        
        log_message(f"🎯 Selected {len(selected_topics)} topics for {num_pairs} pairs", self.log_file)
        
        # Generate questions with incremental saving
        log_message("=" * 60, self.log_file)
        log_message("PHASE 1: GENERATING QUESTIONS", self.log_file)
        log_message("=" * 60, self.log_file)
        questions = generate_questions(self.api_key, self.model, selected_topics, num_per_topic)
        
        if not questions:
            log_message("❌ No questions were generated successfully. Exiting.", self.log_file)
            return False
        
        # Generate responses for each question with incremental saving
        log_message("=" * 60, self.log_file)
        log_message("PHASE 2: GENERATING SOCRATIC RESPONSES", self.log_file)
        log_message("=" * 60, self.log_file)
        dataset = generate_socratic_responses(self.api_key, self.model, questions)
        
        if not dataset:
            log_message("❌ No responses were generated successfully. Exiting.", self.log_file)
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
                "target_pairs": num_pairs
            }
        }
        
        if save_incremental_data(output_json, "data/generated_dataset.json"):
            log_message("🎉 Dataset generation completed successfully!", self.log_file)
            log_message(f"📊 Final dataset size: {len(dataset)} question-response pairs", self.log_file)
            log_message(f"📁 Saved to: data/generated_dataset.json", self.log_file)
            log_message(f"📜 Full log saved to: {self.log_file}", self.log_file)
            return True
        else:
            log_message("❌ Final save failed!", self.log_file)
            return False
