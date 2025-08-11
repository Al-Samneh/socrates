# Socrates AI - Philosophical Dialogue Generation

A comprehensive toolkit for generating Socratic dialogues and training conversational AI models in the style of philosophical inquiry.

## 🏗️ Project Structure

```
socrates/
├── src/                          # Source code modules
│   └── socrates_ai/             # Main package
│       ├── __init__.py          # Package initialization
│       ├── characters.py        # Historical character definitions
│       ├── dialogue_generator.py # Dialogue generation logic
│       └── question_generator.py # Question/response generation
├── scripts/                     # Executable scripts
│   ├── generate_dialogues.py   # Generate historical dialogues
│   ├── generate_questions.py   # Generate Q&A pairs
│   ├── sft_finetune.py         # Supervised fine-tuning
│   ├── dpo_improvement.py      # DPO improvement training
│   └── web_app.py              # Web application
├── data/                        # Generated datasets and progress files
│   ├── socratic_dialogues_dataset.json
│   ├── generated_dataset.json
│   └── *.backup*               # Automatic backups
├── logs/                        # Generation logs
├── config/                      # Configuration files
│   └── sample_topics.txt       # Sample philosophical topics
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

Get your API key from [OpenAI](https://platform.openai.com/api-keys) and have it ready.

### 3. Generate Dialogues

Generate conversational exchanges between Socrates and historical figures that I picked:

```bash
python scripts/generate_dialogues.py
```

### 4. Generate Q&A Pairs

Generate traditional question-response pairs in Socratic style:

```bash
python scripts/generate_questions.py
```

## 📊 Output Formats

### Dialogue Format (Character Input → Socrates Output)

Each dialogue is broken into individual exchange pairs:

```json
{
  "dataset": [
    {
      "input": "Socrates, I believe virtue can be taught like any other skill.",
      "output": "Ah, my dear Aristotle, but if virtue is teachable, why do we see children of virtuous parents who lack virtue themselves?"
    },
    {
      "input": "Perhaps because virtue requires not just instruction but practice and the right disposition.",
      "output": "You speak wisely. But this raises another question: how do we acquire that right disposition?"
    }
  ]
}
```

### Question-Response Format

Traditional Socratic Q&A pairs:

```json
{
  "dataset": [
    {
      "input": "What is the nature of justice?",
      "output": "Justice, my friend, appears to be giving each their due. But what constitutes 'due'? Is it what the law prescribes, what benefits society, or something else entirely?"
    }
  ]
}
```

## 🎭 Historical Characters

The system includes 26+ historical figures with authentic personalities:

- **Philosophers**: Aristotle, Plato, Kant, Nietzsche, Confucius, Lao Tzu
- **Leaders**: Alexander the Great, Julius Caesar, Napoleon, Churchill
- **Scientists**: Einstein, Newton, Tesla, Marie Curie
- **And many more...**

Each character has:
- Detailed biographical background
- Authentic speaking style
- Key personality traits and philosophical positions

## 🔧 Features

- **Incremental Generation**: Resume interrupted sessions
- **Automatic Backups**: Prevent data loss with rotating backups
- **Progress Tracking**: Detailed logs of generation process
- **Character Authenticity**: Historically accurate personalities
- **Flexible Topics**: Generate from philosophical themes
- **Multiple Formats**: Dialogue exchanges or Q&A pairs

## 📁 Data Management

- **Automatic Backups**: `.backup1` files
- **Progress Tracking**: Resume from where you left off
- **Log Files**: Detailed generation logs in `logs/` directory
- **Configuration**: Customizable topics in `config/` directory

## 🛠️ Advanced Usage

### Custom Topics

Edit `config/sample_topics.txt` to add your own philosophical topics:

```python
[
    "The nature of consciousness",
    "Free will vs determinism",
    "The meaning of life",
    # Add your topics here...
]
```

### Fine-tuning Scripts

Use the generated datasets for model training:

```bash
# Supervised Fine-tuning
python scripts/sft_finetune.py

# DPO Improvement
python scripts/dpo_improvement.py

# Web Interface
python scripts/web_app.py
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see `requirements.txt`)

## 🤝 Contributing

Feel free to:
- Add new historical characters
- Improve dialogue generation prompts
- Enhance the character personality system
- Add new philosophical topics

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Related

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Socratic Method](https://en.wikipedia.org/wiki/Socratic_method)
- [Philosophical Dialogue](https://plato.stanford.edu/entries/ancient-dialogue/)