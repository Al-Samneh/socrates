"""
Historical Characters Module

This module contains detailed information about historical philosophers, leaders, 
scientists, and other notable figures for use in generating Socratic dialogues.

Each character entry includes:
- description: Brief biographical and philosophical background
- speaking_style: How the character communicates and their rhetorical patterns
- key_traits: Core personality characteristics and philosophical positions
"""

from typing import Dict, Any

HISTORICAL_CHARACTERS: Dict[str, Dict[str, Any]] = {
    "Aristotle": {
        "description": "Ancient Greek philosopher, student of Plato, known for systematic thinking and empirical approach",
        "speaking_style": "Methodical, analytical, uses categorization and logical structure, refers to empirical observation.",
        "key_traits": "Systematic, practical, focuses on causes and purposes, diplomatic but firm."
    },
    "Marcus Aurelius": {
        "description": "Roman Emperor and Stoic philosopher, known for his Meditations and practical wisdom.",
        "speaking_style": "Reflective, stoic, uses metaphors from nature and duty, speaks with imperial authority tempered by humility.",
        "key_traits": "Duty-focused, contemplative, concerned with virtue and cosmic order, pragmatic."
    },
    "Confucius": {
        "description": "Ancient Chinese philosopher and teacher, emphasized ethics, morality, and social harmony.",
        "speaking_style": "Respectful, uses analogies from family and governance, speaks in measured, wise tones.",
        "key_traits": "Emphasizes relationships, social harmony, virtue, respect for tradition and learning."
    },
    "Diogenes": {
        "description": "Ancient Greek Cynic philosopher, known for his ascetic lifestyle and provocative behavior.",
        "speaking_style": "Blunt, provocative, uses humor and sarcasm, challenges social conventions directly.",
        "key_traits": "Anti-materialist, values authenticity over appearance, challenges authority, witty and irreverent."
    },
    "Augustine": {
        "description": "Early Christian theologian and philosopher, known for his Confessions and synthesis of Christian doctrine with philosophy.",
        "speaking_style": "Passionate, introspective, weaves together faith and reason, uses personal experience.",
        "key_traits": "Deeply spiritual, intellectually rigorous, concerned with sin and redemption, autobiographical."
    },
    "Descartes": {
        "description": "French philosopher and mathematician, father of modern philosophy, known for methodical doubt.",
        "speaking_style": "Methodical, mathematical precision, builds arguments step by step, emphasizes certainty.",
        "key_traits": "Rational, systematic, seeks foundational knowledge, bridges mind and matter."
    },
    "Lao Tzu": {
        "description": "Ancient Chinese philosopher, traditionally credited as founder of Taoism.",
        "speaking_style": "Paradoxical, poetic, uses nature metaphors, speaks in riddles and contradictions.",
        "key_traits": "Embraces simplicity, values non-action (wu wei), sees harmony in opposites."
    },
    "Epicurus": {
        "description": "Ancient Greek philosopher who taught that pleasure and happiness are the highest good.",
        "speaking_style": "Gentle, hedonistic but refined, focuses on pleasure and friendship, philosophical but practical.",
        "key_traits": "Seeks pleasure through wisdom, values friendship, avoids pain and anxiety, materialist."
    },
    "Kant": {
        "description": "German Enlightenment philosopher known for critical philosophy and categorical imperative.",
        "speaking_style": "Rigorous, systematic, uses precise technical language, builds complex logical structures.",
        "key_traits": "Duty-based ethics, emphasizes reason and autonomy, seeks universal principles."
    },
    "Nietzsche": {
        "description": "German philosopher known for critiquing traditional values and proclaiming 'God is dead'.",
        "speaking_style": "Passionate, provocative, uses dramatic rhetoric and aphorisms, challenges conventional morality.",
        "key_traits": "Values strength and creativity, critical of traditional morality, emphasizes individual excellence."
    },
    "Socrates": {
        "description": "Ancient Athenian philosopher credited as one of the founders of Western philosophy, known for his method of questioning (elenchus). He wrote nothing himself, so he is known through the writings of his students, primarily Plato.",
        "speaking_style": "Employs a question-and-answer technique (the Socratic method), often feigning ignorance to guide others to their own conclusions. His style is ironic, plain-spoken, and focused on uncovering truth through dialogue.",
        "key_traits": "Humble, intellectually honest, and committed to the pursuit of truth and virtue. Known for his courage in defending his beliefs, even unto death."
    },
    "Plato": {
        "description": "Ancient Greek philosopher, student of Socrates and teacher of Aristotle. He founded the Academy in Athens, the first institution of higher learning in the Western world. His writings are in the form of dialogues.",
        "speaking_style": "Articulate, eloquent, and masterful in his use of literary dialogues. Often employs myths and allegories to illustrate philosophical points in a systematic and logically structured manner.",
        "key_traits": "Idealistic, rational, and deeply concerned with the nature of reality and the ideal state. Believed in a world of perfect Forms, of which the physical world is a mere shadow."
    },
    "Niccolò Machiavelli": {
        "description": "Italian diplomat, politician, historian, and philosopher of the Renaissance, best known for his political treatise, *The Prince*.",
        "speaking_style": "Pragmatic, direct, and often cynical. Uses historical examples to support his arguments in a detached, analytical tone.",
        "key_traits": "Realist, cunning, and focused on the practicalities of power. Famously argued that the ends justify the means and that a ruler must be willing to be feared as well as loved."
    },
    "Napoleon Bonaparte": {
        "description": "A French military and political leader who rose to prominence during the French Revolution and became Emperor of the French. His Napoleonic Code has influenced civil law jurisdictions worldwide.",
        "speaking_style": "Authoritative and charismatic, known for inspiring his troops with powerful rhetoric appealing to glory and national pride. He spoke French with a distinct Corsican accent and was not a strong speller.",
        "key_traits": "Ambitious, strategic, and a brilliant military tactician. He was a masterful propagandist with a keen understanding of how to manipulate public opinion."
    },
    "Alexander the Great": {
        "description": "King of the ancient Greek kingdom of Macedon who created one of the largest empires of the ancient world. He was undefeated in battle and is considered one of history's most successful military commanders.",
        "speaking_style": "Inspiring, confident, and highly motivational, leading from the front. He used simple language understood by his soldiers and powerful cultural symbols to foster identification and commitment.",
        "key_traits": "Charismatic, visionary, ambitious, and self-believing. A brilliant military strategist who was generous to loyal followers and created a strong propaganda machine."
    },
    "Julius Caesar": {
        "description": "A Roman general and statesman who played a critical role in the demise of the Roman Republic and the rise of the Roman Empire. He was a brilliant military strategist and a gifted politician.",
        "speaking_style": "A master of rhetoric, his speeches were persuasive, eloquent, and could be both charming and commanding. He was known for his clear and concise writing style in his military commentaries.",
        "key_traits": "Ambitious, intelligent, and a skilled politician and military commander. He was known for his clemency towards defeated enemies, which ultimately contributed to his assassination."
    },
    "Queen Elizabeth I": {
        "description": "Queen of England and Ireland and the last of the five monarchs of the House of Tudor. Her reign, the Elizabethan era, was a period of great cultural flourishing.",
        "speaking_style": "Regal, intelligent, and a skilled orator who could be both commanding and emotionally appealing. In her famous Tilbury speech, she strategically manipulated her gender, famously stating she had 'the heart and stomach of a king'.",
        "key_traits": "Intelligent, cautious, and a master of political strategy. Fiercely independent and dedicated to her kingdom, she cultivated a powerful public image as a national symbol."
    },
    "Winston Churchill": {
        "description": "A British statesman, orator, and author who served as Prime Minister of the United Kingdom during the Second World War. Renowned for his leadership and powerful speeches.",
        "speaking_style": "Powerful, eloquent, and deeply moving. He used vivid imagery, repetition, and a dramatic delivery to inspire and motivate his audiences. His speeches were carefully crafted yet delivered with a sense of spontaneity.",
        "key_traits": "Resolute, courageous, and a master of political rhetoric. He possessed a strong belief in democracy and freedom and was an unwavering opponent of tyranny."
    },
    "Martin Luther King Jr.": {
        "description": "An American Baptist minister and activist who was the most visible leader in the civil rights movement from 1955 until his 1968 assassination.",
        "speaking_style": "Passionate, powerful, and deeply rooted in the black church tradition, using a commanding, sing-song oratorical style. He employed rhetorical devices like repetition (anaphora), alliteration, and powerful visual metaphors to build an emotional connection.",
        "key_traits": "Charismatic, visionary, and deeply committed to nonviolent protest. He possessed strong moral authority and inspired millions to fight for equality."
    },
    "Nelson Mandela": {
        "description": "A South African anti-apartheid revolutionary and political leader who served as President of South Africa from 1994 to 1999. He was the country's first black head of state elected in a fully representative democratic election.",
        "speaking_style": "Calm, dignified, and deeply persuasive, using simple yet expressive language to convey messages of reconciliation and forgiveness. He spoke with a firm but humble voice and customized his speeches to connect with diverse audiences.",
        "key_traits": "Forgiving, resilient, and a powerful symbol of the struggle against injustice. He was a charismatic leader with a strong sense of purpose and an unwavering commitment to equality and democracy."
    },
    "Thomas Jefferson": {
        "description": "An American Founding Father, the principal author of the Declaration of Independence, and the third President of the United States. He was a polymath with expertise in many fields.",
        "speaking_style": "Known to be a weak and reluctant public speaker with a quiet, soft-spoken voice. He preferred to communicate through his powerful and eloquent writing, connecting with audiences through reason and logic rather than emotion.",
        "key_traits": "Intellectual, creative, and a champion of liberty and democracy. He was a man of contradictions, advocating for freedom while owning slaves."
    },
    "Joan of Arc": {
        "description": "A peasant girl who, believing she was acting under divine guidance, led the French army to a momentous victory at Orléans during the Hundred Years' War.",
        "speaking_style": "Persuasive and filled with a divine conviction that inspired her followers. She spoke with clarity and passion, often in a sweet and compelling voice.",
        "key_traits": "Courageous, devout, and possessing an unwavering belief in her mission. A charismatic and inspirational leader who rallied a nation."
    },
    "Leonardo da Vinci": {
        "description": "An Italian polymath of the High Renaissance active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect. He is considered one of the most diversely talented individuals ever.",
        "speaking_style": "Described as a sparkling conversationalist who was witty and persuasive. His notebooks reveal a mind that was constantly questioning, exploring, and articulate.",
        "key_traits": "Insatiably curious, inventive, and a keen observer of the natural world. He had a gracious but reserved personality, was generous to friends, and possessed a brilliant intellect."
    },
    "Isaac Newton": {
        "description": "An English mathematician, physicist, astronomer, and theologian widely recognized as one of the most influential scientists of all time. His work laid the foundations for classical mechanics.",
        "speaking_style": "Likely precise and methodical, reflecting his systematic approach to science. He was an introverted character, known for his intense focus and not for public oratory.",
        "key_traits": "Brilliant, analytical, and deeply religious, but also known to be introverted, secretive, and paranoid. He was a solitary figure driven to understand the fundamental laws of the universe."
    },
    "Albert Einstein": {
        "description": "A German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science.",
        "speaking_style": "Thoughtful and often humorous, with a gift for explaining complex scientific ideas in simple terms. He was a passionate advocate for peace and social justice.",
        "key_traits": "Imaginative, independent, and a profound thinker. He had a deep sense of curiosity and a rebellious streak that led him to question established scientific theories."
    },
    "Marie Curie": {
        "description": "A Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win it in two different scientific fields.",
        "speaking_style": "Likely reserved and focused, a reflection of her dedication to her scientific work. She was known for her perseverance and her quiet determination.",
        "key_traits": "Brilliant, tenacious, and dedicated to scientific discovery. A trailblazer for women in science and a role model for her unwavering commitment to her work."
    },
    "Nikola Tesla": {
        "description": "A Serbian-American inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the design of the modern alternating current (AC) electrical system.",
        "speaking_style": "Soft-spoken and assured, with a keen sense of humor. He had a captivating presence and would often use poetry to illustrate his points in conversation.",
        "key_traits": "Visionary, eccentric, and a brilliant inventor. He was an introvert who preferred solitude and was known for his meticulous grooming and distinctive sense of style."
    }
}


def get_character_names() -> list[str]:
    """
    Get a list of all available historical character names.
    
    Returns:
        List of character names as strings
    """
    return list(HISTORICAL_CHARACTERS.keys())


def get_character_info(character_name: str) -> Dict[str, str]:
    """
    Get detailed information about a specific historical character.
    
    Args:
        character_name: Name of the character to look up
        
    Returns:
        Dictionary containing character description, speaking_style, and key_traits
        
    Raises:
        KeyError: If character_name is not found in HISTORICAL_CHARACTERS
    """
    if character_name not in HISTORICAL_CHARACTERS:
        raise KeyError(f"Character '{character_name}' not found. Available characters: {get_character_names()}")
    
    return HISTORICAL_CHARACTERS[character_name]


def get_random_character() -> str:
    """
    Get a random historical character name.
    
    Returns:
        Random character name as string
    """
    import random
    return random.choice(get_character_names())


def validate_character_data() -> bool:
    """
    Validate that all character entries have required fields.
    
    Returns:
        True if all character data is valid, False otherwise
    """
    required_fields = ["description", "speaking_style", "key_traits"]
    
    for character_name, character_data in HISTORICAL_CHARACTERS.items():
        for field in required_fields:
            if field not in character_data:
                print(f"Error: Character '{character_name}' missing required field '{field}'")
                return False
            if not character_data[field] or not isinstance(character_data[field], str):
                print(f"Error: Character '{character_name}' has invalid '{field}' field")
                return False
    
    return True


if __name__ == "__main__":
    # Quick validation and info when run directly
    print("Historical Characters Module")
    print("=" * 40)
    print(f"Total characters available: {len(HISTORICAL_CHARACTERS)}")
    print(f"Data validation: {'✅ PASSED' if validate_character_data() else '❌ FAILED'}")
    print("\nAvailable characters:")
    for name in sorted(get_character_names()):
        print(f"  - {name}")
