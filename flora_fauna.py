"""
============================================================
Topic: Extraction of Indian Cultural Elements Using RAG
============================================================

Author Name(s): Khushi Jha

Short Description: Names of Indian flora and fauna extracted from
Kalidasa's works in Arthur W. Ryder's public-domain English translation
(Shakuntala, Urvashi, Malavika and Agnimitra, The Dynasty of Raghu,
The Birth of the War-god, The Cloud-Messenger and The Seasons) with a
Retrieval-Augmented Generation (RAG) pipeline. Passages were embedded with
nomic-embed-text, the passages most relevant to 12
flora/fauna queries were retrieved, and a local LLM (qwen2.5:7b,
run through Ollama) listed the plant and animal names in each passage.
Every name was then checked against the passage it came from, and
names not found in the text were discarded. A validation pass (the LLM
plus recorded author corrections) removed non-living items such as
ornaments, rivers and characters, and merged variants such as
"jasmine-blossoms" under one base name. See rag_pipeline.py and
validate_results.py.

Semester: Monsoon Semester 2026

Affiliation: Department of Humanities and Social Sciences,
Indian Institute of Technology (Indian School of Mines) Dhanbad
============================================================
"""

flora_RAG_extraction = [
    "Almond", "Amaranth", "Ashoka", "Bakul", "Bamboo",
    "Coral Tree", "Deodar", "Fig", "Jasmine", "Jujube",
    "Ketak (Screw-Pine)", "Lily", "Lodh", "Lotus", "Madder",
    "Mango", "Mushroom", "Nipa", "Panic Grass", "Plantain",
    "Reed", "Rice", "Rose-Apple", "Sacred Grass (Kusha)", "Siris",
    "Spring Creeper", "Trumpet-Flower", "Water-Lily",
]

fauna_RAG_extraction = [
    "Antelope", "Ape", "Bear", "Bee", "Bison",
    "Boar", "Buffalo", "Carp", "Cobra", "Cow",
    "Crane", "Cuckoo", "Deer", "Elephant", "Frog",
    "Heron", "Jackal", "Lion", "Nightingale", "Parrot",
    "Peacock", "Plover", "Ruddy Goose (Chakravaka)", "Snake", "Spider",
    "Starling", "Swan", "Tiger", "Yak",
]


# ---- Additional detail: names grouped by category ----

flora_by_category = {
    "creeper": ["Spring Creeper"],
    "crop_or_fruit": ["Plantain", "Rice", "Rose-Apple"],
    "flower": ["Amaranth", "Jasmine", "Ketak (Screw-Pine)", "Lily", "Lodh", "Lotus", "Siris", "Trumpet-Flower", "Water-Lily"],
    "herb_or_grass": ["Madder", "Mushroom", "Panic Grass", "Reed", "Sacred Grass (Kusha)"],
    "tree": ["Almond", "Ashoka", "Bakul", "Bamboo", "Coral Tree", "Deodar", "Fig", "Jujube", "Mango", "Nipa"],
}

fauna_by_category = {
    "amphibian": ["Frog"],
    "bird": ["Crane", "Cuckoo", "Heron", "Nightingale", "Parrot", "Peacock", "Plover", "Ruddy Goose (Chakravaka)", "Starling", "Swan"],
    "fish_or_aquatic": ["Carp"],
    "insect": ["Bee", "Spider"],
    "mammal": ["Antelope", "Ape", "Bear", "Bison", "Boar", "Buffalo", "Cow", "Deer", "Elephant", "Jackal", "Lion", "Tiger", "Yak"],
    "reptile": ["Cobra", "Snake"],
}

# Works of Kalidasa in which each name was found
found_in = {
    "Almond": ["Shakuntala"],
    "Amaranth": ["Shakuntala", "The Cloud-Messenger"],
    "Antelope": ["Malavika and Agnimitra; Urvashi"],
    "Ape": ["The Dynasty of Raghu"],
    "Ashoka": ["Malavika and Agnimitra; Urvashi", "Shakuntala", "The Cloud-Messenger", "The Dynasty of Raghu"],
    "Bakul": ["The Cloud-Messenger"],
    "Bamboo": ["The Cloud-Messenger"],
    "Bear": ["Shakuntala"],
    "Bee": ["Malavika and Agnimitra; Urvashi", "Shakuntala", "The Dynasty of Raghu"],
    "Bison": ["The Dynasty of Raghu"],
    "Boar": ["Shakuntala"],
    "Buffalo": ["Shakuntala"],
    "Carp": ["Shakuntala"],
    "Cobra": ["Malavika and Agnimitra; Urvashi", "The Dynasty of Raghu", "The Seasons"],
    "Coral Tree": ["Shakuntala", "The Cloud-Messenger"],
    "Cow": ["Shakuntala", "The Dynasty of Raghu"],
    "Crane": ["The Cloud-Messenger", "The Dynasty of Raghu"],
    "Cuckoo": ["Malavika and Agnimitra; Urvashi", "Shakuntala"],
    "Deer": ["Shakuntala", "The Dynasty of Raghu"],
    "Deodar": ["The Cloud-Messenger"],
    "Elephant": ["Malavika and Agnimitra; Urvashi", "Shakuntala", "The Birth of the War-god", "The Cloud-Messenger"],
    "Fig": ["The Cloud-Messenger"],
    "Frog": ["The Seasons"],
    "Heron": ["The Seasons"],
    "Jackal": ["Shakuntala", "The Birth of the War-god", "The Dynasty of Raghu"],
    "Jasmine": ["Shakuntala", "The Cloud-Messenger"],
    "Jujube": ["Shakuntala"],
    "Ketak (Screw-Pine)": ["The Seasons"],
    "Lily": ["Shakuntala", "The Dynasty of Raghu", "The Seasons"],
    "Lion": ["Shakuntala", "The Dynasty of Raghu"],
    "Lodh": ["The Cloud-Messenger"],
    "Lotus": ["Shakuntala", "The Birth of the War-god", "The Cloud-Messenger", "The Dynasty of Raghu", "The Seasons"],
    "Madder": ["The Cloud-Messenger", "The Dynasty of Raghu"],
    "Mango": ["Shakuntala", "The Birth of the War-god", "The Cloud-Messenger", "The Dynasty of Raghu", "The Seasons"],
    "Mushroom": ["The Cloud-Messenger"],
    "Nightingale": ["The Dynasty of Raghu"],
    "Nipa": ["The Seasons"],
    "Panic Grass": ["Shakuntala"],
    "Parrot": ["Shakuntala"],
    "Peacock": ["Malavika and Agnimitra; Urvashi", "Shakuntala", "The Cloud-Messenger", "The Dynasty of Raghu", "The Seasons"],
    "Plantain": ["The Cloud-Messenger"],
    "Plover": ["Shakuntala", "The Cloud-Messenger"],
    "Reed": ["The Cloud-Messenger"],
    "Rice": ["The Seasons"],
    "Rose-Apple": ["The Cloud-Messenger"],
    "Ruddy Goose (Chakravaka)": ["Malavika and Agnimitra; Urvashi"],
    "Sacred Grass (Kusha)": ["Shakuntala"],
    "Siris": ["Shakuntala", "The Cloud-Messenger"],
    "Snake": ["Shakuntala", "The Birth of the War-god", "The Seasons"],
    "Spider": ["The Dynasty of Raghu"],
    "Spring Creeper": ["Shakuntala"],
    "Starling": ["The Cloud-Messenger"],
    "Swan": ["Malavika and Agnimitra; Urvashi", "Shakuntala", "The Cloud-Messenger", "The Dynasty of Raghu"],
    "Tiger": ["The Dynasty of Raghu"],
    "Trumpet-Flower": ["The Seasons"],
    "Water-Lily": ["Shakuntala"],
    "Yak": ["Shakuntala", "The Cloud-Messenger"],
}

if __name__ == "__main__":
    print("Flora:", len(flora_RAG_extraction), "names")
    print("Fauna:", len(fauna_RAG_extraction), "names")
