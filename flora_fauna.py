"""
============================================================
Topic: Extraction of Indian Cultural Elements Using RAG
============================================================

Author Name(s): Khushi Jha

Short Description: Names of Indian flora and fauna extracted with a
Retrieval-Augmented Generation (RAG) pipeline from a repository of five
Indian literary texts in public-domain English translation:
  1. Kalidasa, Shakuntala and Other Works (Sanskrit; tr. Arthur W. Ryder)
  2. Buddhist Birth Stories, or Jataka Tales, vol. 1 (Pali; tr. T. W.
     Rhys Davids)
  3. Bankim Chandra Chattopadhyay, The Poison Tree (Bengali novel
     Vishabriksha; tr. Miriam S. Knight)
  4. Sarojini Naidu, The Golden Threshold (written in English)
  5. Rabindranath Tagore, The Gardener (Bengali; tr. by the author)
Passages were embedded with nomic-embed-text, the passages of
each book most relevant to 12 flora/fauna queries were retrieved, and a
local LLM (qwen2.5:7b,
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
    "Amaranth", "Ashoka", "Babul (Acacia)", "Bakul", "Bamboo",
    "Banyan", "Barley", "Bel (Wood-Apple)", "Betel-Nut", "Brinjal",
    "Casuarina", "Champa", "Coconut", "Coral Tree", "Corn",
    "Cucumber", "Deodar", "Durva Grass", "Fig", "Gourd",
    "Hemp", "Henna", "Hyacinth", "Jackfruit", "Jasmine",
    "Jujube", "Kadam", "Ketak (Screw-Pine)", "Kusum", "Lichi",
    "Lily", "Linseed", "Lodh", "Lotus", "Madder",
    "Mango", "Mushroom", "Mustard", "Neem", "Nipa",
    "Oleander", "Orange", "Palm", "Panic Grass", "Peach",
    "Plantain", "Plum", "Pomegranate", "Poppy", "Pumpkin",
    "Reed", "Rice", "Rose", "Rose-Apple", "Sacred Grass (Kusha)",
    "Saffron", "Sandalwood", "Silk Cotton Tree", "Siris", "Spring Creeper",
    "Tamal", "Trumpet-Flower", "Tulip", "Tulsi", "Water Lily",
]

fauna_RAG_extraction = [
    "Antelope", "Ape", "Bat", "Bear", "Bee",
    "Bison", "Boar", "Buffalo", "Bull", "Butterfly",
    "Carp", "Cat", "Centipede", "Cobra", "Cockroach",
    "Cow", "Crane", "Crow", "Cuckoo", "Dahuk (Waterhen)",
    "Deer", "Dhadikula", "Dog", "Dove", "Duck",
    "Elephant", "Firefly", "Fly", "Frog", "Gazelle",
    "Goat", "Goose", "Gull", "Halcyon", "Hare",
    "Heron", "Horse", "Jackal", "Kite", "Koel",
    "Lion", "Lizard", "Monkey", "Mouse", "Musk Deer",
    "Musk-Rat", "Nightingale", "Oriole", "Owl", "Ox",
    "Paddy Bird (Pond Heron)", "Panther", "Papiya (Hawk-Cuckoo)", "Parrot", "Partridge",
    "Peacock", "Pig", "Pigeon", "Plover", "Quail",
    "Rat", "Sheep", "Snake", "Sparrow", "Spider",
    "Squirrel", "Starling", "Swan",
]


# ---- Additional detail: names grouped by category ----

flora_by_category = {
    "creeper": ["Spring Creeper"],
    "crop_or_fruit": ["Barley", "Bel (Wood-Apple)", "Brinjal", "Corn", "Cucumber", "Gourd", "Jackfruit", "Jujube", "Linseed", "Mustard", "Plantain", "Pumpkin", "Rice", "Rose-Apple"],
    "flower": ["Champa", "Hyacinth", "Jasmine", "Ketak (Screw-Pine)", "Kusum", "Lily", "Lodh", "Lotus", "Oleander", "Poppy", "Rose", "Saffron", "Trumpet-Flower", "Tulip", "Water Lily"],
    "herb_or_grass": ["Amaranth", "Durva Grass", "Hemp", "Henna", "Madder", "Mushroom", "Panic Grass", "Reed", "Sacred Grass (Kusha)", "Tulsi"],
    "tree": ["Ashoka", "Babul (Acacia)", "Bakul", "Bamboo", "Banyan", "Betel-Nut", "Casuarina", "Coconut", "Coral Tree", "Deodar", "Fig", "Kadam", "Lichi", "Mango", "Neem", "Nipa", "Orange", "Palm", "Peach", "Plum", "Pomegranate", "Sandalwood", "Silk Cotton Tree", "Siris", "Tamal"],
}

fauna_by_category = {
    "amphibian": ["Frog"],
    "bird": ["Crane", "Crow", "Cuckoo", "Dahuk (Waterhen)", "Dhadikula", "Dove", "Duck", "Goose", "Gull", "Halcyon", "Heron", "Kite", "Koel", "Nightingale", "Oriole", "Owl", "Paddy Bird (Pond Heron)", "Papiya (Hawk-Cuckoo)", "Parrot", "Partridge", "Peacock", "Pigeon", "Plover", "Quail", "Sparrow", "Starling", "Swan"],
    "fish_or_aquatic": ["Carp"],
    "insect": ["Bee", "Butterfly", "Centipede", "Cockroach", "Firefly", "Fly", "Spider"],
    "mammal": ["Antelope", "Ape", "Bat", "Bear", "Bison", "Boar", "Buffalo", "Bull", "Cat", "Cow", "Deer", "Dog", "Elephant", "Gazelle", "Goat", "Hare", "Horse", "Jackal", "Lion", "Monkey", "Mouse", "Musk Deer", "Musk-Rat", "Ox", "Panther", "Pig", "Rat", "Sheep", "Squirrel"],
    "reptile": ["Cobra", "Lizard", "Snake"],
}

# Books in which each name was found
found_in = {
    "Amaranth": ["Kalidasa"],
    "Antelope": ["Jataka Tales"],
    "Ape": ["Kalidasa"],
    "Ashoka": ["Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Babul (Acacia)": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Bakul": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Bamboo": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa", "Rabindranath Tagore, The Gardener"],
    "Banyan": ["Jataka Tales"],
    "Barley": ["Rabindranath Tagore, The Gardener"],
    "Bat": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Bear": ["Kalidasa"],
    "Bee": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa", "Rabindranath Tagore, The Gardener"],
    "Bel (Wood-Apple)": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Betel-Nut": ["Sarojini Naidu, The Golden Threshold"],
    "Bison": ["Kalidasa"],
    "Boar": ["Kalidasa"],
    "Brinjal": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Buffalo": ["Kalidasa", "Rabindranath Tagore, The Gardener"],
    "Bull": ["Jataka Tales"],
    "Butterfly": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Carp": ["Kalidasa"],
    "Casuarina": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Cat": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Centipede": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Champa": ["Sarojini Naidu, The Golden Threshold"],
    "Cobra": ["Kalidasa"],
    "Cockroach": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Coconut": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Sarojini Naidu, The Golden Threshold"],
    "Coral Tree": ["Kalidasa"],
    "Corn": ["Jataka Tales"],
    "Cow": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa", "Sarojini Naidu, The Golden Threshold"],
    "Crane": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Crow": ["Kalidasa"],
    "Cuckoo": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Cucumber": ["Jataka Tales"],
    "Dahuk (Waterhen)": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Deer": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa"],
    "Deodar": ["Rabindranath Tagore, The Gardener"],
    "Dhadikula": ["Sarojini Naidu, The Golden Threshold"],
    "Dog": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa"],
    "Dove": ["Kalidasa", "Rabindranath Tagore, The Gardener"],
    "Duck": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Rabindranath Tagore, The Gardener"],
    "Durva Grass": ["Jataka Tales"],
    "Elephant": ["Jataka Tales", "Kalidasa"],
    "Fig": ["Kalidasa"],
    "Firefly": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Sarojini Naidu, The Golden Threshold"],
    "Fly": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Frog": ["Kalidasa"],
    "Gazelle": ["Sarojini Naidu, The Golden Threshold"],
    "Goat": ["Jataka Tales"],
    "Goose": ["Jataka Tales"],
    "Gourd": ["Sarojini Naidu, The Golden Threshold"],
    "Gull": ["Sarojini Naidu, The Golden Threshold"],
    "Halcyon": ["Sarojini Naidu, The Golden Threshold"],
    "Hare": ["Jataka Tales"],
    "Hemp": ["Rabindranath Tagore, The Gardener"],
    "Henna": ["Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Heron": ["Kalidasa"],
    "Horse": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales"],
    "Hyacinth": ["Sarojini Naidu, The Golden Threshold"],
    "Jackal": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa", "Rabindranath Tagore, The Gardener"],
    "Jackfruit": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Jasmine": ["Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Jujube": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Kadam": ["Rabindranath Tagore, The Gardener"],
    "Ketak (Screw-Pine)": ["Kalidasa", "Sarojini Naidu, The Golden Threshold"],
    "Kite": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Koel": ["Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Kusum": ["Rabindranath Tagore, The Gardener"],
    "Lichi": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Lily": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa", "Sarojini Naidu, The Golden Threshold"],
    "Linseed": ["Rabindranath Tagore, The Gardener"],
    "Lion": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Lizard": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Lodh": ["Kalidasa"],
    "Lotus": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Madder": ["Kalidasa"],
    "Mango": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Monkey": ["Jataka Tales"],
    "Mouse": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Mushroom": ["Kalidasa"],
    "Musk Deer": ["Rabindranath Tagore, The Gardener"],
    "Musk-Rat": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Mustard": ["Rabindranath Tagore, The Gardener"],
    "Neem": ["Sarojini Naidu, The Golden Threshold"],
    "Nightingale": ["Kalidasa", "Sarojini Naidu, The Golden Threshold"],
    "Nipa": ["Kalidasa"],
    "Oleander": ["Sarojini Naidu, The Golden Threshold"],
    "Orange": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Oriole": ["Sarojini Naidu, The Golden Threshold"],
    "Owl": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Ox": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales"],
    "Paddy Bird (Pond Heron)": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Palm": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Rabindranath Tagore, The Gardener"],
    "Panic Grass": ["Kalidasa"],
    "Panther": ["Sarojini Naidu, The Golden Threshold"],
    "Papiya (Hawk-Cuckoo)": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Parrot": ["Sarojini Naidu, The Golden Threshold"],
    "Partridge": ["Jataka Tales"],
    "Peach": ["Sarojini Naidu, The Golden Threshold"],
    "Peacock": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Pig": ["Jataka Tales"],
    "Pigeon": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Plantain": ["Kalidasa"],
    "Plover": ["Kalidasa"],
    "Plum": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Pomegranate": ["Sarojini Naidu, The Golden Threshold"],
    "Poppy": ["Sarojini Naidu, The Golden Threshold"],
    "Pumpkin": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Quail": ["Jataka Tales"],
    "Rat": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Reed": ["Kalidasa"],
    "Rice": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Jataka Tales", "Kalidasa", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Rose": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Rose-Apple": ["Kalidasa"],
    "Sacred Grass (Kusha)": ["Kalidasa"],
    "Saffron": ["Rabindranath Tagore, The Gardener"],
    "Sandalwood": ["Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Sheep": ["Rabindranath Tagore, The Gardener", "Sarojini Naidu, The Golden Threshold"],
    "Silk Cotton Tree": ["Jataka Tales"],
    "Siris": ["Kalidasa", "Sarojini Naidu, The Golden Threshold"],
    "Snake": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Sparrow": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Spider": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Spring Creeper": ["Kalidasa"],
    "Squirrel": ["Sarojini Naidu, The Golden Threshold"],
    "Starling": ["Kalidasa"],
    "Swan": ["Bankim Chandra Chattopadhyay, The Poison Tree", "Kalidasa"],
    "Tamal": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Trumpet-Flower": ["Kalidasa"],
    "Tulip": ["Sarojini Naidu, The Golden Threshold"],
    "Tulsi": ["Bankim Chandra Chattopadhyay, The Poison Tree"],
    "Water Lily": ["Kalidasa"],
}

if __name__ == "__main__":
    print("Flora:", len(flora_RAG_extraction), "names")
    print("Fauna:", len(fauna_RAG_extraction), "names")
