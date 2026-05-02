from dataclasses import dataclass
from typing import Literal

@dataclass
class Entity:
    name: str
    entity_type: Literal["person", "place"]
    wikipedia_title: str

PEOPLE: list[Entity] = [
    Entity("Albert Einstein", "person", "Albert Einstein"),
    Entity("Marie Curie", "person", "Marie Curie"),
    Entity("Leonardo da Vinci", "person", "Leonardo da Vinci"),
    Entity("William Shakespeare", "person", "William Shakespeare"),
    Entity("Ada Lovelace", "person", "Ada Lovelace"),
    Entity("Nikola Tesla", "person", "Nikola Tesla"),
    Entity("Lionel Messi", "person", "Lionel Messi"),
    Entity("Cristiano Ronaldo", "person", "Cristiano Ronaldo"),
    Entity("Taylor Swift", "person", "Taylor Swift"),
    Entity("Frida Kahlo", "person", "Frida Kahlo"),
    Entity("Isaac Newton", "person", "Isaac Newton"),
    Entity("Charles Darwin", "person", "Charles Darwin"),
    Entity("Cleopatra", "person", "Cleopatra"),
    Entity("Napoleon Bonaparte", "person", "Napoleon Bonaparte"),
    Entity("Mahatma Gandhi", "person", "Mahatma Gandhi"),
    Entity("Nelson Mandela", "person", "Nelson Mandela"),
    Entity("Wolfgang Amadeus Mozart", "person", "Wolfgang Amadeus Mozart"),
    Entity("Vincent van Gogh", "person", "Vincent van Gogh"),
    Entity("Stephen Hawking", "person", "Stephen Hawking"),
    Entity("Aristotle", "person", "Aristotle"),
]

PLACES: list[Entity] = [
    Entity("Eiffel Tower", "place", "Eiffel Tower"),
    Entity("Great Wall of China", "place", "Great Wall of China"),
    Entity("Taj Mahal", "place", "Taj Mahal"),
    Entity("Grand Canyon", "place", "Grand Canyon"),
    Entity("Machu Picchu", "place", "Machu Picchu"),
    Entity("Colosseum", "place", "Colosseum"),
    Entity("Hagia Sophia", "place", "Hagia Sophia"),
    Entity("Statue of Liberty", "place", "Statue of Liberty"),
    Entity("Pyramids of Giza", "place", "Pyramids of Giza"),
    Entity("Mount Everest", "place", "Mount Everest"),
    Entity("Stonehenge", "place", "Stonehenge"),
    Entity("Angkor Wat", "place", "Angkor Wat"),
    Entity("Petra", "place", "Petra"),
    Entity("Chichen Itza", "place", "Chichen Itza"),
    Entity("Acropolis of Athens", "place", "Acropolis of Athens"),
    Entity("Vatican City", "place", "Vatican City"),
    Entity("Niagara Falls", "place", "Niagara Falls"),
    Entity("Amazon rainforest", "place", "Amazon rainforest"),
    Entity("Sahara Desert", "place", "Sahara"),
    Entity("Great Barrier Reef", "place", "Great Barrier Reef"),
]

ALL_ENTITIES: list[Entity] = PEOPLE + PLACES
