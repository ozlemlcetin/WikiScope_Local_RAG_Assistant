from src.entities import PEOPLE, PLACES

_PEOPLE_NAMES = {e.name.lower() for e in PEOPLE}
_PLACE_NAMES = {e.name.lower() for e in PLACES}

_PLACE_KEYWORDS = {"tower", "wall", "mahal", "canyon", "picchu", "colosseum", "sophia",
                   "statue", "pyramid", "everest", "stonehenge", "angkor", "petra",
                   "chichen", "acropolis", "vatican", "niagara", "amazon", "sahara", "reef",
                   "mountain", "city", "monument", "landmark", "wonder"}
_PERSON_KEYWORDS = {"born", "died", "scientist", "artist", "musician", "athlete", "writer",
                    "physicist", "mathematician", "philosopher", "painter", "footballer",
                    "singer", "inventor", "politician", "leader", "general", "emperor"}
_COMPARE_KEYWORDS = {"compare", "vs", "versus", "difference", "similarities", "similar", "both",
                     "between", "contrast"}

def route_query(query: str) -> str:
    q = query.lower()
    is_compare = any(kw in q for kw in _COMPARE_KEYWORDS)
    has_person = any(name in q for name in _PEOPLE_NAMES) or any(kw in q for kw in _PERSON_KEYWORDS)
    has_place = any(name in q for name in _PLACE_NAMES) or any(kw in q for kw in _PLACE_KEYWORDS)
    if is_compare or (has_person and has_place):
        return "both"
    if has_person:
        return "people"
    if has_place:
        return "places"
    return "both"
