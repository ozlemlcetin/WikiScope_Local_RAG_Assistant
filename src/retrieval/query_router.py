import re
from src.entities import PEOPLE, PLACES, Entity

_ALL_ENTITIES: list[Entity] = PEOPLE + PLACES
_PEOPLE_NAMES = {e.name.lower(): e.name for e in PEOPLE}
_PLACE_NAMES = {e.name.lower(): e.name for e in PLACES}

# Entity-specific terms that only appear in place contexts
_PLACE_KEYWORDS = {
    "tower", "wall", "mahal", "canyon", "picchu", "colosseum", "sophia",
    "statue", "pyramid", "everest", "stonehenge", "angkor", "petra",
    "chichen", "acropolis", "vatican", "niagara", "amazon", "sahara", "reef",
    "monument", "landmark", "wonder", "temple", "palace", "castle", "ruins",
    # attribute-based place indicators
    "place", "places", "located", "location", "where", "site", "attraction",
    "destination", "building", "structure", "park", "ocean", "sea", "river",
    "lake", "desert", "mountain", "island", "continent", "region", "country",
    "city", "capital", "constructed", "built", "architecture",
    # countries and cities in our dataset's context
    "turkey", "egypt", "france", "china", "india", "peru", "italy", "greece",
    "cambodia", "jordan", "mexico", "australia", "nepal", "usa", "america",
    "england", "brazil", "spain",
    "istanbul", "cairo", "paris", "rome", "athens", "agra", "beijing",
    "lima", "cusco", "giza",
}

_PERSON_KEYWORDS = {
    "born", "died", "scientist", "artist", "musician", "athlete", "writer",
    "physicist", "mathematician", "philosopher", "painter", "footballer",
    "singer", "inventor", "politician", "leader", "general", "emperor",
    "who", "biography", "life", "career", "achievements", "discovery",
    "theory", "award", "nobel", "olympic",
}

# Regex-based comparison detection with word boundaries (replaces the old set).
# Covers: "difference between", "different from", "compare", "comparison",
#         "versus", "vs", "vs.", "similarities", "differences".
_COMPARE_RE = re.compile(
    r"\bdifference\s+between\b"
    r"|\bdifferent\s+from\b"
    r"|\bcompare\b"
    r"|\bcomparison\b"
    r"|\bversus\b"
    r"|\bvs\.?\b"
    r"|\bsimilarities\b"
    r"|\bdifferences\b"
    r"|\bcontrast\b",
    re.IGNORECASE,
)

# Country/city terms used for chunk reranking (superset of routing terms)
_LOCATION_TERMS = {
    "turkey", "turkish", "istanbul", "ankara",
    "egypt", "egyptian", "cairo", "giza",
    "france", "french", "paris",
    "china", "chinese", "beijing",
    "india", "indian", "agra", "delhi",
    "peru", "peruvian", "lima", "cusco", "machu",
    "italy", "italian", "rome", "roman",
    "greece", "greek", "athens",
    "cambodia", "khmer", "angkor",
    "jordan", "petra",
    "mexico", "mexican",
    "australia", "australian",
    "nepal", "himalaya", "himalayan",
    "usa", "american", "america", "new york", "washington",
    "england", "british", "london",
    "brazil", "amazon",
    "sahara", "africa", "african",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def get_matched_entities(query: str) -> list[str]:
    """Return canonical entity names found verbatim in the query."""
    q = query.lower()
    matched = []
    for lower_name, canonical in {**_PEOPLE_NAMES, **_PLACE_NAMES}.items():
        if lower_name in q:
            matched.append(canonical)
    return matched


def get_location_keywords(query: str) -> set[str]:
    """Extract country/city terms from the query for chunk reranking."""
    q = query.lower()
    return {term for term in _LOCATION_TERMS if term in q}


def route_query(query: str) -> str:
    q = query.lower()
    qwords = _words(q)
    is_compare = bool(_COMPARE_RE.search(q))
    matched = get_matched_entities(q)
    matched_lower = {m.lower() for m in matched}
    has_person = bool(matched_lower & _PEOPLE_NAMES.keys()) or bool(qwords & _PERSON_KEYWORDS)
    has_place = bool(matched_lower & _PLACE_NAMES.keys()) or bool(qwords & _PLACE_KEYWORDS)
    if is_compare:
        return "both"
    if has_person and has_place:
        return "both"
    if has_person:
        return "people"
    if has_place:
        return "places"
    return "both"
