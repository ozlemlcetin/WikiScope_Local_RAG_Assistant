import re
from src.retrieval.retriever import retrieve
from src.retrieval.context_builder import build_context
from src.generation.ollama_client import generate_text
from src.retrieval.query_router import route_query, get_matched_entities

FALLBACK = "I don't know based on the local Wikipedia data."

# Mirrors query_router._COMPARE_RE — kept here to avoid a circular import
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

# One unsupported "both…" sentence is already enough to trigger the safer retry
_HALLUCINATION_RE = re.compile(
    r"\bboth\s+(?:men|women|were|was|worked|had|studied|developed|contributed|became|are|of\s+them)\b"
    r"|\bsimilarly\b"
    r"|\blikewise\b"
    r"|\bin\s+the\s+same\s+(?:way|field|era|period|century)\b",
    re.IGNORECASE,
)


def _is_comparison(query: str) -> bool:
    return bool(_COMPARE_RE.search(query))


def _hallucination_count(text: str) -> int:
    return len(_HALLUCINATION_RE.findall(text))


_REFUSAL_RE = re.compile(
    r"\b("
    r"i can't|i cannot|cannot answer|can't answer|outside knowledge|"
    r"only allowed|not enough information|not provided in the excerpts|"
    r"don't have enough|unable to answer|can't fulfill|cannot fulfill|"
    r"requires outside|only use wikipedia"
    r")\b",
    re.IGNORECASE,
)


# Catches any diff line where "— vs —" and a fallback phrase appear on the same line,
# regardless of their relative order or what text appears between them.
_MALFORMED_DIFF_RE = re.compile(
    # fallback phrase anywhere after "— vs —" on the same line
    r"—\s*vs\s*—[^\n]*(?:no safe difference|no clean excerpt|no excerpt available|no directly comparable)"
    # fallback phrase anywhere before "— vs —" on the same line
    r"|(?:no safe difference|no clean excerpt|no excerpt available|no directly comparable)[^\n]*—\s*vs\s*—",
    re.IGNORECASE,
)

# Literal brace/bracket placeholders copied verbatim from the prompt template.
#
# Strings that actually appear at runtime in the prompts (after f-string interpolation):
#   [fact from Albert Einstein excerpt]       ← caught by \[fact\s+from
#   [max 4 facts]                             ← caught by \[max\s+\d+
#   [Albert Einstein: listed fact — vs — …]  ← caught by \blisted\s+fact (no leading [)
#   [max 3; write "No safe difference…"]      ← caught by \[max\s+\d+
#   {a}'s fact  /  {b}'s fact                ← caught by \{[ab]\}
#
# "Entity A" / "Entity B" are fallback values only when entity_names is empty;
# that never occurs on a real comparison query, so no pattern is needed for them.
_PLACEHOLDER_RE = re.compile(
    r"\{[ab]\}'?s?\s+fact"      # {a}'s fact  {b}'s fact
    r"|\[fact\s+from"           # [fact from Albert Einstein excerpt]
    r"|\blisted\s+fact\b"       # listed fact  (inside [Entity: listed fact — vs — …])
    r"|\[max\s+\d+"             # [max 4 facts] / [max 3; write …]
    r"|\[facts\s+from"          # [facts from excerpt]
    r"|\[only\s+explicitly"     # [only explicitly supported…]
    r"|\[max\s+\d+\s+facts",    # [max 4 facts total]
    re.IGNORECASE,
)


def _has_supported_diff_bullets(text: str) -> bool:
    """True if the response contains at least one real '— vs —' bullet under Supported differences."""
    m = re.search(r"Supported differences:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    return bool(re.search(r"^\s*[-*]\s+.*—\s*vs\s*—", m.group(1), re.MULTILINE))


def _has_standalone_no_safe_difference(text: str) -> bool:
    """True if a no-difference fallback sentence appears as its own line (not inside a bullet)."""
    return bool(re.search(
        r"^\s*(?:No safe difference found from the retrieved excerpts"
        r"|No directly comparable same-dimension differences were found in the retrieved excerpts)\.?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    ))


def _is_bad_comparison_answer(text: str) -> bool:
    """True if the answer contains hallucination patterns, a refusal,
    a malformed diff bullet, a literal prompt placeholder, or a contradictory
    diff section that mixes real bullets with the standalone fallback sentence."""
    if not text or not text.strip():
        return True
    if _has_supported_diff_bullets(text) and _has_standalone_no_safe_difference(text):
        return True
    return (
        _hallucination_count(text) >= 1
        or bool(_REFUSAL_RE.search(text))
        or bool(_MALFORMED_DIFF_RE.search(text))
        or bool(_PLACEHOLDER_RE.search(text))
    )


# ── Sentence extraction ────────────────────────────────────────────────────────

# Words that mark a mid-sentence continuation if a chunk starts with them
_CONTINUATION_RE = re.compile(
    r"^(?:and|or|but|which|that|who|where|when|however|also|although|"
    r"therefore|thus|hence|moreover|furthermore|nevertheless|yet|so)\b",
    re.IGNORECASE,
)

# Abbreviations whose trailing period must NOT be treated as a sentence boundary.
# Lookahead (?=[\s\d\u2009–-]) ensures we only protect when the period is
# followed by whitespace or a digit/dash (date contexts), not at true sentence ends.
_ABBREV_PROTECT_RE = re.compile(
    r"\b(?:c|ca|etc|vs|approx|no|vol|pp|fig|cf)\.(?=[\s\d\u2009–-])"
    r"|(?:e\.g|i\.e)\.(?=\s)",
    re.IGNORECASE,
)
_PLACEHOLDER = "\x00"  # null byte — never appears in Wikipedia text


def _protect_abbrevs(text: str) -> str:
    """Replace the period in known abbreviations with a placeholder to block false splits."""
    return _ABBREV_PROTECT_RE.sub(lambda m: m.group(0)[:-1] + _PLACEHOLDER, text)


def _restore_abbrevs(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


# Trailing words that indicate the sentence was cut mid-phrase by a false split.
_DANGLING_RE = re.compile(
    r"\s+(?:between|c|ca|AD|BC|and|or|of|in|at|by|for|from|to|the|a|an)\s*$",
    re.IGNORECASE,
)


def _clean_fact_for_display(fact: str, max_len: int = 320) -> str:
    """Remove dangling trailing fragments and cap length at a word boundary.

    _DANGLING_RE is applied in a loop so chained fragments like
    'between c' are fully stripped ('c' removed first, then 'between').
    """
    fact = _restore_abbrevs(fact).strip()
    while True:
        cleaned = _DANGLING_RE.sub("", fact).strip()
        if cleaned == fact:
            break
        fact = cleaned
    if fact and fact[-1] not in ".!?":
        fact += "."
    if len(fact) > max_len:
        cut = fact[:max_len].rfind(" ")
        fact = (fact[: cut if cut > max_len // 2 else max_len]).rstrip(".,;:") + "…"
    return fact


def _clean_sentences(text: str, max_count: int = 4) -> list[str]:
    """Return up to max_count complete, well-formed sentences from text.

    Skips fragments that:
    - start with a lowercase letter or digit (mid-sentence continuation)
    - start with punctuation (,  ;  :  —)
    - start with a continuation word (and, but, which …)
    - are shorter than 35 characters
    Protects abbreviations like c. / ca. / e.g. from false sentence splits.
    """
    protected = _protect_abbrevs(text.replace("\n", " "))
    raw = re.split(r"(?<=[.!?])\s+", protected)
    result: list[str] = []
    for s in raw:
        s = _restore_abbrevs(s).strip()
        if len(s) < 35:
            continue
        if not s[0].isupper():
            continue
        if s[0] in (",", ";", ":", "—", "-"):
            continue
        if _CONTINUATION_RE.match(s):
            continue
        result.append(s.rstrip(".") + ".")
        if len(result) >= max_count:
            break
    return result


_KNOWN_FALLBACK_PHRASES = {
    "no safe difference found from the retrieved excerpts",
    "no directly comparable same-dimension differences were found in the retrieved excerpts",
    "no clean excerpt available in local data",
    "no excerpt available in local data",
    # prompt template placeholders that the LLM might copy verbatim
    "[fact from excerpt only]",
    "[fact from excerpt]",
    "[listed fact]",
    "[max 4 facts]",
    "[max 4 facts total]",
}


def _is_fallback_sentence(text: str) -> bool:
    """Return True if text is a known placeholder/fallback string, not a real fact.
    Normalises by lower-casing and stripping trailing punctuation/space before
    checking the phrase set, so both "No safe difference found..." and
    "[fact from excerpt]" are caught regardless of trailing period.
    """
    normalised = text.lower().rstrip(".]").strip()
    return normalised in _KNOWN_FALLBACK_PHRASES or bool(_PLACEHOLDER_RE.search(text))


# ── Context helpers ────────────────────────────────────────────────────────────

def _per_entity_context(chunks: list[dict], entity_names: list[str]) -> dict[str, str]:
    """Group chunk texts by entity name, preserving declaration order."""
    grouped: dict[str, list[str]] = {e: [] for e in entity_names}
    for c in chunks:
        name = c["entity_name"]
        if name in grouped:
            grouped[name].append(c["text"])
    return {e: "\n\n".join(texts) for e, texts in grouped.items() if texts}


def _format_per_entity_context(grouped: dict[str, str]) -> str:
    parts = []
    for entity, text in grouped.items():
        parts.append(f"=== {entity} ===\n{text}")
    return "\n\n".join(parts)


# ── Prompts ────────────────────────────────────────────────────────────────────

def build_prompt(query: str, context: str) -> str:
    return f"""You are WikiScope, a Wikipedia-based assistant. Your job is to answer questions using ONLY the Wikipedia excerpts provided below.

Guidelines:
- Read the context carefully and answer the question directly.
- Use specific facts, dates, locations, and names from the context.
- If the context contains information about the topic, you MUST provide an answer based on it.
- Only say "I don't know based on the local Wikipedia data." if the context contains absolutely no information relevant to the question.
- Do not add information not present in the context.
- Be concise and factual.

Wikipedia Context:
{context}

Question: {query}

Answer:"""


def _build_strict_comparison_prompt(
    query: str, per_entity_ctx: str, entity_names: list[str]
) -> str:
    a = entity_names[0] if len(entity_names) > 0 else "Entity A"
    b = entity_names[1] if len(entity_names) > 1 else "Entity B"
    return f"""You are WikiScope, a strict Wikipedia-based assistant.

CRITICAL RULES — no exceptions:
1. Use ONLY the Wikipedia excerpts below. Zero outside knowledge.
2. Write max 4 bullet facts per entity from that entity's own excerpt section only.
3. "Directly supported differences" section:
   - ONLY reuse facts already listed in the entity sections above.
   - Each bullet must compare the SAME attribute/dimension for both entities.
     Good examples: birthplace vs birthplace, date vs date, field vs field.
     Bad examples: an architect fact vs an unrelated attribution fact.
   - Format: "{a}: <fact about dimension X> — vs — {b}: <fact about same dimension X>"
   - NEVER leave either side of "— vs —" empty. If you cannot fill both sides, omit the bullet.
   - Max 3 bullets.
   - If no same-dimension pair exists, write exactly:
     "No directly comparable same-dimension differences were found in the retrieved excerpts."
4. FORBIDDEN anywhere in your answer:
   "both men", "both were", "both worked", "both had", "both studied",
   "similarly", "likewise", "in the same way", "in the same field",
   "in the same era", "in the same period".
5. Do NOT add categories (nationality, education, religion, era) unless stated in the excerpt.
6. Do NOT refuse to answer. If you have excerpts for an entity, list facts from them.

Output EXACTLY this structure — nothing else:

Based only on the local Wikipedia excerpts:

{a}:
- [fact from {a} excerpt]
- [max 4 facts]

{b}:
- [fact from {b} excerpt]
- [max 4 facts]

Directly supported differences:
- [{a}: fact about dimension X — vs — {b}: fact about same dimension X]
- [max 3 bullets; write the safe fallback sentence if no same-dimension pair exists]

Wikipedia Excerpts:
{per_entity_ctx}

Question: {query}

Answer:"""


def _build_facts_only_prompt(
    query: str, per_entity_ctx: str, entity_names: list[str]
) -> str:
    """Retry prompt: no comparison section, pure per-entity facts."""
    return f"""You are WikiScope. List ONLY facts from each entity's Wikipedia excerpt.

STRICT RULES:
- Do NOT write any comparison section.
- Do NOT write any sentence starting with "Both", "Similarly", or "Likewise".
- List only facts explicitly stated in each entity's excerpt below.

Output format:

Based only on the local Wikipedia excerpts:

{chr(10).join(f"{e}:" + chr(10) + "- [fact from excerpt]" for e in entity_names)}

(Comparison section omitted — not enough shared evidence in local excerpts.)

Wikipedia Excerpts:
{per_entity_ctx}

Question: {query}

Answer:"""


# ── Same-dimension matcher ────────────────────────────────────────────────────
#
# Each entry is (dimension_name, compiled_regex).  Patterns are checked in order;
# the FIRST match wins.  Strict phrases are used so that compound adjectives like
# "German-born" do NOT trigger birth_origin, and "built for" / "built during" /
# "designed by" land in distinct groups rather than a single broad construction group.

_DIMENSION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # birth_origin — requires "was born", "born in/on/and", not bare "born"
    ("birth_origin",      re.compile(
        r"\bwas born\b|\bborn (?:in|on|and|c\.)\b|\bbirthplace\b|\bbirthdate\b",
        re.I,
    )),
    # death
    ("death",             re.compile(r"\b(?:died|death|passed away)\b", re.I)),
    # nationality / citizenship
    ("citizenship",       re.compile(r"\b(?:citizen(?:ship)?|nationality|naturalized)\b", re.I)),
    # relocation / migration
    ("migration",         re.compile(r"\bmoved? to\b|\bmigrat|\bemigrat|\bimmigrat", re.I)),
    # geographic location of a place
    ("location",          re.compile(
        r"\b(?:located|situated|lies (?:in|on)|stands (?:in|on))\b", re.I,
    )),
    # when something was built / completed / opened (date dimension)
    ("construction_date", re.compile(
        r"\bcompleted (?:in|ad|\d)"
        r"|\bconstructed (?:in|c\.)"
        r"|\bbuilt (?:in|during|between|c\.)"
        r"|\bopened (?:in|on)"
        r"|\berected (?:in|during|between)",
        re.I,
    )),
    # who designed / built / commissioned it
    ("designer_builder",  re.compile(
        r"\bdesigned by\b|\bbuilt by\b|\bcommissioned by\b"
        r"|\barchitects? of\b|\bby (?:the )?architect\b",
        re.I,
    )),
    # what something was used for / served as
    ("purpose_use",       re.compile(
        r"\bserved as\b|\bused (?:as|for)\b|\bbuilt for\b|\bintended (?:as|for)\b",
        re.I,
    )),
    # intellectual contribution (people)
    ("intellectual_work", re.compile(
        r"\b(?:invented|invention|developed|discovered|theory|contribution|published|created)\b",
        re.I,
    )),
    # awards and prizes
    ("award",             re.compile(
        r"\b(?:award(?:ed)?|prize|nobel|olympic medal)\b", re.I,
    )),
    # physical dimensions / height
    ("dimensions",        re.compile(
        r"\b(?:tall(?:er|est)?|height|elevation|metres?|meters?|feet)\b", re.I,
    )),
]


def _dimension_group(sentence: str) -> str | None:
    """Return the name of the first matching dimension group, or None."""
    for name, pattern in _DIMENSION_PATTERNS:
        if pattern.search(sentence):
            return name
    return None


def _find_same_dimension_pairs(
    facts_a: list[str], facts_b: list[str], max_pairs: int = 3
) -> list[tuple[str, str]]:
    """Return up to max_pairs of (fact_a, fact_b) that share an explicit dimension."""
    used_b: set[int] = set()
    pairs: list[tuple[str, str]] = []
    for fa in facts_a:
        dim = _dimension_group(fa)
        if dim is None:
            continue
        for j, fb in enumerate(facts_b):
            if j in used_b:
                continue
            if _dimension_group(fb) == dim:
                pairs.append((fa, fb))
                used_b.add(j)
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


# ── Deterministic diff-section builder ────────────────────────────────────────

_DIFF_HEADING_RE = re.compile(
    r"\n[ \t]*(?:Directly\s+supported\s+differences|Supported\s+differences)\s*:",
    re.IGNORECASE,
)
_NO_DIFF_LINE = (
    "- No directly comparable same-dimension differences were found in the retrieved excerpts."
)


def _strip_diff_section(text: str) -> str:
    """Remove the diff-section heading and everything after it, leaving entity summaries."""
    m = _DIFF_HEADING_RE.search(text)
    return text[:m.start()].rstrip() if m else text.rstrip()


def _build_deterministic_diff(chunks: list[dict], entity_names: list[str]) -> str:
    """Return a 'Directly supported differences:' block built purely from chunk text.

    Collects up to 6 clean sentences per entity (across all its chunks),
    then applies same-dimension matching.  Prefers false negatives.
    """
    if len(entity_names) < 2:
        return f"\n\nDirectly supported differences:\n{_NO_DIFF_LINE}"

    # Group and sort chunks by entity
    grouped: dict[str, list[dict]] = {e: [] for e in entity_names}
    for c in chunks:
        if c["entity_name"] in grouped:
            grouped[c["entity_name"]].append(c)
    for name in grouped:
        grouped[name].sort(key=lambda c: c["chunk_index"])

    # Collect up to 6 clean, non-fallback sentences per entity across its chunks
    facts: dict[str, list[str]] = {}
    for entity in entity_names:
        collected: list[str] = []
        for ec in grouped.get(entity, []):
            for s in _clean_sentences(ec["text"], max_count=6):
                if not _is_fallback_sentence(s) and s not in collected:
                    collected.append(s)
            if len(collected) >= 6:
                break
        facts[entity] = collected[:6]

    a, b = entity_names[0], entity_names[1]
    pairs = _find_same_dimension_pairs(facts.get(a, []), facts.get(b, []))

    lines = ["\n\nDirectly supported differences:"]
    if pairs:
        for fa, fb in pairs:
            lines.append(
                f"- {a}: {_clean_fact_for_display(fa).rstrip('.')}"
                f" — vs — "
                f"{b}: {_clean_fact_for_display(fb).rstrip('.')}"
            )
    else:
        lines.append(_NO_DIFF_LINE)

    return "\n".join(lines)


# ── Conservative Python fallback (no LLM synthesis) ───────────────────────────

def _conservative_answer(chunks: list[dict], entity_names: list[str]) -> str:
    """Deterministic per-entity answer; no LLM, no synthesis.

    1. Groups chunks by entity, sorted by chunk_index (intro first).
    2. Extracts up to 4 clean sentences per entity via _clean_sentences().
    3. Pairs facts only when both sides explicitly share a dimension keyword group.
       Prefers false negatives — uses the safe fallback rather than a wrong comparison.
    """
    # Group and sort
    grouped_chunks: dict[str, list[dict]] = {e: [] for e in entity_names}
    for c in chunks:
        name = c["entity_name"]
        if name in grouped_chunks:
            grouped_chunks[name].append(c)
    for name in grouped_chunks:
        grouped_chunks[name].sort(key=lambda c: c["chunk_index"])

    # Extract facts per entity (try each chunk until we get ≥ 2 clean sentences)
    facts: dict[str, list[str]] = {}
    for entity in entity_names:
        collected: list[str] = []
        for ec in grouped_chunks.get(entity, []):
            collected = _clean_sentences(ec["text"], max_count=4)
            if len(collected) >= 2:
                break
        facts[entity] = collected

    # Build per-entity sections
    lines = ["Based only on the local Wikipedia excerpts:\n"]
    for entity in entity_names:
        lines.append(f"\n{entity}:")
        if facts[entity]:
            for s in facts[entity]:
                lines.append(f"- {s}")
        else:
            lines.append("- No clean excerpt available in local data.")

    # Build diff section using same-dimension matching only — never positional zip.
    lines.append("\nDirectly supported differences:")
    if len(entity_names) >= 2:
        a, b = entity_names[0], entity_names[1]
        real_a = [f for f in facts.get(a, []) if not _is_fallback_sentence(f)]
        real_b = [f for f in facts.get(b, []) if not _is_fallback_sentence(f)]
        pairs = _find_same_dimension_pairs(real_a, real_b)
        if pairs:
            for fa, fb in pairs:
                lines.append(
                    f"- {a}: {_clean_fact_for_display(fa).rstrip('.')}"
                    f" — vs — "
                    f"{b}: {_clean_fact_for_display(fb).rstrip('.')}"
                )
        else:
            lines.append("- No directly comparable same-dimension differences were found in the retrieved excerpts.")
    else:
        lines.append("- No directly comparable same-dimension differences were found in the retrieved excerpts.")

    return "\n".join(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def answer(
    query: str,
    top_k: int = 5,
    debug: bool = False,
    model_name: str | None = None,
) -> dict:
    """Return a RAG answer dict.

    model_name overrides the generation model only; retrieval is unchanged.
    Omit or pass None to use the configured default LLM_MODEL.
    """
    route = route_query(query)
    chunks = retrieve(query, top_k=top_k, debug=debug)
    context = build_context(chunks)

    if debug:
        print(f"[DEBUG] context length : {len(context)} chars")
        print(f"[DEBUG] context preview: {context[:300]!r}")

    if not context.strip():
        if debug:
            print("[DEBUG] fallback: context is empty")
        return {"answer": FALLBACK, "chunks": [], "route": route, "context": ""}

    # ── Comparison path ────────────────────────────────────────────────────────
    if _is_comparison(query):
        matched = get_matched_entities(query)
        if debug:
            print(f"[DEBUG] comparison mode | matched_entities: {matched}")

        grouped = _per_entity_context(chunks, matched) if matched else {}
        per_entity_ctx = _format_per_entity_context(grouped) if grouped else context

        # Attempt 1: strict comparison prompt
        prompt1 = _build_strict_comparison_prompt(query, per_entity_ctx, matched or [])
        response = generate_text(prompt1, model_name=model_name)
        if debug:
            print(f"[DEBUG] comparison attempt 1: {response[:200]!r}")
            print(f"[DEBUG] hallucination count: {_hallucination_count(response)} | "
                  f"refusal: {bool(_REFUSAL_RE.search(response or ''))}")

        used_conservative = False
        if _is_bad_comparison_answer(response):
            # Attempt 2: facts-only prompt (no comparison synthesis)
            prompt2 = _build_facts_only_prompt(query, per_entity_ctx, matched or [])
            response = generate_text(prompt2, model_name=model_name)
            if debug:
                print(f"[DEBUG] comparison attempt 2 (facts-only): {response[:200]!r}")
                print(f"[DEBUG] attempt 2 bad: {_is_bad_comparison_answer(response)}")

            if _is_bad_comparison_answer(response):
                # Deterministic Python fallback — no LLM involved
                if debug:
                    print("[DEBUG] using conservative Python fallback")
                response = _conservative_answer(chunks, matched or [])
                used_conservative = True

        # For any LLM response (attempt 1 or 2), replace the LLM-generated diff
        # section with a deterministic one built from chunk text.
        # _conservative_answer already produces a correct diff section — skip it.
        if not used_conservative and response:
            body = _strip_diff_section(response)
            response = body + _build_deterministic_diff(chunks, matched or [])
            if debug:
                print(f"[DEBUG] deterministic diff rebuilt | entity_names={matched}")

        if not response or not response.strip():
            response = FALLBACK

        return {"answer": response, "chunks": chunks, "route": route, "context": context}

    # ── Standard path ──────────────────────────────────────────────────────────
    prompt = build_prompt(query, context)
    response = generate_text(prompt, model_name=model_name)

    if debug:
        print(f"[DEBUG] raw LLM response: {response!r}")

    if not response or not response.strip():
        response = FALLBACK

    return {"answer": response, "chunks": chunks, "route": route, "context": context}
