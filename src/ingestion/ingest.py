import json
from pathlib import Path
from src.config import RAW_DIR, METADATA_DIR
from src.entities import ALL_ENTITIES, Entity
from src.ingestion.wikipedia_fetcher import fetch_wikipedia_text

def get_raw_path(entity: Entity) -> Path:
    subdir = "people" if entity.entity_type == "person" else "places"
    safe = entity.name.replace(" ", "_").replace("/", "_")
    return RAW_DIR / subdir / f"{safe}.txt"

def ingest_all(force: bool = False) -> None:
    metadata = []
    for entity in ALL_ENTITIES:
        path = get_raw_path(entity)
        if path.exists() and not force:
            print(f"  [skip] {entity.name}")
            meta = {"name": entity.name, "type": entity.entity_type,
                    "wikipedia_title": entity.wikipedia_title, "path": str(path)}
            metadata.append(meta)
            continue
        try:
            print(f"  [fetch] {entity.name} ...", end=" ", flush=True)
            text = fetch_wikipedia_text(entity.wikipedia_title)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"OK ({len(text):,} chars)")
            meta = {"name": entity.name, "type": entity.entity_type,
                    "wikipedia_title": entity.wikipedia_title, "path": str(path)}
            metadata.append(meta)
        except Exception as e:
            print(f"ERROR: {e}")

    entities_json = METADATA_DIR / "entities.json"
    entities_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetadata saved to {entities_json}")
