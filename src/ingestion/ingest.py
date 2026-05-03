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
    failed = []
    skipped = downloaded = 0

    for entity in ALL_ENTITIES:
        path = get_raw_path(entity)
        if path.exists() and not force:
            print(f"  [skip] {entity.name}")
            metadata.append({"name": entity.name, "type": entity.entity_type,
                              "wikipedia_title": entity.wikipedia_title, "path": str(path)})
            skipped += 1
            continue
        try:
            print(f"  [fetch] {entity.name} ...", end=" ", flush=True)
            text = fetch_wikipedia_text(entity.wikipedia_title, entity.fallback_titles)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"OK ({len(text):,} chars)")
            metadata.append({"name": entity.name, "type": entity.entity_type,
                              "wikipedia_title": entity.wikipedia_title, "path": str(path)})
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({"name": entity.name, "type": entity.entity_type,
                           "wikipedia_title": entity.wikipedia_title, "error": str(e)})

    entities_json = METADATA_DIR / "entities.json"
    entities_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    failed_json = METADATA_DIR / "failed_entities.json"
    failed_json.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- Summary ---")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Downloaded:              {downloaded}")
    print(f"  Failed:                  {len(failed)}")
    if failed:
        print(f"  Failed entities saved to: {failed_json}")
        for f in failed:
            print(f"    - {f['name']}: {f['error']}")
    print(f"  Metadata saved to: {entities_json}")
