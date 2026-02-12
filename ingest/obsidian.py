from pathlib import Path
import yaml
import re
from typing import List, Dict


class ObsidianReader:
    def __init__(self, vault_path: str):
        vault = Path(vault_path)

        if not vault.exists():
            raise FileNotFoundError(f"Vault path '{vault}' does not exist.")
        self._vault = vault

    def read_all_vault_notes(self) -> List[Dict]:
        notes = []
        for md_file in self._vault.rglob("*.md"):
            if any(
                part.startswith(".") or part.startswith("_") or "Excalidraw" in part
                for part in md_file.parts
            ):
                continue

            notes.append(self._parse_note(md_file))

        return notes

    def _parse_note(self, md_file: Path) -> Dict:
        note_data = {
            "title": md_file.stem,
            "path": str(md_file.relative_to(self._vault)),
            "frontmatter": {},
            "tags": [],
            "content": "",
        }

        full_content = md_file.read_text()
        if full_content.startswith("---"):
            note_parts = full_content.split("---", 2)
            if len(note_parts) >= 3:
                try:
                    note_data["frontmatter"] = yaml.safe_load(note_parts[1]) or {}
                    note_data["content"] = note_parts[2]
                except yaml.YAMLError as err:
                    print(f"Couldn't parse frontmatter for '{md_file.stem}.md': {err}")
                    note_data["content"] = full_content
            else:
                note_data["content"] = full_content
        else:
            note_data["content"] = full_content

        note_data["tags"] = self._parse_tags(
            note_data["frontmatter"], note_data["content"]
        )

        return note_data

    def _parse_tags(self, frontmatter: Dict, note_body: str) -> List[str]:
        frontmatter_tags = frontmatter.get("tags", []) or []
        body_tags = re.findall(r"(?<=#)[^\s]*\b", note_body)
        return list(set(frontmatter_tags + body_tags))


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")

    reader = ObsidianReader(VAULT_PATH)
    notes = reader.read_all_vault_notes()
    print(f"Found {len(notes)} notes")
    for note in notes[10:13]:
        print(f"  - {note['title']} {len(note['content'])} chars, tags: {note['tags']}")
