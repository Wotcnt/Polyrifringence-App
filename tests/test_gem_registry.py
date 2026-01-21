from pathlib import Path
import json

def test_gem_registry_json_is_valid():
    p = Path(__file__).resolve().parents[1] / "docs" / "gem_registry.json"
    assert p.exists(), "docs/gem_registry.json missing"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, (dict, list)), "gem_registry.json must be JSON object or array"
