import yaml

def load_taxonomy(path: str = "industries.yaml") -> list[dict]:
    """
    Loads the hierarchical taxonomy and flattens it into a list of nodes.
    Returns a list of dicts: {id, name}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    flat = []

    def recurse(nodes):
        for node in nodes:
            flat.append({
                "id": node["id"],
                "name": node["name"]
            })
            if "children" in node:
                recurse(node["children"])

    recurse(data["taxonomy"])
    return flat

