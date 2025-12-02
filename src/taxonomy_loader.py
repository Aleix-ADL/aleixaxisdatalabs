# Old code
    # import yaml
    # def load_taxonomy(path: str = "industries.yaml") -> list[dict]:
    #     """
    #     Loads the hierarchical taxonomy and flattens it into a list of nodes.
    #     Returns a list of dicts: {id, name}
    #     """
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = yaml.safe_load(f)

    #     flat = []

    #     def recurse(nodes):
    #         for node in nodes:
    #             flat.append({
    #                 "id": node["id"],
    #                 "name": node["name"]
    #             })
    #             if "children" in node:
    #                 recurse(node["children"])

    #     recurse(data["taxonomy"])
    #     return flat

import yaml

def load_taxonomy(path: str = "industries.yaml") -> list[dict]:
    """
    Loads the hierarchical taxonomy and flattens it into a list of nodes.
    Each node includes: id, name, description, keywords, and a combined text field.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    flat = []

    def recurse(nodes):
        for node in nodes:
            name = node.get("name", "")
            description = node.get("description", "")
            keywords = node.get("keywords", [])
 
            # Build final text to embed
            text_parts = [name]
            if description:
                text_parts.append(description)
            if keywords:
                text_parts.append("Keywords: " + ", ".join(keywords))

            combined_text = " | ".join(text_parts)

            flat.append({
                "id": node["id"],
                "name": name,
                "description": description,
                "keywords": keywords,
                "text": combined_text
            })

            if "children" in node:
                recurse(node["children"])

    recurse(data["taxonomy"])
    return flat
