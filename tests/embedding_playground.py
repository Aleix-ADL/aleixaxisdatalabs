import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings import get_embedding
from itertools import combinations


TEXTS = [
    "Pasta restaurants",
    "Hotdog stands",
    "Event agencies",
    "Email marketing SaaS for ecommerce",
    "Burger places",
    "Sushi restaurants near me",
    "Italian restaurants",
    "Wedding planners",
    "CRM SaaS for sales teams",
    "SaaS analytics platform",
    "Chicago pizza places",
    "Fast food",
    "Taco restaurants",
    "Enterprise recruiting software",
    "Local bakeries",
    "Steakhouse restaurants",
    "Yoga studios",
    "Ecommerce marketing automation tools",
    "PR agencies for tech startups",
    "Comedy events"
]

TEXTSI = [
    "Pasta restaurants specializing in fresh homemade noodles",
    "Hotdog stands offering gourmet toppings and street-food style menus",
    "Full-service event agencies that plan corporate conferences and private parties",
    "Email marketing SaaS platforms built for ecommerce brands",
    "Burger places known for smash burgers and fast-casual dining",
    "Sushi restaurants near me that offer omakase and premium rolls",
    "Traditional Italian restaurants focusing on regional cuisine",
    "Wedding planners specializing in luxury destination weddings",
    "CRM SaaS designed for sales teams in growing small businesses",
    "Advanced SaaS analytics platforms for tracking user behavior",
    "Chicago pizza places known for deep-dish and stuffed crust options",
    "Fast-food chains serving burgers, fries, and affordable meals",
    "Taco restaurants offering authentic Mexican street tacos",
    "Enterprise recruiting software for large HR departments",
    "Local bakeries selling artisan bread and pastries",
    "Steakhouse restaurants serving premium cuts and dry-aged beef",
    "Yoga studios offering classes for beginners and advanced practitioners",
    "Ecommerce marketing automation tools for personalized campaigns",
    "PR agencies specializing in media outreach for tech startups",
    "Comedy events featuring stand-up shows and improv nights",
    
    # Added + Improved
    "Food delivery services connecting restaurants with local customers",
    "Vegan restaurants with plant-based menus and organic ingredients",
    "Specialty coffee shops serving espresso, cold brew, and pastries",
    "Restaurant POS software for managing orders and payments",
    "Small-business invoicing platforms with automated billing features",
    "Catering companies offering menus for weddings and corporate events",
    "Gluten-free bakeries focused on healthy and allergen-friendly desserts",
    "Online scheduling tools for salons, spas, and service providers",
    "Payroll SaaS designed for small to mid-sized businesses",
    "Mediterranean restaurants serving grilled meats, mezze, and fresh salads"
]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_evaluation():
    print("Embedding all texts...")
    embeddings = [get_embedding(t) for t in TEXTS]

    report = []

    for i, text in enumerate(TEXTS):
        sims = []
        for j, other in enumerate(TEXTS):
            if i == j:
                continue
            score = cosine(embeddings[i], embeddings[j])
            sims.append((other, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]

        report.append({
            "text": text,
            "top3": top3,
        })

    return report


def generate_markdown_output(report):
    lines = ["# Embedding Similarity Report\n"]

    for item in report:
        lines.append(f"## Query: **{item['text']}**")
        for other, score in item["top3"]:
            lines.append(f"- {score:.4f} — {other}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_evaluation()
    md = generate_markdown_output(report)

    with open("embedding_similarity_report.md", "w") as f:
        f.write(md)

    print("\nReport saved to embedding_similarity_report.md")
