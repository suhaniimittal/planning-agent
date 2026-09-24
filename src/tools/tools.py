"""Tool definitions for the project."""

from __future__ import annotations

import asyncio
from aetherion_sdk import tool

def count_words(text: str) -> int:
    """Count words in a text string."""
    return len([w for w in text.strip().split() if w])

@tool()
async def greet_user(name: str) -> str:
    """Generate a personalized greeting."""
    await asyncio.sleep(0.1)
    return f"Hello, {name}! Welcome to Aetherion."


@tool()
async def analyze_text(text: str) -> dict:
    """Analyze the provided text using a utility function from utils."""
    await asyncio.sleep(0.05)
    words = count_words(text)
    chars = len(text)
    return {
        "word_count": words,
        "char_count": chars,
        "avg_word_length": (chars / words) if words else 0.0,
    }

@tool()
async def enrich_greeting(name: str, emphasis: bool = True, delay_ms: int = 200) -> str:
    """Enrich a greeting with optional emphasis and simulated delay."""
    await asyncio.sleep(delay_ms / 1000)
    base = f"Great to see you, {name}"
    return (base.upper() + "!!!") if emphasis else base
