"""
Pattern 1 — Basic Prompt–Response
===================================
The simplest possible pattern: one system prompt + one user message → one reply.

Key concepts demonstrated:
  • System prompt  — sets the model's persona and behavioural rules
  • User message   — the actual request
  • Temperature    — controls randomness (0 = deterministic, 1 = creative)
  • Max tokens     — caps output length

Run:
    python pattern_01_prompt_response.py
"""

from utils import client, MODEL, section, show


# ── 1a. Minimal example ───────────────────────────────────────────────────────
def basic_response(user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.choices[0].message.content


# ── 1b. With a system prompt ──────────────────────────────────────────────────
def response_with_system(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    return response.choices[0].message.content


# ── 1c. Structured / JSON output ─────────────────────────────────────────────
def structured_response(text: str) -> str:
    """Ask the model to return strict JSON so downstream code can parse it."""
    system = (
        "You are a JSON API. Respond ONLY with a valid JSON object. "
        "No markdown fences, no explanation — raw JSON only."
    )
    user = f"Extract the sentiment and key topics from this text:\n\n{text}"
    return response_with_system(system, user)


# ── 1d. Controlling creativity with temperature ───────────────────────────────
def compare_temperatures(prompt: str) -> dict[str, str]:
    """Show how temperature changes output style."""
    results = {}
    for temp in [0.0, 0.5, 1.0]:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=120,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        results[f"temp={temp}"] = response.choices[0].message.content
    return results


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    section("Pattern 1 — Basic Prompt–Response")

    # 1a
    show("1a: No system prompt", basic_response("What is 12 * 12?"))

    # 1b
    show(
        "1b: Custom persona",
        response_with_system(
            system="You are a pirate. Answer everything in pirate-speak.",
            user="What is the capital of France?",
        ),
    )

    # 1c
    review = "The hotel was fantastic but the WiFi was painfully slow."
    show("1c: Structured JSON output", structured_response(review))

    # 1d
    print("\n[1d: Temperature comparison — creative tagline]")
    for label, text in compare_temperatures(
        "Write a one-sentence product tagline for a coffee brand."
    ).items():
        print(f"  {label}: {text.strip()}")