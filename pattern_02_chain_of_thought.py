"""
Pattern 2 — Chain of Thought (CoT)
=====================================
Make the model reason step-by-step before answering.
This dramatically improves accuracy on multi-step reasoning tasks.

Two variants:
  • Zero-shot CoT  — append "Think step by step." to any prompt
  • Few-shot CoT   — provide worked examples in the prompt
  • Extended Thinking — use Anthropic's native extended thinking API
                        (claude-sonnet-4-6 / claude-opus-4-6 only)

Run:
    python pattern_02_chain_of_thought.py
"""

from utils import client, MODEL, section, show

# ── 2a. Zero-shot CoT ─────────────────────────────────────────────────────────
def zero_shot_cot(question: str) -> tuple[str, str]:
    """
    Returns (direct_answer, cot_answer) so you can compare.
    """
    # Direct answer — no reasoning cue
    direct = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    ).choices[0].message.content

    # CoT — magic phrase forces step-by-step reasoning
    cot_prompt = f"{question}\n\nLet's think step by step before answering."
    cot = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": cot_prompt}],
    ).choices[0].message.content

    return direct, cot


# ── 2b. Few-shot CoT ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
Solve these word problems step by step.

Q: A bakery makes 240 cookies per day. They sell 75% on weekdays and 90%
   on weekends. How many cookies are left unsold over a 5-day week (Mon–Fri)?
A: Step 1 — Cookies made in 5 weekdays = 240 × 5 = 1,200.
   Step 2 — Sold each weekday = 240 × 0.75 = 180.
   Step 3 — Sold over 5 days = 180 × 5 = 900.
   Step 4 — Unsold = 1,200 − 900 = 300.
   Answer: 300 cookies left unsold.

Q: A train leaves Station A at 8:00 AM travelling at 60 mph. Another train
   leaves Station B (300 miles away) at 9:00 AM travelling toward Station A
   at 90 mph. At what time do they meet?
A: Step 1 — By 9:00 AM the first train has covered 60 × 1 = 60 miles.
   Step 2 — Remaining gap = 300 − 60 = 240 miles.
   Step 3 — Combined speed = 60 + 90 = 150 mph.
   Step 4 — Time to close 240 miles = 240 / 150 = 1.6 hours = 1 h 36 min.
   Step 5 — Meeting time = 9:00 AM + 1 h 36 min = 10:36 AM.
   Answer: 10:36 AM.
"""

def few_shot_cot(question: str) -> str:
    prompt = f"{FEW_SHOT_EXAMPLES}\nQ: {question}\nA:"
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content


# ── 2c. Structured CoT with XML tags ──────────────────────────────────────────
def structured_cot(question: str) -> dict[str, str]:
    """
    Ask the model to put reasoning and answer in separate XML tags.
    Makes it easy to extract just the final answer programmatically.
    """
    system = (
        "You are an expert problem solver. "
        "Always think through problems carefully. "
        "Put your reasoning inside <think> tags "
        "and your final answer inside <answer> tags."
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
    )

    full_content = response.choices[0].message.content
    
    thinking_text = "No thinking block found."
    answer_text = full_content # Default to full content if parsing fails

    # Parsing the tags/structured response
    if "<think>" in full_content and "</think>" in full_content:
        # Extract everything between <think> and </think>
        thinking_text = full_content.split("<think>")[1].split("</think>")[0].strip()
        
        # Extract everything inside <answer> if it exists, else everything after </think>
        if "<answer>" in full_content:
            answer_text = full_content.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            answer_text = full_content.split("</think>")[1].strip()

    return {"thinking": thinking_text, "answer": answer_text}


# ── 2d. Extended Thinking (Anthropic-native) ──────────────────────────────────
def extended_thinking_cot(question: str, budget_tokens: int = 5000) -> dict[str, str]:
    """
    Uses Anthropic's native extended thinking feature.
    The model allocates an internal thinking budget before responding.

    NOTE: Requires claude-sonnet-4-6 or claude-opus-4-6.
          Temperature must be 1 (fixed for extended thinking).
    """
    import anthropic
    thinking_model = "claude-sonnet-4-6"
    response = client.messages.create(
        model=thinking_model,
        max_tokens=budget_tokens + 1024,
        temperature=1,          # required for extended thinking
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens,
        },
        messages=[{"role": "user", "content": question}],
    )

    thinking_text = ""
    answer_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            answer_text = block.text

    return {"thinking": thinking_text, "answer": answer_text}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    section("Pattern 2 — Chain of Thought")

    # 2a
    question = (
        "A store buys 50 items at $8 each and sells them at a 40% markup. "
        "If 10 items remain unsold, what is the total profit?"
    )
    direct, cot = zero_shot_cot(question)
    show("2a: Direct (no CoT)", direct)
    show("2a: Zero-shot CoT", cot)

    # 2b
    hard_q = (
        "Alice has twice as many apples as Bob. Charlie has 5 fewer apples "
        "than Alice. Together they have 55 apples. How many does Bob have?"
    )
    show("2b: Few-shot CoT", few_shot_cot(hard_q))

    # 2c
    logic_q = (
        "All Bloops are Razzies. All Razzies are Lazzies. "
        "Are all Bloops definitely Lazzies?"
    )
    result = structured_cot(logic_q)
    show("2c: Structured CoT — Thinking", result["thinking"])
    show("2c: Structured CoT — Answer", result["answer"])

    # 2d — uncomment if you have access to claude-sonnet-4-6
    # result = extended_thinking_cot(
    #     "How many prime numbers are there between 1 and 100?"
    # )
    # show("2d: Extended Thinking", result["thinking"][:500] + "...")
    # show("2d: Extended Thinking — Answer", result["answer"])