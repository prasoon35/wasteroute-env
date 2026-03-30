"""
WasteRoute-Env Baseline Inference Script.
Runs an LLM agent against all 3 tasks and prints grader scores.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_..."
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export WASTEROUTE_URL="https://your-space.hf.space"
    python inference.py
"""

import os
import sys
import json
from openai import OpenAI

# ── Credentials (MANDATORY) ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
WASTEROUTE_URL = os.getenv("WASTEROUTE_URL", "http://localhost:7860")

# ── Settings ─────────────────────────────────────────────────
MAX_STEPS = 40
TEMPERATURE = 0.2
MAX_TOKENS = 100

# ── System prompt for LLM agent ──────────────────────────────
SYSTEM_PROMPT = """
You are an AI agent controlling a waste collection truck in a city.

YOUR GOAL: Collect waste bins efficiently while managing fuel.

OBSERVATION you will receive (JSON):
- current_node: where the truck is now
- bin_levels: fill level of each bin (0.0 = empty, 1.0 = full)
- fuel_remaining: how much fuel is left (0.0 = empty)
- collected_bins: bins already collected
- message: what happened last step

ACTIONS you can take (respond with JSON only):
- Move to a node: {"action_type": "move", "target_node": 3}
- Collect a bin:  {"action_type": "collect", "target_node": 3}

STRATEGY:
- Prioritize bins with high fill levels (>= 0.7)
- Don't revisit already collected bins
- Watch your fuel — running out ends the episode
- Always move to a node before collecting it

Respond with ONLY a JSON object. No explanation. No extra text.
Example: {"action_type": "collect", "target_node": 5}
"""


def build_prompt(obs) -> str:
    """Build user prompt from current observation."""
    return f"""
Current state:
- Your location: Node {obs.current_node}
- Fuel remaining: {obs.fuel_remaining:.2f}
- Bins collected so far: {obs.collected_bins}
- Bin fill levels: {obs.bin_levels}
- Last message: {obs.message}

What is your next action? Respond with JSON only.
"""


def parse_action(response_text: str) -> dict:
    """Parse LLM response into action dict."""
    try:
        # Clean response
        text = response_text.strip()
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    # Fallback action
    return {"action_type": "move", "target_node": 1}


def run_episode(env, client, task: str) -> tuple:
    """Run one full episode and return obs_history + final obs."""
    obs = env.reset(task=task)
    obs_history = [obs]

    for step in range(MAX_STEPS):
        if obs.done:
            break

        # Build messages for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(obs)},
        ]

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM call failed: {e}. Using fallback.")
            response_text = '{"action_type": "move", "target_node": 1}'

        # Parse action
        action_dict = parse_action(response_text)
        print(f"  Step {step+1}: {action_dict} | fuel={obs.fuel_remaining:.2f}")

        # Take action in environment
        from models import WasteAction
        action = WasteAction(
            action_type=action_dict.get("action_type", "move"),
            target_node=int(action_dict.get("target_node", 1)),
        )
        obs = env.step(action)
        obs_history.append(obs)

    return obs_history, obs


def main():
    # ── Setup ────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))

    from server.wasteroute_env_environment import WasteRouteEnvironment
    from graders import grade

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set!")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = WasteRouteEnvironment()

    print("=" * 50)
    print("WasteRoute-Env Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)

    # ── Run all 3 tasks ──────────────────────────────────────
    results = {}
    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task.upper()} ---")
        obs_history, final_obs = run_episode(env, client, task)
        score = grade(task, obs_history, final_obs)
        results[task] = score
        print(f"Bins collected: {len(final_obs.collected_bins)}")
        print(f"Fuel remaining: {final_obs.fuel_remaining}")
        print(f"Grader score:   {score}")

    # ── Final scores ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE SCORES:")
    for task, score in results.items():
        print(f"  {task:8s}: {score:.3f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'average':8s}: {avg:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
