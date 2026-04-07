"""
WasteRoute-Env Baseline Inference Script
Runs an LLM agent against all 3 tasks and prints grader scores.

Usage:
    set HF_TOKEN=hf_...
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    python inference.py
"""

import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

# ── Credentials ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "wasteroute_env"
MAX_STEPS = 40
TEMPERATURE = 0.2
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.3

# ── Logging helpers ───────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── System prompt ─────────────────────────────────────────────
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
""".strip()


def build_prompt(obs) -> str:
    return f"""
Current state:
- Your location: Node {obs.current_node}
- Fuel remaining: {obs.fuel_remaining:.2f}
- Bins collected so far: {obs.collected_bins}
- Bin fill levels: {obs.bin_levels}
- Last message: {obs.message}

What is your next action? Respond with JSON only.
""".strip()


def parse_action(response_text: str) -> dict:
    try:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"action_type": "move", "target_node": 1}


def run_episode(env, client, task: str) -> tuple:
    """Run one full episode with proper logging."""
    from graders import grade

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task=task)
    obs_history = [obs]
    rewards = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(obs)},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = '{"action_type": "move", "target_node": 1}'

        action_dict = parse_action(response_text)
        action_str = json.dumps(action_dict)

        from models import WasteAction
        action = WasteAction(
            action_type=action_dict.get("action_type", "move"),
            target_node=int(action_dict.get("target_node", 1)),
        )

        obs = env.step(action)
        obs_history.append(obs)

        reward = obs.total_reward - (rewards[-1] if rewards else 0.0)
        rewards.append(round(reward, 2))
        steps_taken = step

        log_step(
            step=step,
            action=action_str,
            reward=reward,
            done=obs.done,
            error=None
        )

        if obs.done:
            break

    score = grade(task, obs_history, obs)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards
    )

    return obs_history, obs, score


def main():
    sys.path.insert(0, os.path.dirname(__file__))

    from server.wasteroute_env_environment import WasteRouteEnvironment
    from graders import grade

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY not set!")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = WasteRouteEnvironment()

    print("=" * 50)
    print("WasteRoute-Env Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)

    results = {}
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        _, _, score = run_episode(env, client, task)
        results[task] = score

    print("\n" + "=" * 50)
    print("FINAL BASELINE SCORES:")
    for task, score in results.items():
        print(f"  {task:8s}: {score:.3f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'average':8s}: {avg:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()