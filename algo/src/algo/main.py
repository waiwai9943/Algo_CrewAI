#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from algo.crew import Algo

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# ─────────────────────────────────────────────────────────────
# Progress Callbacks — fires after each task and each agent step
# ─────────────────────────────────────────────────────────────

TASK_LABELS = {
    "strategy_design_task" : "[1/4] Strategy Design",
    "backtest_task"         : "[2/4] Historical Backtest",
    "code_architecture_task": "[3/4] Code Architecture",
    "risk_review_task"      : "[4/4] Risk Review",
}


def task_callback(task_output) -> None:
    """Called by CrewAI after each task completes."""
    ts = datetime.now().strftime("%H:%M:%S")
    name = getattr(task_output, 'name', '') or ''
    label = TASK_LABELS.get(name, name)
    raw = str(getattr(task_output, 'raw', task_output))
    preview = raw[:300].replace('\n', ' ')
    print(f"\n{'='*65}")
    print(f"  TASK DONE [{ts}]  {label}")
    print(f"  Preview: {preview}...")
    print(f"{'='*65}\n")


def step_callback(agent_output) -> None:
    """Called after every agent reasoning step (thought/action/observation)."""
    ts = datetime.now().strftime("%H:%M:%S")
    # Only print tool-use lines to avoid flooding the terminal
    text = str(agent_output)
    if 'Action:' in text or 'Tool:' in text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines:
            if any(kw in line for kw in ('Action:', 'Action Input:', 'Tool:')):
                print(f"  [{ts}] {line}")


def run():
    """
    Run the Algo quant hedge fund crew.

    Launches the sequential 3-agent pipeline:
        quant_researcher -> strategy_engineer -> risk_manager

    The crew will design a VWAP+RSI intraday futures strategy,
    produce a Python/ib_insync code architecture, and generate
    a mandatory Risk Control Addendum saved to risk_control_addendum.md.
    """
    inputs = {
        'instrument': 'NQ',          # Target futures: 'NQ' (Nasdaq-100) or 'CL' (Crude Oil)
        'bar_timeframe': '5min',     # Bar resolution for strategy
        'session': 'RTH',            # Regular Trading Hours
        'account_size_usd': '50000', # Assumed account size for position sizing
        'current_year': str(datetime.now().year),
    }

    try:
        Algo().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'instrument': 'NQ',
        'bar_timeframe': '5min',
        'session': 'RTH',
        'account_size_usd': '50000',
        'current_year': str(datetime.now().year),
    }
    try:
        Algo().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Algo().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'instrument': 'NQ',
        'bar_timeframe': '5min',
        'session': 'RTH',
        'account_size_usd': '50000',
        'current_year': str(datetime.now().year),
    }

    try:
        Algo().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = Algo().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
