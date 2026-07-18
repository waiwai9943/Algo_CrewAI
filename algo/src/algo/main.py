#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from algo.crew import Algo
from algo.tools.keepalive import start_keepalive, stop_keepalive

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# ─────────────────────────────────────────────────────────────
# Progress Callbacks — fires after each task and each agent step
# ─────────────────────────────────────────────────────────────

TASK_LABELS = {
    "strategy_design_task" : "[1/5] Strategy Design",
    "backtest_task"         : "[2/5] Historical Backtest",
    "overfitting_analysis_task": "[3/5] Overfitting Analysis",
    "code_architecture_task": "[4/5] Code Architecture",
    "risk_review_task"      : "[5/5] Risk Review",
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

    Launches the sequential pipeline:
        quant_researcher -> overfitting_analyst -> strategy_engineer -> risk_manager

    Reruns the crew if the strategy is overfitted or has a Sharpe Ratio <= 1.0.
    """
    from algo.crew import OverfittingAnalysis

    inputs = {
        'instrument': 'NQ',          # Target futures: 'NQ' (Nasdaq-100) or 'CL' (Crude Oil)
        'bar_timeframe': '1h',       # Bar resolution for strategy
        'session': 'RTH',            # Regular Trading Hours
        'account_size_usd': '50000', # Assumed account size for position sizing
        'current_year': str(datetime.now().year),
        'previous_attempts': '',     # Feedback loop variable
    }

    attempt = 1
    max_attempts = 10

    # Start keep-alive thread to prevent LM Studio model unloading
    start_keepalive()

    try:
        while attempt <= max_attempts:
            print(f"\n{'#'*80}")
            print(f"  STARTING CREW KICKOFF ATTEMPT #{attempt}")
            print(f"{'#'*80}\n")

            try:
                result = Algo().crew().kickoff(inputs=inputs)

                # Find the overfitting analysis output from tasks
                analysis = None
                for task_out in result.tasks_output:
                    if isinstance(task_out.pydantic, OverfittingAnalysis):
                        analysis = task_out.pydantic
                        break

                if analysis:
                    is_overfitted = analysis.is_overfitted
                    sharpe = analysis.sharpe_ratio
                    reasoning = analysis.reasoning
                    print(f"\nAttempt #{attempt} validation:")
                    print(f"  - Sharpe Ratio: {sharpe}")
                    print(f"  - Is Overfitted: {is_overfitted}")
                    print(f"  - Reasoning: {reasoning}\n")

                    if not is_overfitted and sharpe > 1.0:
                        print(f"Success! Found a robust trading strategy with Sharpe Ratio: {sharpe}")
                        break
                    else:
                        print(f"Strategy rejected. Sharpe Ratio <= 1.0 or Overfitted. Retrying in 10s...")
                        import time
                        time.sleep(10)
                        attempt_feedback = (
                            f"Attempt {attempt}: Sharpe Ratio {sharpe}. "
                            f"Is Overfitted: {is_overfitted}. "
                            f"Feedback/Reasoning: {reasoning}\n"
                        )
                        inputs['previous_attempts'] += attempt_feedback
                else:
                    print("Warning: OverfittingAnalysis output was not found or failed to parse. Rerunning in 10s...")
                    import time
                    time.sleep(10)

            except Exception as e:
                print(f"An error occurred during kickoff: {e}")
                print("Retrying in 20s to allow LM Studio to recover...")
                import time
                time.sleep(20)

            attempt += 1
        else:
            print(f"\nReached maximum attempts ({max_attempts}) without finding a strategy meeting the criteria.")
    finally:
        stop_keepalive()


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
