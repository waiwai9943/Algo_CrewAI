from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field

from algo.tools.market_data_tool import MarketDataTool
from algo.tools.indicator_tool import IndicatorTool
from algo.tools.backtest_tool import BacktestTool


class OverfittingAnalysis(BaseModel):
    is_overfitted: bool = Field(..., description="True if the strategy is overfitted or does not meet criteria (e.g. Sharpe <= 1.0), False otherwise.")
    sharpe_ratio: float = Field(..., description="The estimated annualized Sharpe ratio of the strategy.")
    reasoning: str = Field(..., description="Detailed explanation/reasoning of the overfitting and performance evaluation.")


def _progress_callback(task_output) -> None:
    """Inline progress printer — avoids circular import with main.py."""
    from datetime import datetime
    labels = {
        "strategy_design_task" : "[1/5] Strategy Design",
        "backtest_task"         : "[2/5] Historical Backtest",
        "overfitting_analysis_task": "[3/5] Overfitting Analysis",
        "code_architecture_task": "[4/5] Code Architecture",
        "risk_review_task"      : "[5/5] Risk Review",
    }
    ts = datetime.now().strftime("%H:%M:%S")
    name = getattr(task_output, 'name', '') or ''
    label = labels.get(name, name or 'task')
    raw = str(getattr(task_output, 'raw', task_output))
    preview = raw[:250].replace('\n', ' ')
    print(f"\n{'='*65}")
    print(f"  [OK] COMPLETED [{ts}]  {label}")
    print(f"  {preview}...")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# LM Studio — Local LLM Configuration
# Model: gemma-4-26b-a4b-qat
# ─────────────────────────────────────────────────────────────
lm_studio_llm = LLM(
    model="google/gemma-4-26b-a4b-qat",
    provider="openai",
    base_url="http://192.168.18.10:1234/v1",
    api_key="lm-studio",
    temperature=0.2,           # lower = more deterministic tool calls
    max_tokens=4096,           # Set to 4096 to avoid exceeding LM Studio default context limit
    timeout=300,               # 5 min timeout for complex reasoning tasks
)

# ─────────────────────────────────────────────────────────────
# Tool Instances
# ─────────────────────────────────────────────────────────────
market_data_tool = MarketDataTool()
indicator_tool   = IndicatorTool()
backtest_tool    = BacktestTool()


@CrewBase
class Algo():
    """
    Algo — Virtual Quantitative Hedge Fund Crew

    A sequential pipeline that designs, backtests with 10-year data (WFO),
    engineers, and stress-tests an intraday + swing futures trading strategy
    (Donchian Channel Breakout + EMA on 5-minute bars) for NQ (Nasdaq-100) futures.

    LLM Backend: LM Studio (http://192.168.18.10:1234) — google/gemma-4-26b-a4b-qat

    Pipeline (Sequential):
        1. quant_researcher  [strategy_design_task]  → Designs strategy using real data
        2. quant_researcher  [backtest_task]          → Validates with 10-year walk-forward optimization
        3. overfitting_analyst [overfitting_analysis_task] → Validates out-of-sample performance
        4. strategy_engineer [code_architecture_task] → Python/ib_insync code blueprint
        5. risk_manager      [risk_review_task]       → Risk review + Risk Control Addendum
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    # ─────────────────────────────────────────────
    # AGENTS
    # ─────────────────────────────────────────────

    @agent
    def quant_researcher(self) -> Agent:
        """
        Agent A — Quantitative Researcher
        Has access to: MarketDataTool, IndicatorTool, BacktestTool.
        Fetches real NQ 5-min data, computes VWAP+RSI, and runs backtests.
        """
        return Agent(
            config=self.agents_config['quant_researcher'],  # type: ignore[index]
            llm=lm_studio_llm,
            tools=[market_data_tool, indicator_tool, backtest_tool],
            verbose=True,
        )

    @agent
    def strategy_engineer(self) -> Agent:
        """
        Agent B — Strategy Engineer
        Receives validated strategy spec and backtest results.
        Produces Python/ib_insync code architecture (no data tools needed).
        """
        return Agent(
            config=self.agents_config['strategy_engineer'],  # type: ignore[index]
            llm=lm_studio_llm,
            verbose=True,
        )

    @agent
    def risk_manager(self) -> Agent:
        """
        Agent C — Chief Risk Officer
        Has access to BacktestTool for stress-testing with modified parameters.
        Produces the mandatory Risk Control Addendum.
        """
        return Agent(
            config=self.agents_config['risk_manager'],  # type: ignore[index]
            llm=lm_studio_llm,
            tools=[backtest_tool],
            verbose=True,
        )

    @agent
    def overfitting_analyst(self) -> Agent:
        """
        Agent D — Lead Overfitting Analyst & Validation Engineer
        Assesses if the proposed strategy has an actual edge or is overfitted to noise.
        """
        return Agent(
            config=self.agents_config['overfitting_analyst'],  # type: ignore[index]
            llm=lm_studio_llm,
            verbose=True,
        )

    # ─────────────────────────────────────────────
    # TASKS
    # ─────────────────────────────────────────────

    @task
    def strategy_design_task(self) -> Task:
        """
        Task 1: Researcher fetches real NQ data, computes indicators,
        and designs the VWAP+RSI strategy spec grounded in actual statistics.
        """
        return Task(
            config=self.tasks_config['strategy_design_task'],  # type: ignore[index]
            callback=_progress_callback,
        )

    @task
    def backtest_task(self) -> Task:
        """
        Task 2: Researcher runs the backtester with the designed parameters,
        performs sensitivity analysis, and produces a validated Backtest Report.
        """
        return Task(
            config=self.tasks_config['backtest_task'],  # type: ignore[index]
            callback=_progress_callback,
        )

    @task
    def overfitting_analysis_task(self) -> Task:
        """
        Task 3: Overfitting Analyst reviews the backtests and walk forward results,
        detecting parameter overfitting and data-snooping bias.
        """
        return Task(
            config=self.tasks_config['overfitting_analysis_task'],  # type: ignore[index]
            output_pydantic=OverfittingAnalysis,
            callback=_progress_callback,
        )

    @task
    def code_architecture_task(self) -> Task:
        """
        Task 3: Engineer translates the empirically validated strategy
        into a complete Python/ib_insync code blueprint.
        """
        return Task(
            config=self.tasks_config['code_architecture_task'],  # type: ignore[index]
            callback=_progress_callback,
        )

    @task
    def risk_review_task(self) -> Task:
        """
        Task 4: Risk Manager reviews all prior outputs adversarially,
        mandates risk controls, and produces the Risk Control Addendum.
        """
        return Task(
            config=self.tasks_config['risk_review_task'],  # type: ignore[index]
            output_file='risk_control_addendum.md',
            callback=_progress_callback,
        )

    # ─────────────────────────────────────────────
    # CREW
    # ─────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """
        Creates the Algo quant hedge fund crew.

        Process: Sequential — each task's output feeds into the next task's context.
            strategy_design → backtest → code_architecture → risk_review
        """
        return Crew(
            agents=self.agents,   # auto-populated by @agent decorators
            tasks=self.tasks,     # auto-populated by @task decorators
            process=Process.sequential,
            verbose=True,
        )
