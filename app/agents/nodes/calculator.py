import logfire

from app.agents.state import AgentState
from app.tools.calculator import calculate


def calculator_node(state: AgentState) -> dict:
    query = state.get("current_query", "")

    try:
        result = calculate(query)
        answer = f"Result: {result}"

        logfire.info("Calculator executed successfully")

        return {
            "tool_result": answer,
            "final_answer": answer,
            "status": "Completed",
            "plan": state.get("plan", []) + ["Calculator executed"],
        }

    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        answer = f"Unable to calculate expression: {exc}"

        return {
            "tool_result": answer,
            "final_answer": answer,
            "status": "Tool input error",
            "plan": state.get("plan", []) + ["Calculator input rejected"],
        }
