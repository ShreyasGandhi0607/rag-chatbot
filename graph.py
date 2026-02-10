from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from state import GlobalState
from nodes import (
    decompose_tasks,
    load_task,
    extract_fields,
    recalc_missing,
    ask_followup,
    perform_action,
    next_task,
)
from routes import route_after_validation, route_after_action


checkpointer = SqliteSaver("domain_agent_memory.db")

graph = StateGraph(GlobalState)

graph.add_node("decompose", decompose_tasks)
graph.add_node("load_task", load_task)
graph.add_node("extract", extract_fields)
graph.add_node("recalc", recalc_missing)
graph.add_node("ask_followup", ask_followup)
graph.add_node("perform_action", perform_action)
graph.add_node("next_task", next_task)

graph.set_entry_point("decompose")

graph.add_edge("decompose", "load_task")
graph.add_edge("load_task", "extract")
graph.add_edge("extract", "recalc")

graph.add_conditional_edges(
    "recalc",
    route_after_validation,
    {
        "ask_followup": "ask_followup",
        "perform_action": "perform_action",
    }
)

graph.add_conditional_edges(
    "perform_action",
    route_after_action,
    {
        "next_task": "next_task",
        "__end__": "__end__",
    }
)

graph.add_edge("next_task", "load_task")

app = graph.compile(checkpointer=checkpointer)