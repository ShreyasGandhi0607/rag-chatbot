from state import GlobalState


def route_after_validation(state: GlobalState):
    if state.missing_fields:
        return "ask_followup"
    return "perform_action"


def route_after_action(state: GlobalState):
    if state.current_task_index + 1 < len(state.tasks):
        return "next_task"
    return "__end__"
