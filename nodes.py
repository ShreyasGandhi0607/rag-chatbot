from state import GlobalState, Task
from prompts import DECOMPOSE_PROMPT, SLOT_PROMPT
from requirements import REQUIRED_FIELDS, FOLLOWUPS
from llm import llm


def decompose_tasks(state: GlobalState):
    if state.tasks:
        return state

    result = llm.invoke(
        DECOMPOSE_PROMPT.format(
            user_message=state.messages[-1]["content"]
        )
    )

    state.tasks = [Task(**t) for t in result["tasks"]]

    state.messages.append({
        "role": "assistant",
        "content": f"I found {len(state.tasks)} request(s). I’ll handle them one by one."
    })
    return state


def load_task(state: GlobalState):
    task = state.tasks[state.current_task_index]
    state.topic = task.topic
    state.intent = task.intent
    state.domain_name = task.domain_name
    return state


def extract_fields(state: GlobalState):
    result = llm.invoke(
        SLOT_PROMPT.format(
            intent=state.intent,
            domain_name=state.domain_name,
            account_id=state.account_id,
            auth_code=state.auth_code,
            settings_type=state.settings_type,
            user_message=state.messages[-1]["content"]
        )
    )

    for k, v in result.items():
        if hasattr(state, k) and v and getattr(state, k) is None:
            setattr(state, k, v)

    return state


def recalc_missing(state: GlobalState):
    required = REQUIRED_FIELDS[state.intent]
    state.missing_fields = [
        f for f in required if getattr(state, f) is None
    ]
    return state


def ask_followup(state: GlobalState):
    field = state.missing_fields[0]
    state.messages.append({
        "role": "assistant",
        "content": FOLLOWUPS[field]
    })
    return state


def perform_action(state: GlobalState):
    state.messages.append({
        "role": "assistant",
        "content": f"{state.intent} completed for {state.domain_name or 'your account'}"
    })
    return state


def next_task(state: GlobalState):
    state.current_task_index += 1

    state.intent = None
    state.domain_name = None
    state.account_id = None
    state.auth_code = None
    state.settings_type = None
    state.missing_fields = []

    return state