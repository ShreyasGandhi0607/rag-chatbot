def perform_action(state: GlobalState):
    if not state.intent or state.task_completed:
        return state

    if state.intent == "procurement":
        result = domain_procurement_flow(...)
        state.messages.append({
            "role": "assistant",
            "content": result["message"]
        })

    elif state.intent == "transfer":
        state.messages.append({
            "role": "assistant",
            "content": "transfer_domain completed"
        })

    state.task_completed = True   # 🔒 CRITICAL
    return state