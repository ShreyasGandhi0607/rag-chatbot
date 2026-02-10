def perform_action(state: GlobalState):
    # Prevent duplicate execution (LangGraph re-runs nodes)
    if state.task_completed or not state.intent:
        return state

    if state.intent == "procurement":
        result = handle_procurement(state)

    elif state.intent == "transfer":
        result = handle_transfer(state)

    else:
        result = {
            "status": "ERROR",
            "action": "UNKNOWN",
            "message": f"Unsupported intent: {state.intent}"
        }

    # Save structured result
    state.last_action_result = result

    # Send user-facing message
    state.messages.append({
        "role": "assistant",
        "content": result["message"]
    })

    state.task_completed = True
    return state