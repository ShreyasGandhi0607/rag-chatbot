def perform_action(state: GlobalState):
    # Safety guard (important for retries)
    if not state.intent or state.task_completed:
        return state

    result = None

    # -------------------------------
    # PROCURE DOMAIN
    # -------------------------------
    if state.intent == "procure_domain":
        result = {
            "status": "SUCCESS",
            "action": "PROCURE_DOMAIN",
            "domain": state.domain_name,
            "account_id": state.account_id,
            "order_id": "ORD-982374",
            "message": f"Procure Domain completed for {state.domain_name}."
        }

    # -------------------------------
    # TRANSFER DOMAIN
    # -------------------------------
    elif state.intent == "transfer_domain":
        result = {
            "status": "SUCCESS",
            "action": "TRANSFER_DOMAIN",
            "domain": state.domain_name,
            "account_id": state.account_id,
            "transfer_id": "TRF-773892",
            "message": f"Transfer initiated for {state.domain_name}."
        }

    # -------------------------------
    # UNKNOWN INTENT (defensive)
    # -------------------------------
    else:
        result = {
            "status": "ERROR",
            "message": f"Unsupported intent: {state.intent}"
        }

    # -------------------------------
    # Persist results
    # -------------------------------
    state.last_action_result = result

    state.messages.append({
        "role": "assistant",
        "content": result["message"]
    })

    state.task_completed = True  # 🔐 REQUIRED for next_task routing
    return state