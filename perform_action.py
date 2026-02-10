def perform_action(state: GlobalState):
    if state.intent == "procure_domain":
        response = {
            "status": "SUCCESS",
            "action": "PROCURE_DOMAIN",
            "domain": state.domain_name,
            "account_id": state.account_id,
            "order_id": "ORD-982374",
            "message": f"Domain {state.domain_name} has been successfully procured."
        }

    elif state.intent == "transfer_domain":
        response = {
            "status": "SUCCESS",
            "action": "TRANSFER_DOMAIN",
            "domain": state.domain_name,
            "account_id": state.account_id,
            "transfer_id": "TRF-773892",
            "message": f"Transfer initiated for {state.domain_name}."
        }

    else:
        response = {
            "status": "ERROR",
            "message": f"Unknown intent: {state.intent}"
        }

    # Append to chat history
    state.messages.append({
        "role": "assistant",
        "content": response["message"]
    })

    # Store structured output (important for APIs later)
    state.last_action_result = response

    return state