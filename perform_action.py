def perform_action(state: GlobalState):
    if not state.intent:
        return state

    if state.intent == "procurement":
        result = domain_procurement_flow(
            domain=state.domain_name,
            dis_response=state.dis_response,
            business_reason=state.business_reason,
            sub_account=state.sub_account
        )
        state.messages.append({
            "role": "assistant",
            "content": result["message"]
        })

    elif state.intent == "transfer":
        state.messages.append({
            "role": "assistant",
            "content": f"Transfer completed for {state.domain_name}"
        })

    else:
        state.messages.append({
            "role": "assistant",
            "content": f"{state.intent} completed"
        })

    return state