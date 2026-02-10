def decompose_tasks(state: GlobalState):
    # Do not re-run decomposition if tasks already exist
    if state.tasks:
        return state

    # Extract user message
    user_message = state.messages[-1]["content"]

    # Build conversation history (same logic you already use)
    conversation_history = []
    for msg in state.messages[:-1]:
        conversation_history.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Call your existing intent identification logic
    try:
        result = classify_intent(
            question=user_message,
            conversation_history=conversation_history
        )
    except Exception:
        # Hard fallback – never break the graph
        state.messages.append({
            "role": "assistant",
            "content": "Sorry, I couldn’t understand your request."
        })
        return state

    # Expected shape:
    # {
    #   "tasks": [
    #     {
    #       "intent": "...",
    #       "domain_name": "...",
    #       "supporting_sentence": "..."
    #     }
    #   ]
    # }

    raw_tasks = result.get("tasks", [])

    # Normalize into internal Task model
    state.tasks = [
        Task(
            topic=f"domain_{t['intent']}",
            intent=t["intent"],
            domain_name=t.get("domain_name")
        )
        for t in raw_tasks
    ]

    # User-facing acknowledgement
    state.messages.append({
        "role": "assistant",
        "content": f"I found {len(state.tasks)} request(s). I’ll handle them one by one."
    })

    return state