from graph import app

def test_single_task_multiturn():
    # TURN 1: user starts request
    state = {
        "messages": [
            {"role": "user", "content": "Transfer example.com"}
        ]
    }

    result = app.invoke(state)

    # Agent should ask for a missing field
    last_message = result["messages"][-1]["content"]
    assert "authorization code" in last_message.lower() or "account id" in last_message.lower()

    # TURN 2: user provides auth code
    result["messages"].append(
        {"role": "user", "content": "The auth code is AUTH123"}
    )

    result = app.invoke(result)

    last_message = result["messages"][-1]["content"]
    assert "account id" in last_message.lower()

    # TURN 3: user provides account ID
    result["messages"].append(
        {"role": "user", "content": "My account ID is 99999"}
    )

    result = app.invoke(result)

    # Final response
    last_message = result["messages"][-1]["content"]
    assert "completed" in last_message.lower()

    print("✅ Single-task multi-turn test passed")


if __name__ == "__main__":
    test_single_task_multiturn()