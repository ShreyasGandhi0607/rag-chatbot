from graph import app

THREAD_ID = "test-single-task-1"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

def test_single_task_multiturn():
    # TURN 1
    state = {
        "messages": [
            {"role": "user", "content": "Transfer example.com"}
        ]
    }

    result = app.invoke(state, config=CONFIG)

    assert (
        "authorization" in result["messages"][-1]["content"].lower()
        or "account id" in result["messages"][-1]["content"].lower()
    )

    # TURN 2
    result["messages"].append(
        {"role": "user", "content": "Auth code is AUTH123"}
    )

    result = app.invoke(result, config=CONFIG)

    assert "account id" in result["messages"][-1]["content"].lower()

    # TURN 3
    result["messages"].append(
        {"role": "user", "content": "My account ID is 99999"}
    )

    result = app.invoke(result, config=CONFIG)

    assert "completed" in result["messages"][-1]["content"].lower()

    print("✅ Single-task multi-turn test passed")


if __name__ == "__main__":
    test_single_task_multiturn()