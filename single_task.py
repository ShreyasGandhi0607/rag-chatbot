from graph import app

THREAD_ID = "test-single-task-1"

def test_single_task_multiturn():
    state = {
        "messages": [
            {"role": "user", "content": "Transfer example.com"}
        ]
    }

    # Turn 1
    result = app.invoke(
        state,
        config={"configurable": {"thread_id": THREAD_ID}}
    )

    assert "authorization" in result["messages"][-1]["content"].lower() \
        or "account id" in result["messages"][-1]["content"].lower()

    # Turn 2
    result["messages"].append(
        {"role": "user", "content": "Auth code is AUTH123"}
    )

    result = app.invoke(
        result,
        config={"configurable": {"thread_id": THREAD_ID}}
    )

    assert "account id" in result["messages"][-1]["content"].lower()

    # Turn 3
    result["messages"].append(
        {"role": "user", "content": "My account ID is 99999"}
    )

    result = app.invoke(
        result,
        config={"configurable": {"thread_id": THREAD_ID}}
    )

    assert "completed" in result["messages"][-1]["content"].lower()

    print("✅ Single-task multi-turn test passed")


if __name__ == "__main__":
    test_single_task_multiturn()