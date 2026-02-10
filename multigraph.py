from graph import app

def run():
    config = {
        "configurable": {
            "thread_id": "multi-task-test"
        }
    }

    # Turn 1
    state = {
        "messages": [
            {"role": "user", "content": "Procure code.com and transfer example.org"}
        ]
    }
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

    # Turn 2
    state["messages"].append(
        {"role": "user", "content": "Account id is 123"}
    )
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

    # Turn 3
    state["messages"].append(
        {"role": "user", "content": "Auth code is XYZ-999"}
    )
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

if __name__ == "__main__":
    run()