from graph import app

def run_conversation():
    config = {
        "configurable": {
            "thread_id": "test-multi-task"
        }
    }

    state = {
        "messages": [
            {
                "role": "user",
                "content": "Procure code.com and transfer example.org"
            }
        ]
    }

    # Turn 1
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

    # Turn 2 – account id
    state["messages"].append({
        "role": "user",
        "content": "Account id is 123"
    })
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

    # Turn 3 – auth code
    state["messages"].append({
        "role": "user",
        "content": "Auth code is XYZ-999"
    })
    state = app.invoke(state, config=config)
    print(state["messages"][-1]["content"])

if __name__ == "__main__":
    run_conversation()