from graph import app

THREAD_ID = "test-thread-1"

def run_turn(messages):
    state = {"messages": messages}
    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }
    return app.invoke(state, config=config)

# ---- TURN 1: initial request
result = run_turn([
    {"role": "user", "content": "I want to procure code.com"}
])

print(result["messages"][-1]["content"])
# → "What is your account ID?"

# ---- TURN 2: provide account
result = run_turn([
    {"role": "user", "content": "My account id is 12345"}
])

print(result["messages"][-1]["content"])
# → "Procurement completed for code.com"