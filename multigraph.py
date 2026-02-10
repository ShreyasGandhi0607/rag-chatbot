from graph import app

CONFIG = {
    "configurable": {
        "thread_id": "multi-task-test"
    }
}

def turn(user_input):
    return app.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=CONFIG
    )

# ---- TURN 1
r = turn("Procure code.com and transfer example.org")
print(r["messages"][-1]["content"])
# → "What is your account ID?"

# ---- TURN 2 (procurement)
r = turn("Account id is 123")
print(r["messages"][-1]["content"])
# → "Procure Domain completed for code.com"

# ---- TURN 3 (transfer)
r = turn("Auth code is XYZ-999")
print(r["messages"][-1]["content"])
# → "Transfer completed for example.org"