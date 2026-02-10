DECOMPOSE_PROMPT = """
You are a task decomposition engine.

SUPPORTED TOPICS:
- domain_transfer
- domain_settings_change
- domain_procurement
- billing

RULES:
1. One task = one intent + one domain
2. If multiple domains are mentioned, create separate tasks
3. Preserve user order

User message:
"{user_message}"

Return JSON ONLY:
{
  "tasks": [
    {
      "topic": "domain_transfer",
      "intent": "transfer",
      "domain_name": "example.com"
    }
  ]
}
"""


SLOT_PROMPT = """
Extract missing fields for the current task.

Intent: {intent}
Domain: {domain_name}

Known values:
- account_id: {account_id}
- auth_code: {auth_code}
- settings_type: {settings_type}

User message:
"{user_message}"

Return JSON only. Return {{}} if nothing found.
"""