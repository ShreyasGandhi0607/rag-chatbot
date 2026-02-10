REQUIRED_FIELDS = {
    "transfer": ["domain_name", "auth_code", "account_id"],
    "settings_change": ["domain_name", "settings_type", "account_id"],
    "procurement": ["domain_name", "account_id"],
    "billing_status": ["account_id"],
}


FOLLOWUPS = {
    "domain_name": "Please provide the domain name.",
    "auth_code": "Please provide the transfer authorization code.",
    "account_id": "What is your account ID?",
    "settings_type": "What setting would you like to change?",
}

INTENT_NORMALIZATION = {
    "transfer domain": "transfer",
    "domain transfer": "transfer",
    "transfer": "transfer",

    "procure domain": "procurement",
    "domain procurement": "procurement",
    "procurement": "procurement",

    "billing": "billing_status",
    "billing status": "billing_status",
}