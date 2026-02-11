Perfect 👍 let’s build this properly and production-ready.

You asked for:
	•	✅ Multi-TLD RDAP router (.com, .org, .net auto detect)
	•	✅ Async scalable version
	•	✅ Clean availability + expiry extraction
	•	✅ Tool integration for your domain agent

We’ll build this step-by-step.

⸻

🧠 Architecture Overview

User → LLM Agent
        ↓
   rdap_lookup_tool(domain)
        ↓
   RDAP Registry Endpoint
        ↓
   Parsed JSON (available, expiry, registrar, etc.)

No third-party APIs.
Only official registry RDAP endpoints.

⸻

🌍 Official RDAP Endpoints
	•	.com / .net → operated by Verisign
https://rdap.verisign.com/com/v1/domain/
https://rdap.verisign.com/net/v1/domain/
	•	.org → operated by Public Interest Registry
https://rdap.publicinterestregistry.org/rdap/org/domain/

Under authority of ICANN

⸻

🚀 1️⃣ Multi-TLD RDAP Router (Async + Scalable)

We’ll use httpx for async HTTP.

Install:

pip install httpx


⸻

✅ rdap_service.py

import httpx
from datetime import datetime
from typing import Dict, Optional


RDAP_SERVERS = {
    "com": "https://rdap.verisign.com/com/v1/domain/",
    "net": "https://rdap.verisign.com/net/v1/domain/",
    "org": "https://rdap.publicinterestregistry.org/rdap/org/domain/",
}


class RDAPService:

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def _get_tld(self, domain: str) -> str:
        return domain.split(".")[-1].lower()

    def _get_rdap_url(self, domain: str) -> Optional[str]:
        tld = self._get_tld(domain)
        base = RDAP_SERVERS.get(tld)
        if not base:
            return None
        return base + domain

    async def lookup(self, domain: str) -> Dict:
        url = self._get_rdap_url(domain)

        if not url:
            return {
                "domain": domain,
                "supported": False,
                "error": "TLD not supported"
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)

        if response.status_code == 404:
            return {
                "domain": domain,
                "available": True,
                "registered": False
            }

        if response.status_code != 200:
            return {
                "domain": domain,
                "error": f"RDAP error {response.status_code}"
            }

        data = response.json()

        return self._parse_rdap_response(domain, data)

    def _parse_rdap_response(self, domain: str, data: Dict) -> Dict:
        expiry = None
        registrar = None

        # Extract expiry
        for event in data.get("events", []):
            if event.get("eventAction") == "expiration":
                expiry = event.get("eventDate")

        # Extract registrar
        for entity in data.get("entities", []):
            roles = entity.get("roles", [])
            if "registrar" in roles:
                registrar = entity.get("vcardArray", [])

        return {
            "domain": domain,
            "available": False,
            "registered": True,
            "expiry_date": expiry,
            "registrar_raw": registrar
        }


⸻

⚡ 2️⃣ Batch Async Scalable Lookup

Handles 1 domain or 1000 concurrently.

import asyncio
from rdap_service import RDAPService


async def check_domains(domains):
    service = RDAPService()

    tasks = [service.lookup(domain) for domain in domains]
    results = await asyncio.gather(*tasks)

    return results


if __name__ == "__main__":
    domains = [
        "code.com",
        "example.org",
        "somerandomnonexist123456.com"
    ]

    results = asyncio.run(check_domains(domains))

    for r in results:
        print(r)

This scales horizontally and supports concurrency cleanly.

⸻

🧹 3️⃣ Clean Output Version (Production Ready Parsing)

Replace _parse_rdap_response with this cleaner version:

def _parse_rdap_response(self, domain: str, data: Dict) -> Dict:
    expiry = None
    creation = None
    registrar_name = None

    for event in data.get("events", []):
        if event.get("eventAction") == "expiration":
            expiry = event.get("eventDate")
        if event.get("eventAction") == "registration":
            creation = event.get("eventDate")

    for entity in data.get("entities", []):
        if "registrar" in entity.get("roles", []):
            vcard = entity.get("vcardArray", [])
            if len(vcard) > 1:
                for field in vcard[1]:
                    if field[0] == "fn":
                        registrar_name = field[3]

    return {
        "domain": domain,
        "available": False,
        "registered": True,
        "creation_date": creation,
        "expiry_date": expiry,
        "registrar": registrar_name
    }

Now your output is clean and usable.

⸻

🤖 4️⃣ Integrate as Tool Inside Domain Agent

If you’re using a tool-based LLM agent:

async def rdap_lookup_tool(domain: str) -> dict:
    service = RDAPService()
    return await service.lookup(domain)

Then register it as a tool in your agent framework.

LLM flow:

User: "Is code.com available?"
LLM → calls rdap_lookup_tool("code.com")
Tool → returns structured JSON
LLM → formats final response

LLM never guesses availability.

⸻

🏗 Production Improvements (Optional but Smart)

If you scale:
	•	Reuse AsyncClient globally
	•	Add rate limiting
	•	Add caching (Redis)
	•	Add retry with exponential backoff
	•	Support more TLDs dynamically

⸻

🔥 What You Now Have

✅ No third-party API
✅ Registry-backed official data
✅ Async scalable
✅ Multi-TLD router
✅ Clean expiry + registrar extraction
✅ Agent-ready tool function

⸻

If you want next level:
	•	🌎 Add automatic RDAP discovery via IANA bootstrap
	•	⚡ Convert into FastAPI microservice
	•	🧠 Add intelligent domain suggestion engine
	•	📊 Add bulk availability checker with concurrency limits

Tell me your deployment target (Cloud Function? FastAPI? Docker?) and I’ll tailor it.
