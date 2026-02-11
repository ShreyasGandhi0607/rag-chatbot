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