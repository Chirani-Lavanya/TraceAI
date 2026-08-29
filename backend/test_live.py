import requests

payload = "The ATM withdrawal amount must be between $500 and $100,000, and the account must have sufficient balance."

print("SENDING:", repr(payload))

response = requests.post(
    "http://127.0.0.1:9022/generate/",
    json={"text": payload},
    timeout=60
)

print("STATUS:", response.status_code)

data = response.json()

print("REQUIREMENT:", repr(data.get("requirement", {}).get("text")))

print("BVA:", [
    x.get("values")
    for x in data.get("test_cases", {}).get("bva", [])
])

print(
    "BVA_APPLICABLE:",
    data.get("evaluation", {}).get("details", {}).get("bva_applicable")
)

print(
    "BVA_QUALITY:",
    data.get("evaluation", {}).get("details", {}).get("bva_quality")
)
