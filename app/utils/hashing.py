import hashlib


def calculate_content_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest() # return 64 characters string