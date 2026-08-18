import hashlib


def calculate_content_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest() # return 64 characters string



def calculate_stream_hash(
    stream: BinaryIO,
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[str, int]:
    hasher = hashlib.sha256()
    file_size = 0

    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
        file_size += len(chunk)

    stream.seek(0) 
    """ stream.seek(0) -> reset sthe stream position after reading, otherwise Azure upload may receive the stream at the end.
    [ Start -------------------- End ]
        ↑
        stream position reset
    """
    return hasher.hexdigest(), file_size