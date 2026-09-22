import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{API_URL}/api/v1/documents"

DOCUMENT_NAME = "Business Loan Underwriting Policy"
TENANT_ID = "underwriting-test"

CATEGORIES = [
    "Business Loan",
    "Underwriting",
]

TAGS = [
    "business-loan",
    "underwriting-test",
    "multimodal",
]


def build_multipart_body(
    pdf_path: Path,
) -> tuple[bytes, str]:
    """
    Build a multipart/form-data request body manually.

    This intentionally avoids httpx multipart handling so the
    integration test exercises only the real FastAPI endpoint.
    """

    boundary = (
        "----ContextOpsBoundary"
        + uuid.uuid4().hex
    )

    body = bytearray()

    def add_text_field(
        name: str,
        value: str,
    ) -> None:
        body.extend(
            f"--{boundary}\r\n".encode("utf-8")
        )
        body.extend(
            (
                f'Content-Disposition: form-data; '
                f'name="{name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(
            b"Content-Type: text/plain; "
            b"charset=utf-8\r\n"
        )
        body.extend(b"\r\n")
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    def add_file_field(
        name: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> None:
        body.extend(
            f"--{boundary}\r\n".encode("utf-8")
        )

        body.extend(
            (
                f'Content-Disposition: form-data; '
                f'name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )

        body.extend(
            (
                f"Content-Type: {content_type}\r\n"
            ).encode("utf-8")
        )

        body.extend(b"\r\n")
        body.extend(content)
        body.extend(b"\r\n")

    # ---------------------------------------------------------
    # FORM FIELDS
    # ---------------------------------------------------------

    add_text_field(
        "document_name",
        DOCUMENT_NAME,
    )

    for category in CATEGORIES:
        add_text_field(
            "categories",
            category,
        )

    for tag in TAGS:
        add_text_field(
            "tags",
            tag,
        )

    # ---------------------------------------------------------
    # PDF FILE
    # ---------------------------------------------------------

    pdf_bytes = pdf_path.read_bytes()

    add_file_field(
        name="file",
        filename=pdf_path.name,
        content=pdf_bytes,
        content_type="application/pdf",
    )

    # ---------------------------------------------------------
    # END MULTIPART BODY
    # ---------------------------------------------------------

    body.extend(
        f"--{boundary}--\r\n".encode("utf-8")
    )

    content_type = (
        f"multipart/form-data; boundary={boundary}"
    )

    return bytes(body), content_type


def upload_document(
    pdf_path: Path,
) -> None:

    # ---------------------------------------------------------
    # VALIDATE PDF
    # ---------------------------------------------------------

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"Path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Integration test requires a PDF file."
        )

    pdf_size = pdf_path.stat().st_size

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print("=" * 70)
    print(
        "ContextOps — REAL PDF INGESTION "
        "INTEGRATION TEST"
    )
    print("=" * 70)

    print()
    print("PDF:")
    print(f"  {pdf_path}")

    print()
    print("PDF SIZE:")
    print(f"  {pdf_size:,} bytes")

    print()
    print("API:")
    print(f"  POST {UPLOAD_URL}")

    print()
    print("Document:")
    print(f"  {DOCUMENT_NAME}")

    print()
    print("Tenant:")
    print(f"  {TENANT_ID}")

    print()
    print("Categories:")

    for category in CATEGORIES:
        print(f"  - {category}")

    print()
    print("Tags:")

    for tag in TAGS:
        print(f"  - {tag}")

    # ---------------------------------------------------------
    # BUILD MULTIPART REQUEST
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Building multipart request...")
    print("=" * 70)

    body, content_type = build_multipart_body(
        pdf_path
    )

    print()
    print("Multipart body size:")
    print(f"  {len(body):,} bytes")

    print()
    print("Content-Type:")
    print(f"  {content_type}")

    # ---------------------------------------------------------
    # SEND REQUEST
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Uploading PDF...")
    print("=" * 70)

    request = Request(
        UPLOAD_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=120,
        ) as response:

            status_code = response.status
            response_body = response.read()

    except HTTPError as exc:
        status_code = exc.code
        response_body = exc.read()

    except URLError as exc:
        raise RuntimeError(
            "Could not connect to ContextOps API.\n"
            f"URL: {UPLOAD_URL}\n"
            f"Error: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # RESPONSE
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("UPLOAD RESPONSE")
    print("=" * 70)

    print()
    print(f"HTTP STATUS: {status_code}")

    response_text = response_body.decode(
        "utf-8",
        errors="replace",
    )

    print()
    print("Response:")
    print(response_text)

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    if status_code != 202:
        raise RuntimeError(
            "Document upload failed.\n"
            f"HTTP status: {status_code}\n"
            f"Response: {response_text}"
        )

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("UPLOAD ACCEPTED")
    print("=" * 70)

    print()
    print(
        "FastAPI accepted the document with HTTP 202."
    )

    print()
    print(
        "The document ingestion should now be running "
        "as a background task."
    )

    # ---------------------------------------------------------
    # EXPECTED INGESTION FLOW
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPECTED BACKGROUND INGESTION FLOW")
    print("=" * 70)

    print()
    print("  PDF Upload")
    print("      ↓")
    print("  PostgreSQL + Azure Blob")
    print("      ↓")
    print("  Background Ingestion")
    print("      ↓")
    print("  Docling PDF Extraction")
    print("      ↓")
    print("  Structure-aware Chunking")
    print("      ↓")
    print("  Dense Embeddings")
    print("      ↓")
    print("  BM25 Sparse Embeddings")
    print("      ↓")
    print("  Qdrant Upsert")
    print("      ↓")
    print("  INDEXED")

    print()
    print("=" * 70)
    print("WATCH THE UVICORN TERMINAL")
    print("=" * 70)

    print()
    print(
        "The actual ingestion logs will appear in the "
        "FastAPI/Uvicorn terminal."
    )

    print()


def main() -> None:

    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print()
        print(
            "uv run python -m "
            "integration_tests.test_real_pdf_ingestion "
            "<pdf_path>"
        )
        print()
        raise SystemExit(1)

    pdf_path = Path(
        sys.argv[1]
    ).resolve()

    upload_document(
        pdf_path
    )


if __name__ == "__main__":
    main()