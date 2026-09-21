import asyncio
import logging

from app.dependencies.container import (
    create_query_service,
)


QUESTION = (
    "What annual revenue is required "
    "for business loan underwriting?"
)

TENANT_ID = "underwriting-test"


async def main() -> None:
    print(
        "1. Creating ContextOps QueryService..."
    )

    query_service = create_query_service()

    try:
        print()
        print(
            "2. Executing real LangGraph query..."
        )

        print(
            f"   Question: {QUESTION}"
        )

        print(
            f"   Tenant: {TENANT_ID}"
        )

        result = await query_service.ask(
            question=QUESTION,
            tenant_id=TENANT_ID,
        )

        print()
        print("3. Query completed")

        print()
        print("ANSWER:")
        print(
            result.get("answer")
        )

        print()
        print("CITATIONS:")
        print(
            result.get("citations")
        )

        print()
        print("RETRIEVED DOCUMENTS:")

        retrieved_documents = (
            result.get(
                "retrieved_documents",
                [],
            )
        )

        print(
            f"Count: "
            f"{len(retrieved_documents)}"
        )

        for index, document in enumerate(
            retrieved_documents,
            start=1,
        ):
            print()

            print(
                f"   RESULT #{index}"
            )

            print(
                f"   Score: "
                f"{document.score}"
            )

            print(
                f"   Chunk ID: "
                f"{document.chunk.chunk_id}"
            )

            print(
                f"   Content: "
                f"{document.chunk.content}"
            )

        print()
        print(
            "RETRIEVAL VALIDATION:"
        )

        print(
            result.get(
                "retrieval_sufficient"
            )
        )

        print(
            result.get(
                "retrieval_confidence"
            )
        )

        print()
        print("GROUNDING:")

        print(
            result.get(
                "grounding_status"
            )
        )

        print(
            result.get(
                "grounding_reason"
            )
        )

        print()
        print(
            "FINAL RESPONSE METADATA:"
        )

        print(
            result.get(
                "final_response_metadata"
            )
        )

        print()
        print("TIMINGS:")

        print(
            result.get(
                "timings"
            )
        )

        print()
        print("SUCCESS")

    finally:
        print()
        print(
            "4. Closing ContextOps resources..."
        )

        await query_service.aclose()

        print(
            "   Resources closed successfully"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    asyncio.run(main())