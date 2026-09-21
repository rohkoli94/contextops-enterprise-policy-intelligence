import asyncio

from app.dependencies.rag import (
    get_hybrid_retriever,
    get_vector_store,
)


QUERY = (
    "What annual revenue is required "
    "for business loan underwriting?"
)

TENANT_ID = "underwriting-test"
TOP_K = 5

EXPECTED_CONTENT = (
    "Business loan applications require a minimum annual "
    "revenue of $500,000 for underwriting review."
)


async def main() -> None:
    print(
        "1. Getting ContextOps HybridRetriever..."
    )

    hybrid_retriever = (
        get_hybrid_retriever()
    )

    vector_store = get_vector_store()

    try:
        print()
        print(
            "2. Running hybrid retrieval..."
        )

        print(
            f"   Query: {QUERY}"
        )

        print(
            f"   Tenant: {TENANT_ID}"
        )

        print(
            f"   Top-K: {TOP_K}"
        )

        results = (
            await hybrid_retriever.aretrieve(
                query=QUERY,
                tenant_id=TENANT_ID,
                top_k=TOP_K,
            )
        )

        print()
        print(
            f"3. Retrieved results: "
            f"{len(results)}"
        )

        for index, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"   HYBRID RESULT #{index}"
            )

            print(
                f"   Score: {result.score}"
            )

            print(
                f"   Chunk ID: "
                f"{result.chunk.chunk_id}"
            )

            print(
                f"   Content: "
                f"{result.chunk.content}"
            )

            print(
                f"   Tenant: "
                f"{result.chunk.metadata.get('tenant_id')}"
            )

        print()
        print(
            "4. Validating expected "
            "underwriting chunk..."
        )

        expected_result = next(
            (
                result
                for result in results
                if result.chunk.content
                == EXPECTED_CONTENT
            ),
            None,
        )

        if expected_result is None:
            raise AssertionError(
                "Expected business-loan "
                "underwriting chunk was not "
                "returned by HybridRetriever."
            )

        print(
            "   Expected chunk found: YES"
        )

        print(
            f"   Expected chunk score: "
            f"{expected_result.score}"
        )

        print()
        print("SUCCESS")

        print(
            "ContextOps HybridRetriever "
            "successfully retrieved the "
            "business-loan underwriting chunk."
        )

    finally:
        await vector_store.aclose()


if __name__ == "__main__":
    asyncio.run(main())