import asyncio
import sys


def pytest_configure() -> None:
    """
    Configure a Windows event loop compatible with psycopg's
    asynchronous PostgreSQL driver.

    psycopg async connections cannot run on Windows' default
    ProactorEventLoop.
    """

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )