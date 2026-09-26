from __future__ import annotations

from typing import Any

from app.evaluation.golden_dataset_builder import (
    build_golden_candidate,
)
from app.evaluation.golden_dataset_builder import (
    TARGET_OBSERVATIONS,
)
from app.observability.langfuse import (
    get_langfuse_client,
)


class LangfuseGoldenDatasetService:
    """
    Fetch Langfuse trace observations and transform them into
    ContextOps golden-dataset candidates.
    """

    def __init__(self, client: Any | None = None) -> None:
        self.client = client or get_langfuse_client()

    def fetch_observations(
        self,
        trace_id: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch all required observations for a Langfuse trace.
        """

        response = self.client.api.observations.get_many(
            trace_id=trace_id,
            fields=(
                "core,"
                "basic,"
                "time,"
                "io,"
                "metadata,"
                "model,"
                "usage,"
                "prompt,"
                "trace_context"
            ),
            limit=1000,
        )

        observations = getattr(response, "data", None)

        if observations is None:
            raise ValueError(
                f"Langfuse returned no observation data for trace "
                f"{trace_id}."
            )

        return [
            self._observation_to_dict(observation)
            for observation in observations
        ]

    def build_from_trace(
        self,
        trace_id: str,
    ) -> dict[str, Any]:
        """
        Fetch a Langfuse trace and build one golden-dataset candidate.
        """

        observations = self.fetch_observations(
            trace_id=trace_id,
        )

        self._validate_required_observations(
            trace_id=trace_id,
            observations=observations,
        )

        return build_golden_candidate(
            trace_id=trace_id,
            observations=observations,
        )

    def build_from_traces(
        self,
        trace_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Build golden-dataset candidates for multiple traces.
        """

        return [
            self.build_from_trace(trace_id)
            for trace_id in trace_ids
        ]

    @staticmethod
    def _observation_to_dict(
        observation: Any,
    ) -> dict[str, Any]:
        """
        Convert a Langfuse SDK observation object into a JSON-compatible
        dictionary.
        """

        if isinstance(observation, dict):
            return observation

        model_dump = getattr(
            observation,
            "model_dump",
            None,
        )

        if callable(model_dump):
            try:
                value = model_dump(
                    mode="json",
                )
            except TypeError:
                value = model_dump()

            if isinstance(value, dict):
                return value

        dict_method = getattr(
            observation,
            "dict",
            None,
        )

        if callable(dict_method):
            value = dict_method()

            if isinstance(value, dict):
                return value

        if hasattr(observation, "__dict__"):
            return dict(vars(observation))

        raise TypeError(
            "Unsupported Langfuse observation type: "
            f"{type(observation)!r}"
        )

    @staticmethod
    def _validate_required_observations(
        trace_id: str,
        observations: list[dict[str, Any]],
    ) -> None:
        """
        Make sure the trace contains the observations required to
        construct a complete golden-dataset candidate.
        """

        available_names = {
            observation.get("name")
            for observation in observations
            if observation.get("name")
        }

        missing = [
            name
            for name in TARGET_OBSERVATIONS
            if name not in available_names
        ]

        if missing:
            raise ValueError(
                "Langfuse trace is missing required observations. "
                f"trace_id={trace_id}, missing={missing}"
            )