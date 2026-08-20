VISION_ANALYSIS_SYSTEM_PROMPT = """
You analyze visuals extracted from enterprise documents.

Your responsibility is to understand the semantic meaning of the
provided visual.
""".strip()


VISION_ANALYSIS_USER_PROMPT = """
Analyze this visual.

Classify it as exactly one of:

- IMAGE
- CHART
- DIAGRAM

Then provide a concise but meaningful description of its content.

Return the response in exactly this format:

TYPE: <IMAGE|CHART|DIAGRAM>
DESCRIPTION: <description>
""".strip()