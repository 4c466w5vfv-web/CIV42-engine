from __future__ import annotations
"""ARK-42 Decision Lab — private decision engine.

Pipeline: natural-language problem → discipline selection (LLM)
→ independent per-discipline analysis (LLM) → common-ontology translation
→ score tensor → deterministic aggregation + sensitivity → Monte Carlo
→ numbers + report → human decision → outcome recording.
"""
__version__ = "0.1.0"
