"""
prompt-architect-analyst — AI-fluency analyzer for OpenCode sessions.

Package layout
--------------
config      – constants and secure configuration resolution
errors      – exception hierarchy
utils       – pure helpers (path scrubbing, timestamps, hashing)
tools       – tool name registry and semantic categorization
corpus      – Corpus dataclass (contract between reader and scorer)
reader/     – AbstractReader + OpenCodeReader (SQLite, read-only)
scorer      – five dimensions + analyze() + evidence bundle builder
archetype   – cosine-similarity archetype classifier
analyzer    – LLM providers: Anthropic, OpenAI, Ollama, NoOp
reporter    – self-contained HTML report generator
cli         – argparse entry point + orchestration
"""

__version__ = "0.1.0"
