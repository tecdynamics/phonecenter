"""LLM layer — Gemma (llama.cpp, OpenAI-compatible) streaming chat for phase 2.

Kept deliberately thin per spec §4: a direct OpenAI-compatible request, no
LangChain / agent framework. The persona system prompt is loaded from an
external per-client file (see `conversation.load_system_prompt`); token output
is chunked into TTS-able clauses by `clause.ClauseAggregator`.
"""
