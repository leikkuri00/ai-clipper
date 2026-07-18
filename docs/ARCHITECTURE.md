# Architecture Overview — Local AI Coding Agent

This document outlines the planned architecture for the local AI coding agent.

Components:

- Installer scripts (PowerShell)
- Local model runtime using llama.cpp or compatible backends
- Agent core (indexing, planner, executor)
- Memory (vector DB + metadata)
- Obsidian integration for notes and memory
- VS Code integration via workspace settings and recommended extensions

Design choices:
- 100% FOSS stack: Python, llama.cpp, FAISS/Annoy, SQLite
- Modular design: clear separations between runtime, agent, memory, and UI
- Secure by default: require explicit confirmation for destructive actions

Next steps: implement model runtime wrapper, indexing, and planner components.
