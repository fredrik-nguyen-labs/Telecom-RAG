# Contributing

This is a portfolio and research project. Small, focused improvements and reproducible
bug reports are welcome.

1. Check for an existing issue, then open a focused issue or pull request.
2. Keep hosted secrets and any private/subscriber network data out of Git.
3. Make minimal changes that preserve the Streamlit fallback and the shared
   `src/telecom_rag` core.
4. Run `python -m pytest -q tests` with the project dependencies, and
   `npm run build --prefix frontend` for frontend changes.
5. Include clear reproduction steps and evidence for routing/KPI/retrieval changes.

Source material and public dataset licenses must be respected. Don't commit
downloaded documents, generated indexes, evaluation outputs or local credentials.

Security issues should be reported privately according to [SECURITY.md](SECURITY.md).
