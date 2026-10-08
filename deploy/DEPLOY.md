# Deploy checklist (Hugging Face Space, Docker SDK, CPU Basic)

1. **Populate Qdrant Cloud once** (from this machine, with no API server / ingestion running):
   ```powershell
   python scripts/sync_to_cloud.py          # dry run: shows local vs remote counts
   python scripts/sync_to_cloud.py --yes    # copies 19k points + creates payload indexes + verifies them
   ```
2. **Create the Space** (SDK: Docker). Its `README.md` must start with the front matter in
   `deploy/space_readme_header.md` (prepend it to the repo README in the Space repo only).
3. **Space secrets**: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. Do *not* set `QDRANT_FORCE_LOCAL`.
4. Push the repo to the Space. First build downloads CPU torch + 4 models (~10-15 min).
5. Smoke test: `GET <space-url>/health` → `{"status":"ok","chunks":19116,...}`, then ask a question in the UI.
6. GitHub: set repo variable `SPACE_URL` (keepalive workflow) and secrets `GROQ_API_KEY`, `QDRANT_URL`,
   `QDRANT_API_KEY` (eval gate workflow).

Not verified locally (no Docker on the dev machine): the image build itself and Qdrant *server* payload-index
creation. Both are exercised by steps 1 and 4 above; `scripts/ingest_papers.py verify` re-checks the indexes.
