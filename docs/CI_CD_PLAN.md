# CI/CD Plan

## Goals

- Verify every pull request and every push to `main`.
- Keep backend and frontend checks independent and fast enough to run often.
- Publish a deployable backend container image after `main` changes.
- Deploy the static frontend to GitHub Pages after `main` changes.
- Keep secrets out of Git. Runtime credentials live in GitHub Actions secrets,
  repository variables, or the actual deployment platform.

## Current Pipeline

### CI

Workflow: `.github/workflows/ci.yml`

Runs on pull requests and pushes to `main`.

Backend job:

- starts PostgreSQL, Qdrant, and Neo4j service containers
- installs Python dependencies with `pip install -e ".[dev]"`
- runs `pytest -q`

Frontend job:

- installs frontend dependencies with `npm ci`
- runs `npm test -- --reporter=dot`
- runs `npm run build`

### Backend Image Publishing

Workflow: `.github/workflows/docker-publish.yml`

Runs on pushes to `main` and version tags like `v1.2.3`.

Output:

- `ghcr.io/<owner>/<repo>/api:<commit-sha>`
- `ghcr.io/<owner>/<repo>/api:main`
- tag image for version tags

This is the deployable backend artifact. A host such as a VPS, Render,
Fly.io, Railway, ECS, or Kubernetes can pull this image and run it with
PostgreSQL, Qdrant, Neo4j, and LLM environment variables.

### Frontend Deployment

Workflow: `.github/workflows/pages.yml`

Runs on pushes to `main` and manual dispatch.

Output:

- GitHub Pages deployment of `frontend/dist`

Required repository variable for a real hosted frontend:

```text
VITE_API_BASE_URL=https://<your-backend-host>
```

Without that variable, the static frontend will build, but a GitHub Pages
browser session cannot call a local FastAPI server on `localhost:8000`.

## Required Runtime Configuration

Backend deployment secrets:

```text
DATABASE_URL
QDRANT_URL
QDRANT_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
LLM_BASE_URL
LLM_MODEL
LLM_API_KEY
FRONTEND_ORIGINS
```

For the current local demo:

```text
NEO4J_PASSWORD=graphragpass
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## Deployment Strategy

Recommended next production-like setup:

1. Deploy PostgreSQL, Qdrant, and Neo4j as managed services or persistent
   Docker containers on one VPS.
2. Pull the backend image from GHCR:

   ```bash
   docker pull ghcr.io/<owner>/<repo>/api:main
   ```

3. Run the backend with environment variables pointing at those services.
4. Set the GitHub repository variable `VITE_API_BASE_URL` to the public
   backend URL.
5. Enable GitHub Pages with source `GitHub Actions`.
6. Push to `main`; Actions verifies, publishes the backend image, and
   deploys the frontend.

## Verification Gates

Every code change should pass:

- backend unit/integration tests
- frontend unit tests
- frontend production build
- backend Docker image build
- frontend Pages artifact build

Manual smoke after deployment:

- `GET /api/v1/health`
- `GET /api/v1/health/qdrant`
- `GET /api/v1/health/neo4j`
- one semantic query
- one structural query
- one multi-hop query
- one mixed query
- one abstention query

## Current Limitations

GitHub Pages can host only the static frontend. It cannot run FastAPI,
PostgreSQL, Qdrant, or Neo4j. The backend deployment target must be a real
runtime environment. The workflow publishes the backend image to GHCR so
that runtime can update after every push to `main`.
