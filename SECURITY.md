# Security policy

Telecom RAG is a public, experimental research demo. Responsible vulnerability reports
are welcome; it does not carry a formal security SLA or bug bounty.

## Reporting a vulnerability

Please **do not publish exploitable details, secrets, access tokens or personal data**
in public GitHub issues. Prefer GitHub's private vulnerability reporting in the
repository's **Security** tab when it is available, or use a private contact
method advertised by the maintainer.

Include a concise reproduction, the affected route/version, and the likely impact.
Do not test destructive exploits against the live demo or third-party services.

## Security boundaries

- The React frontend receives **no** Cloudflare AI API token or Supabase admin key.
- The API uses a low-privilege Supabase publishable key with database RLS/RPC rules.
- API origins are restricted; input sizes, question lengths and chat rates are bounded.
- Chat content is not deliberately stored in the application's persistent database.
- Public hosting platforms may retain request metadata such as IP addresses and paths.
- The deployment is **not** a hardened multi-tenant production network management system.

Keep secrets in provider environment settings. If a credential is exposed, revoke and
rotate it immediately; deleting it from the current commit alone does not remove it
from Git history.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for operational safeguards.
