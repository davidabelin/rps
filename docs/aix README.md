
Plan

AIX Multi-Lab Expansion Plan (RPS + Euclidorithm + Polyfolds)
Summary
This plan creates a new umbrella Flask app named aix that hosts multiple AI labs under path prefixes (/rps/*, /euclidorithm/*, /polyfolds/*) while preserving and reusing your current RPS implementation with minimal disruption.
Route compatibility is preserved by redirecting old RPS root routes (/play, /training, /rl) to /rps/*.

A) Fork This Conversation Now (practical method)
Create a repo checkpoint before branching conversation context:
Create a git branch for this point (example: checkpoint/rps-ui-pass2).
Optional tag for easy return.
Start a new chat/conversation window and paste:
Branch name.
Current objective (“AIX multi-lab expansion”).
This plan block.
Keep this conversation unchanged as your RPS-focused thread.
If your UI has a built-in duplicate/fork action, use that first; if not, use the branch + new chat workflow above (it is the reliable equivalent).
C) Original Plan Status (where we are now)
Milestone completion snapshot
Milestone 0 (scaffold/app factory/base pages): Complete.
Milestone 1 (notebook archaeology + heuristic migration): Complete enough for production use.
Milestone 2 (engine + deterministic simulation/tournaments): Complete.
Milestone 3 (human-vs-AI page): Complete.
Milestone 4 (supervised pipeline + model registry): Complete.
Milestone 5 (training UI + job lifecycle): Complete.
Milestone 6 (deployment hardening/App Engine/Cloud Tasks/Cloud SQL/Secret Manager): Mostly complete.
Milestone 7 (RL page + RL training): Complete for current scope (tabular RL implementation is functioning).
Intended-but-not-fully-finished items
True split worker as a separate deployable service is not yet fully separated from the web service.
Cross-browser validation is not fully closed (especially iOS Safari priority level noted earlier).
Some polish/documentation and final acceptance criteria remain operational rather than architectural.
Minimum remaining to declare original RPS aim “done”
Run and close final UX/device acceptance matrix (Desktop Chrome, Android Chrome, iOS Safari basic play flow).
Freeze RPS v1 API and route behavior with a release checklist.
Decide whether “separate worker service” is mandatory for v1 completion or deferred to v1.1.
D) Expansion Plan for geometry/ and new top-level app
Decisions locked
App shape: One Flask app.
Path scheme: Lab prefixes (/rps/*, /euclidorithm/*, /polyfolds/*).
Geometry intake: Wrap then refactor.
Route compatibility: Redirect old routes.
Top-level app/package name: aix.
Target architecture
Add a new umbrella package: aix_web.
Keep existing RPS packages (rps_web, rps_core, rps_agents, rps_training, rps_rl, rps_storage) intact initially.
Add lab adapters in aix_web that mount each lab under URL prefixes.
Gradually normalize geometry/euclidorithm and geometry/polyfolds to the same shell conventions used by RPS.
Ideal local file structure (target state)
aix_web/__init__.py (new app factory).
aix_web/lab_registry.py (lab metadata and registration contract).
aix_web/blueprints/hub.py (global splash/home and shared navigation).
aix_web/blueprints/routes_compat.py (legacy redirect routes).
aix_web/templates/base_hub.html (shared shell).
aix_web/static/css/hub_theme.css (shared theme tokens + lab overrides).
aix_web/labs/rps_adapter.py (mount existing RPS blueprints at /rps).
aix_web/labs/euclidorithm_adapter.py (mount geometry/euclidorithm first as wrapped module).
aix_web/labs/polyfolds_adapter.py (initial placeholder + data/training module bridge).
geometry/euclidorithm/* (existing code retained initially, wrapped).
geometry/polyfolds/* (existing/ongoing data-generation assets retained initially).
Important changes/additions to public APIs/interfaces/types
New URL contract:
/ => AIX hub page.
/rps/* => existing RPS pages/APIs.
/euclidorithm/* => Euclidorithm lab pages/APIs.
/polyfolds/* => Polyfolds lab pages/APIs.
Backward-compat redirects:
/play -> /rps/play
/training -> /rps/training
/rl -> /rps/rl
New lab registration interface:
LabSpec(slug, display_name, blueprint, nav_order, theme_tokens, enabled)
One registry function to register all enabled labs.
Shared UI contract:
Every lab page extends shared hub base template.
Lab-specific color tokens override shared CSS variables only.
Reuse strategy (maximum reuse, minimum breakage)
Reuse current RPS code as-is via adapter mounting first.
Reuse rps_web/static/css/theme.css concepts by extracting only shared variables/components to hub theme.
Preserve existing RPS API endpoints and job system under /rps/api/v1/*.
Introduce geometry labs in “compatibility mode” first, then progressively refactor internals.
Cloud migration path (for future separate GCP project)
New App Engine service/project for AIX with same deployment model as RPS.
Keep per-lab env prefixes in config:
RPS_*, EUCLID_*, POLY_* (storage paths, DB URLs, queues, secrets).
Secret Manager naming convention:
aix-rps-database-url, aix-euclid-database-url, aix-poly-database-url, etc.
Cloud Tasks queue split by lab:
rps-training, polyfolds-training (euclidorithm queue only if needed).
Cloud Storage split by lab prefix:
gs://<bucket>/rps/...
gs://<bucket>/euclidorithm/...
gs://<bucket>/polyfolds/...
Keep RPS cloud deployment stable during AIX build; do not couple cutover dates.
Implementation phases (high-level, not code-level detail)
Phase 0: Scaffold aix_web shell and hub homepage.
Phase 1: Mount RPS under /rps/* via adapter and add legacy redirects.
Phase 2: Add geometry/euclidorithm wrapper routes and shared shell integration.
Phase 3: Add geometry/polyfolds wrapper with placeholder web lab + data hooks.
Phase 4: Shared theming + navigation consistency pass across all labs.
Phase 5: AIX deployment config for new GCP project (staging first).
Test cases and scenarios
Routing:
/rps/play, /euclidorithm, /polyfolds all resolve.
/play, /training, /rl redirect correctly to /rps/*.
UI shell:
Shared nav works on desktop/mobile with no horizontal overflow.
Lab isolation:
RPS APIs still work unchanged under /rps/api/v1/*.
Geometry modules do not break RPS runtime imports.
Data paths:
Per-lab storage paths and secrets resolve independently.
Deployment:
Local Flask start for AIX works.
App Engine staging deploy for AIX starts and serves hub + labs.
Explicit assumptions and defaults
geometry/ will be added soon and initially may not match RPS conventions.
We prioritize architectural compatibility first, deep geometry refactors second.
RPS remains the production-stable lab while AIX is built.
We treat split worker separation as optional for AIX v1 unless explicitly required.
We avoid mass file moves in the first integration pass to reduce regression risk.

