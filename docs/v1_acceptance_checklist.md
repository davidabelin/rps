# RPS v1 Acceptance Run
Date:
Tester:
Build/Commit:
URL:
Devices/Browsers:

## 1) Home Page Render
- [X] PASS  [ ] FAIL
- Check: `/` loads; nav visible; no broken layout/text clipping.

## 2) Play Page Basic Loop (Desktop Chrome)
- [X] PASS  [ ] FAIL
- Check: create game, play 20 rounds, score/log/outcome update each round.

## 3) Play Page Mobile Loop (Android Chrome)
- [X] PASS  [ ] FAIL
- Check: create game, play 20 rounds, no horizontal overflow, controls usable.

## 4) Opponent/Agent Selection
- [X] PASS  [ ] FAIL
- Check: agent selector works; hidden trivial agents stay hidden for human play; details text updates.

## 5) Animation + Outcome UX
- [X] PASS  [ ] FAIL
- Check: countdown/reveal/history trail/win-loss outlines/clash effect render correctly and don’t break flow.

## 6) Training Page Form + Readiness
- [X] PASS  [ ] FAIL
- Check: fields editable (lookback/model/lr/epochs/batch); readiness updates correctly.

## 7) Supervised Training Job Lifecycle
- [X] PASS  [ ] FAIL
- Check: submit job -> `queued` -> `running` -> `completed` (or clear `failed` with error shown).

## 8) Benchmark Panel
- [X] PASS  [ ] FAIL
- Check: core/extended suite runs return JSON results; timeout errors are handled cleanly in-page.

## 9) RL Page Job Lifecycle
- [X] PASS  [ ] FAIL
- Check: create RL job; progress updates; completion/failure displayed correctly; model appears in registry.

## 10) Model Activation + Play with Active Model
- [X] PASS  [ ] FAIL
- Check: activate model on training page, then play against `active_model` successfully.

---

## Notes / Defects
- Severity:
- Steps to reproduce:
- Expected:
- Actual:
- Screenshot/Log link:

## Final Decision
- [X] ACCEPT for `v1.0.0`
- [ ] REJECT (fixes required before release)
- Rationale:
