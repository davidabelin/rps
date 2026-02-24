# UI Acceptance Checklist (Step 1)

This checklist defines concrete acceptance criteria before and during UI polish work.

## Scope and priorities

- Priority 1: desktop Chrome (admin and development workflow).
- Priority 2: Android Chrome (primary expected player traffic).
- Priority 3: iOS Safari compatibility for the play loop.

## Global UX criteria

- Navigation is visible and usable on all pages (`Home`, `Play`, `Training`, `RL`).
- Interactive controls are touch-friendly on mobile (minimum 44px tap target).
- No horizontal overflow on mobile except intentionally scrollable data tables.
- Error states are shown as readable text without needing browser console access.
- Loading/working states are visible for all network actions longer than 150 ms.

## Home page criteria

- Hero content is readable at widths from 360px to 1440px.
- Primary calls-to-action route to `Play` and `Training` without broken links.
- Card layout wraps cleanly on narrow screens with no clipped text.

## Play page criteria

- Player can create a game and complete at least 20 rounds without UI breakage.
- Round outcome, both actions, and running score update every round.
- Round log appends newest rounds and remains readable on mobile.
- During round submission, duplicate clicks are prevented.
- UI exposes clear current-opponent context (selected heuristic or active model).
- Latency/perceived-progress feedback appears while waiting for round response.

## Training page criteria

- Training form fields are editable on desktop and Android Chrome.
- Readiness status updates when `Lookback` changes.
- Job state transitions are visible: `queued`, `running`, `completed` or `failed`.
- Failure states include error details in-page.
- Model registry refresh and activation actions work end-to-end.
- Benchmark panel supports suite selection and returns JSON results visibly.

## RL page criteria

- RL job creation works with default form values.
- Job status/progress updates appear without full page refresh.
- Recent jobs table refresh shows new and completed jobs.
- Failed jobs show error text in-page.

## Browser/device validation matrix

- Desktop Chrome (latest): full pass on all pages.
- Android Chrome (latest, Pixel-class device): full pass on `Home`, `Play`, `Training`.
- iOS Safari (recent version): pass at minimum on `Home` and `Play` interactions.

## Performance and responsiveness checks

- Play round request-response median under 250 ms in local/dev environment.
- No noticeable input lag from button press to pending visual state.
- No console errors during normal flows.

## Exit criteria for Step 1

- All Priority 1 criteria pass.
- All Priority 2 criteria pass, with only minor cosmetic issues allowed.
- Priority 3 has no blocking defects for basic play flow.
- Open defects are documented with severity and page references.
