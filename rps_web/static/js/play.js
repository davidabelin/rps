(function () {
  "use strict";
  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;

  const agentSelect = document.getElementById("agentSelect");
  const agentDetails = document.getElementById("agentDetails");
  const newGameBtn = document.getElementById("newGameBtn");
  const gameStatus = document.getElementById("gameStatus");
  const latencyStatus = document.getElementById("latencyStatus");
  const latencyTelemetryToggle = document.getElementById("latencyTelemetryToggle");
  const latencyDebugToggle = document.getElementById("latencyDebugToggle");
  const momentumStatus = document.getElementById("momentumStatus");
  const clashFx = document.getElementById("clashFx");
  const choicePlayed = document.getElementById("choicePlayed");
  const playerActionEl = document.getElementById("playerAction");
  const aiActionEl = document.getElementById("aiAction");
  const playerHistoryEl = document.getElementById("playerHistory");
  const aiHistoryEl = document.getElementById("aiHistory");
  const outcomeBanner = document.getElementById("outcomeBanner");
  const winsEl = document.getElementById("wins");
  const lossesEl = document.getElementById("losses");
  const tiesEl = document.getElementById("ties");
  const roundsEl = document.getElementById("rounds");
  const roundLog = document.getElementById("roundLog");
  const actionButtons = Array.from(document.querySelectorAll(".btn-action"));

  let currentGame = null;
  let isRoundInFlight = false;
  let currentStreakSign = 0;
  let currentStreakCount = 0;
  let agentsByName = {};
  let activeModelSummary = "none";
  let latencyTelemetryEnabled = false;
  let latencyDebugEnabled = false;
  let playerHistoryTokens = [];
  let aiHistoryTokens = [];
  let currentPlayerToken = null;
  let currentAiToken = null;
  let currentPlayerResult = null;
  let currentAiResult = null;
  const hiddenAgents = new Set(["rock", "paper", "scissors", "copy_opponent"]);
  const agentDisplayNames = {
    statistical: "frequency",
  };
  const historyOpacities = [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.125];
  const resultClasses = ["result-win", "result-loss", "result-tie"];
  let handSvgSerial = 0;

  function displayAgentName(name) {
    return agentDisplayNames[name] || name;
  }

  function loadLatencyPreferences() {
    const telemetryRaw = window.localStorage.getItem("rps.latency.telemetry");
    const debugRaw = window.localStorage.getItem("rps.latency.debug");
    latencyTelemetryEnabled = telemetryRaw === "1";
    latencyDebugEnabled = debugRaw === "1";
    if (latencyTelemetryToggle) {
      latencyTelemetryToggle.checked = latencyTelemetryEnabled;
    }
    if (latencyDebugToggle) {
      latencyDebugToggle.checked = latencyDebugEnabled;
    }
  }

  function saveLatencyPreference(key, enabled) {
    window.localStorage.setItem(key, enabled ? "1" : "0");
  }

  function setStatus(message) {
    gameStatus.textContent = message;
  }

  function normalizeToken(value) {
    return String(value || "").trim().toLowerCase();
  }

  function titleCase(value) {
    const token = String(value || "").trim();
    if (!token) {
      return "-";
    }
    return token[0].toUpperCase() + token.slice(1).toLowerCase();
  }

  function tokenLetter(token) {
    const key = normalizeToken(token);
    if (key === "rock") {
      return "R";
    }
    if (key === "paper") {
      return "P";
    }
    if (key === "scissors") {
      return "S";
    }
    return "";
  }

  function clearResultClasses(element) {
    element.classList.remove(...resultClasses);
  }

  function applyResultClass(element, result) {
    clearResultClasses(element);
    const key = normalizeToken(result);
    if (key === "win") {
      element.classList.add("result-win");
      return;
    }
    if (key === "loss") {
      element.classList.add("result-loss");
      return;
    }
    if (key === "tie") {
      element.classList.add("result-tie");
    }
  }

  function perspectiveResult(outcome, perspective) {
    const key = normalizeToken(outcome);
    if (key === "tie") {
      return "tie";
    }
    if (perspective === "player") {
      return key === "player" ? "win" : "loss";
    }
    return key === "ai" ? "win" : "loss";
  }

  function updateChoicePlayed(playerAction, aiAction) {
    if (!choicePlayed) {
      return;
    }
    choicePlayed.textContent = `You: ${titleCase(playerAction)} | Agent: ${titleCase(aiAction)}`;
  }

  function handPartsForToken(token, skinId) {
    const key = normalizeToken(token);
    const finger = (x, y, h, w = 10, rot = 0, r = 5.6) => `
      <rect
        x="${x}"
        y="${y}"
        width="${w}"
        height="${h}"
        rx="${r}"
        ry="${r}"
        transform="${rot ? `rotate(${rot} ${x + w / 2} ${y + h / 2})` : ""}"
        fill="url(#${skinId})"
        stroke="#2a2a2d"
        stroke-width="2.2"
      />
    `;
    const knuckle = (cx, cy, r = 5.2) => `
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2.1"/>
    `;
    const foldedKnuckles = () =>
      [knuckle(56, 56, 5.6), knuckle(68, 55.5, 5.5), knuckle(79, 57, 5.2), knuckle(86, 63, 4.6)].join("");
    if (key === "paper") {
      return [
        finger(35, 18, 42, 11, -3, 5.7),
        finger(46, 13, 48, 11, -1, 5.8),
        finger(58, 12, 50, 11, 1, 5.8),
        finger(70, 17, 44, 11, 4, 5.7),
        `<path d="M80 49 Q88 56 84 66 Q76 67 73 60 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      ].join("");
    }
    if (key === "scissors") {
      return [
        finger(39, 13, 47, 11, -18, 5.8),
        finger(66, 13, 47, 11, 18, 5.8),
        knuckle(60, 56, 5.2),
        knuckle(72, 56, 5.2),
        `<path d="M57 58 L63 64 L69 58" fill="none" stroke="#2a2a2d" stroke-width="2" stroke-linecap="round"/>`,
        `<path d="M78 63 Q85 64 87 69 Q82 72 76 69 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      ].join("");
    }
    if (key === "one") {
      return [
        finger(44, 10, 56, 11, -3, 5.8),
        knuckle(58, 56, 5.6),
        knuckle(70, 55.5, 5.4),
        knuckle(81, 57, 5.1),
        `<path d="M78 63 Q86 64 87 70 Q82 74 75 70 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      ].join("");
    }
    if (key === "two") {
      return [
        finger(43, 12, 52, 10, -2, 5.8),
        finger(55, 11, 53, 10, 1, 5.8),
        knuckle(70, 56, 5.3),
        knuckle(81, 58, 5.1),
        `<path d="M77 63 Q85 65 86 70 Q82 73 76 70 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      ].join("");
    }
    return [
      `<rect x="41" y="45" width="12" height="17" rx="6" ry="6" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      `<rect x="53" y="43" width="12" height="18" rx="6" ry="6" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      `<rect x="65" y="43" width="12" height="18" rx="6" ry="6" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      `<rect x="77" y="45" width="10" height="16" rx="5" ry="5" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
      foldedKnuckles(),
      `<path d="M84 62 Q89 65 89 71 Q83 73 79 69 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2"/>`,
    ].join("");
  }

  function buildHandSvg(token, options) {
    const opts = options || {};
    const mirrorTransform = opts.mirror ? ' transform="translate(120 0) scale(-1 1)"' : "";
    handSvgSerial += 1;
    const skinId = `skinFill${handSvgSerial}`;
    const cuffId = `cuffFill${handSvgSerial}`;
    const badgeOuterId = `badgeOuter${handSvgSerial}`;
    const badgeInnerId = `badgeInner${handSvgSerial}`;
    const parts = handPartsForToken(token, skinId);
    return `
      <svg viewBox="0 0 120 120" class="gesture-svg" role="img" aria-label="${normalizeToken(token)} hand sign">
        <defs>
          <linearGradient id="${skinId}" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#fff0c8"/>
            <stop offset="100%" stop-color="#f8cf88"/>
          </linearGradient>
          <linearGradient id="${cuffId}" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#4f84df"/>
            <stop offset="100%" stop-color="#274f9f"/>
          </linearGradient>
          <radialGradient id="${badgeInnerId}" cx="32%" cy="24%" r="78%">
            <stop offset="0%" stop-color="#ffe98b"/>
            <stop offset="50%" stop-color="#f2d248"/>
            <stop offset="100%" stop-color="#cc9f1b"/>
          </radialGradient>
          <linearGradient id="${badgeOuterId}" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#6e7ba0"/>
            <stop offset="50%" stop-color="#2d3752"/>
            <stop offset="100%" stop-color="#12192f"/>
          </linearGradient>
        </defs>
        <g>
          <circle cx="60" cy="60" r="54" fill="url(#${badgeOuterId})" stroke="#f0e7cb" stroke-width="2"/>
          <circle cx="60" cy="60" r="47" fill="url(#${badgeInnerId})" stroke="#f7edcf" stroke-width="1.8"/>
          <ellipse cx="49" cy="36" rx="20" ry="10" fill="rgba(255,255,255,0.28)"/>
          <ellipse cx="74" cy="84" rx="17" ry="8" fill="rgba(0,0,0,0.09)"/>
        </g>
        <g${mirrorTransform}>
          <rect x="24" y="83" width="72" height="21" rx="8" ry="8" fill="url(#${cuffId})" stroke="#1f2b5f" stroke-width="2.1"/>
          <rect x="26" y="78" width="68" height="9" rx="5" ry="5" fill="#f4d24a" stroke="#997512" stroke-width="1.4"/>
          <path d="M36 84 C35 63 45 46 61 45 C77 44 88 55 87 76 L87 88 L36 88 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2.4"/>
          <path d="M30 63 C25 66 23 71 24 77 C25 82 29 86 34 86 C39 86 41 81 40 76 C39 71 37 67 34 63 Z" fill="url(#${skinId})" stroke="#2a2a2d" stroke-width="2.2"/>
          <path d="M42 70 Q60 63 83 68" fill="none" stroke="rgba(42,42,45,0.2)" stroke-width="1.7"/>
          ${parts}
        </g>
      </svg>
    `;
  }

  function buildHandMarkup(token, options) {
    const letter = tokenLetter(token);
    const label = letter ? `<span class="gesture-label">${letter}</span>` : "";
    return `<div class="gesture-wrap">${buildHandSvg(token, options)}${label}</div>`;
  }

  function resetActionChip(element) {
    element.innerHTML = '<span class="idle-mark">-</span>';
    element.classList.add("idle");
    element.classList.remove("pending");
    element.classList.remove("reveal");
    element.classList.remove("winner-flash");
    clearResultClasses(element);
  }

  function setActionChipIcon(element, token, mirror) {
    element.innerHTML = buildHandMarkup(token, { mirror });
    element.classList.remove("idle");
  }

  function renderHistoryLane(target, tokens, mirror, lane) {
    target.innerHTML = "";
    tokens.slice(0, 8).forEach((token, index) => {
      const payload = typeof token === "string" ? { token, result: null } : token;
      const icon = document.createElement("div");
      icon.className = "history-icon";
      if (index === 0) {
        icon.classList.add("latest");
      }
      applyResultClass(icon, payload.result);
      icon.style.opacity = String(historyOpacities[index] !== undefined ? historyOpacities[index] : 0.1);
      const arcDrop = index < 2 ? 0 : (index - 1) * 3;
      const arcTilt = index < 2 ? 0 : (lane === "player" ? -1 : 1) * Math.min(5, index - 1);
      icon.style.transform = `translateY(${arcDrop}px) rotate(${arcTilt}deg)`;
      if (lane === "player") {
        icon.style.marginRight = `${index === 0 ? 0 : index < 2 ? 6 : -14}px`;
      } else {
        icon.style.marginLeft = `${index === 0 ? 0 : index < 2 ? 6 : -14}px`;
      }
      icon.style.zIndex = String(80 - index);
      icon.innerHTML = buildHandMarkup(payload.token, { mirror });
      target.appendChild(icon);
    });
  }

  function renderHistories() {
    renderHistoryLane(playerHistoryEl, playerHistoryTokens, false, "player");
    renderHistoryLane(aiHistoryEl, aiHistoryTokens, true, "ai");
  }

  function archiveCurrentCenter() {
    if (!currentPlayerToken || !currentAiToken) {
      return;
    }
    playerHistoryTokens.unshift({ token: currentPlayerToken, result: currentPlayerResult });
    aiHistoryTokens.unshift({ token: currentAiToken, result: currentAiResult });
    playerHistoryTokens = playerHistoryTokens.slice(0, 8);
    aiHistoryTokens = aiHistoryTokens.slice(0, 8);
    renderHistories();
  }

  function triggerClash() {
    if (!clashFx) {
      return;
    }
    clashFx.classList.remove("active");
    void clashFx.offsetWidth;
    clashFx.classList.add("active");
  }

  function triggerWinnerFlash(element) {
    element.classList.remove("winner-flash");
    void element.offsetWidth;
    element.classList.add("winner-flash");
    window.setTimeout(() => {
      element.classList.remove("winner-flash");
    }, 520);
  }

  function delay(ms) {
    return new Promise((resolve) => {
      window.setTimeout(resolve, ms);
    });
  }

  function startCountdown() {
    let cancelled = false;
    void (async () => {
      setActionChipIcon(playerActionEl, "one", false);
      setActionChipIcon(aiActionEl, "one", true);
      await delay(500);
      if (cancelled) {
        return;
      }
      setActionChipIcon(playerActionEl, "two", false);
      setActionChipIcon(aiActionEl, "two", true);
      await delay(500);
    })();
    return () => {
      cancelled = true;
    };
  }

  function setOutcome(text, kind) {
    outcomeBanner.textContent = text;
    outcomeBanner.classList.remove("win", "loss", "tie", "pending");
    if (kind) {
      outcomeBanner.classList.add(kind);
    }
  }

  function setRoundInteractionEnabled(enabled) {
    actionButtons.forEach((button) => {
      button.disabled = !enabled;
    });
    newGameBtn.disabled = !enabled;
    isRoundInFlight = !enabled;
  }

  function animateAction(element, value, result) {
    const token = normalizeToken(value);
    const mirror = element === aiActionEl;
    clearResultClasses(element);
    setActionChipIcon(element, token, mirror);
    applyResultClass(element, result);
    element.classList.remove("pending");
    element.classList.remove("reveal");
    window.requestAnimationFrame(() => {
      element.classList.add("reveal");
      window.setTimeout(() => {
        element.classList.remove("reveal");
      }, 280);
    });
  }

  function updateScore(game) {
    winsEl.textContent = String(game.score_player);
    lossesEl.textContent = String(game.score_ai);
    tiesEl.textContent = String(game.score_ties);
    roundsEl.textContent = String(game.rounds_played);
  }

  function updateMomentum(outcome) {
    let sign = 0;
    if (outcome === "player") {
      sign = 1;
    } else if (outcome === "ai") {
      sign = -1;
    }

    if (sign === 0) {
      currentStreakSign = 0;
      currentStreakCount = 0;
      momentumStatus.textContent = "Momentum: neutral (tie round).";
      return;
    }

    if (sign === currentStreakSign) {
      currentStreakCount += 1;
    } else {
      currentStreakSign = sign;
      currentStreakCount = 1;
    }

    if (currentStreakSign > 0) {
      momentumStatus.textContent = `Momentum: you are on a ${currentStreakCount}-round streak.`;
      return;
    }
    momentumStatus.textContent = `Momentum: Agent is on a ${currentStreakCount}-round streak.`;
  }

  function addLogRow(round) {
    const row = document.createElement("li");
    row.textContent = `#${round.round_index + 1}: you ${round.player_action_name}, agent ${round.ai_action_name}, outcome ${round.outcome}`;
    roundLog.prepend(row);
  }

  function sendLatencyTelemetry(sample) {
    if (!latencyTelemetryEnabled) {
      return;
    }
    const payload = JSON.stringify(sample);
    if (navigator.sendBeacon) {
      const blob = new Blob([payload], { type: "application/json" });
      navigator.sendBeacon(`${apiBase}/telemetry/latency`, blob);
      return;
    }
    fetch(`${apiBase}/telemetry/latency`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      keepalive: true,
    }).catch(() => null);
  }

  function updateAgentDetails() {
    const selected = String(agentSelect.value || "");
    const descriptor = agentsByName[selected];
    if (!descriptor) {
      agentDetails.textContent = "Choose an agent to view strategy details.";
      return;
    }
    const shownName = displayAgentName(selected);
    if (selected === "active_model") {
      agentDetails.textContent = `${descriptor.description} Active model: ${activeModelSummary}.`;
      return;
    }
    agentDetails.textContent = `${shownName}: ${descriptor.description}`;
  }

  async function fetchJsonWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...(options || {}), signal: controller.signal });
      const body = await response.json();
      return { response, body };
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  async function fetchActiveModelSummary() {
    try {
      const { response, body } = await fetchJsonWithTimeout(`${apiBase}/models`, {}, 5000);
      if (!response.ok || !Array.isArray(body.models)) {
        activeModelSummary = "none";
        return;
      }
      const active = body.models.find((model) => Boolean(model.is_active));
      if (!active) {
        activeModelSummary = "none";
        return;
      }
      activeModelSummary = `${active.name} (#${active.id}, ${active.model_type})`;
    } catch (err) {
      activeModelSummary = "unknown";
    }
  }

  async function fetchAgents() {
    const { response, body } = await fetchJsonWithTimeout(`${apiBase}/agents`, {}, 5000);
    if (!response.ok) {
      throw new Error(body.error || "Failed to fetch agents");
    }
    agentsByName = {};
    agentSelect.innerHTML = "";
    body.agents.forEach((agent) => {
      if (hiddenAgents.has(agent.name)) {
        return;
      }
      agentsByName[agent.name] = agent;
      const option = document.createElement("option");
      option.value = agent.name;
      option.textContent = `${displayAgentName(agent.name)}`;
      agentSelect.appendChild(option);
    });
    await fetchActiveModelSummary();
    updateAgentDetails();
  }

  async function createGame() {
    if (isRoundInFlight) {
      return;
    }
    setRoundInteractionEnabled(false);
    try {
      const { response, body } = await fetchJsonWithTimeout(
        `${apiBase}/games`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agent: agentSelect.value }),
        },
        7000
      );
      if (!response.ok) {
        setStatus(`Failed to create game: ${body.error || "unknown error"}`);
        return;
      }
      currentGame = body.game;
      currentStreakSign = 0;
      currentStreakCount = 0;
      currentPlayerToken = null;
      currentAiToken = null;
      currentPlayerResult = null;
      currentAiResult = null;
      playerHistoryTokens = [];
      aiHistoryTokens = [];
      renderHistories();
      updateScore(currentGame);
      roundLog.innerHTML = "";
      resetActionChip(playerActionEl);
      resetActionChip(aiActionEl);
      updateChoicePlayed("-", "-");
      setOutcome("Game ready. Make your move.", null);
      momentumStatus.textContent = "Momentum: neutral.";
      setStatus(`Game ${currentGame.game_id} using ${currentGame.agent_name}`);
    } catch (err) {
      setStatus(`Failed to create game: ${String(err)}`);
    } finally {
      setRoundInteractionEnabled(true);
    }
  }

  async function playRound(action) {
    if (!currentGame) {
      setStatus("Create a game first.");
      return;
    }
    if (isRoundInFlight) {
      return;
    }
    setRoundInteractionEnabled(false);
    setOutcome("Agent thinking...", null);
    outcomeBanner.classList.add("pending");
    const stopCountdown = startCountdown();
    playerActionEl.classList.add("pending");
    aiActionEl.classList.add("pending");
    clearResultClasses(playerActionEl);
    clearResultClasses(aiActionEl);
    const startedAt = window.performance.now();
    try {
      const { response, body } = await fetchJsonWithTimeout(
        `${apiBase}/games/${currentGame.game_id}/round`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        },
        10000
      );
      const elapsedForCountdown = window.performance.now() - startedAt;
      if (elapsedForCountdown < 1000) {
        await delay(1000 - elapsedForCountdown);
      }
      const elapsedMs = Math.round(window.performance.now() - startedAt);
      const serverMs = Number(body?.round?.server_elapsed_ms);
      const timings = body?.round?.timings_ms || {};
      const loadMs = Number(timings.cache_or_load);
      const agentMs = Number(timings.agent_step);
      const persistMs = Number(timings.persist);
      const timingsKnown = Number.isFinite(loadMs) && Number.isFinite(agentMs) && Number.isFinite(persistMs);
      if (latencyDebugEnabled && Number.isFinite(serverMs)) {
        latencyStatus.textContent = `Last round latency: client ${elapsedMs} ms, server ${serverMs} ms${
          timingsKnown ? ` (load ${loadMs}, agent ${agentMs}, persist ${persistMs})` : ""
        }`;
      } else {
        latencyStatus.textContent = `Last round latency: ${elapsedMs} ms`;
      }
      if (!response.ok) {
        stopCountdown();
        setStatus(`Round error: ${body.error || "unknown error"}`);
        setOutcome("Round failed. Try again.", null);
        if (currentPlayerToken && currentAiToken) {
          setActionChipIcon(playerActionEl, currentPlayerToken, false);
          setActionChipIcon(aiActionEl, currentAiToken, true);
          applyResultClass(playerActionEl, currentPlayerResult);
          applyResultClass(aiActionEl, currentAiResult);
        } else {
          resetActionChip(playerActionEl);
          resetActionChip(aiActionEl);
        }
        return;
      }
      stopCountdown();
      archiveCurrentCenter();
      currentGame = body.game;
      updateScore(currentGame);
      currentPlayerToken = normalizeToken(body.round.player_action_name);
      currentAiToken = normalizeToken(body.round.ai_action_name);
      currentPlayerResult = perspectiveResult(body.round.outcome, "player");
      currentAiResult = perspectiveResult(body.round.outcome, "ai");
      animateAction(playerActionEl, body.round.player_action_name, currentPlayerResult);
      animateAction(aiActionEl, body.round.ai_action_name, currentAiResult);
      updateChoicePlayed(body.round.player_action_name, body.round.ai_action_name);
      triggerClash();
      if (body.round.outcome === "player") {
        setOutcome("You win this round.", "win");
        triggerWinnerFlash(playerActionEl);
      } else if (body.round.outcome === "ai") {
        setOutcome("Agent wins this round.", "loss");
        triggerWinnerFlash(aiActionEl);
      } else {
        setOutcome("Tie round.", "tie");
      }
      updateMomentum(body.round.outcome);
      addLogRow(body.round);
      if (body.round && Number.isFinite(serverMs)) {
        sendLatencyTelemetry({
          game_id: Number(currentGame.game_id),
          round_id: Number(body.round.id),
          round_index: Number(body.round.round_index),
          agent_name: String(currentGame.agent_name || ""),
          client_elapsed_ms: Number(elapsedMs),
          server_elapsed_ms: Number(serverMs),
          timings_ms: body.round.timings_ms || null,
        });
      }
    } catch (err) {
      stopCountdown();
      const isTimeout = err && String(err.name || "").toLowerCase() === "aborterror";
      setStatus(`Round error: ${isTimeout ? "request timed out" : String(err)}`);
      setOutcome("Round failed. Try again.", null);
      if (currentPlayerToken && currentAiToken) {
        setActionChipIcon(playerActionEl, currentPlayerToken, false);
        setActionChipIcon(aiActionEl, currentAiToken, true);
        applyResultClass(playerActionEl, currentPlayerResult);
        applyResultClass(aiActionEl, currentAiResult);
      } else {
        resetActionChip(playerActionEl);
        resetActionChip(aiActionEl);
      }
    } finally {
      setRoundInteractionEnabled(true);
    }
  }

  newGameBtn.addEventListener("click", createGame);
  agentSelect.addEventListener("change", updateAgentDetails);
  if (latencyTelemetryToggle) {
    latencyTelemetryToggle.addEventListener("change", () => {
      latencyTelemetryEnabled = Boolean(latencyTelemetryToggle.checked);
      saveLatencyPreference("rps.latency.telemetry", latencyTelemetryEnabled);
    });
  }
  if (latencyDebugToggle) {
    latencyDebugToggle.addEventListener("change", () => {
      latencyDebugEnabled = Boolean(latencyDebugToggle.checked);
      saveLatencyPreference("rps.latency.debug", latencyDebugEnabled);
    });
  }
  actionButtons.forEach((button) => {
    button.addEventListener("click", () => playRound(button.dataset.action));
  });

  loadLatencyPreferences();
  updateChoicePlayed("-", "-");
  fetchAgents().catch((err) => {
    setStatus(`Unable to load agents: ${String(err)}`);
  });
})();
