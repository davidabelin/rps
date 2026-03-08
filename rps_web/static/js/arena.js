(function () {
  "use strict";

  const appBasePath = String(window.__APP_BASE_PATH__ || "").replace(/\/+$/, "");
  const apiBase = `${appBasePath}/api/v1`;

  const arenaForm = document.getElementById("arenaForm");
  const agentASelect = document.getElementById("agentASelect");
  const agentBSelect = document.getElementById("agentBSelect");
  const roundsInput = document.getElementById("roundsInput");
  const seedInput = document.getElementById("seedInput");
  const speedSelect = document.getElementById("speedSelect");
  const pauseArenaBtn = document.getElementById("pauseArenaBtn");
  const arenaStatus = document.getElementById("arenaStatus");
  const arenaProgress = document.getElementById("arenaProgress");
  const arenaOutcomeBanner = document.getElementById("arenaOutcomeBanner");
  const arenaAgentAScore = document.getElementById("arenaAgentAScore");
  const arenaAgentBScore = document.getElementById("arenaAgentBScore");
  const arenaTieScore = document.getElementById("arenaTieScore");
  const arenaRoundsShown = document.getElementById("arenaRoundsShown");
  const arenaRoundLog = document.getElementById("arenaRoundLog");
  const arenaMatchesTableBody = document.getElementById("arenaMatchesTableBody");
  const refreshArenaMatchesBtn = document.getElementById("refreshArenaMatchesBtn");
  const agentALabel = document.getElementById("agentALabel");
  const agentBLabel = document.getElementById("agentBLabel");
  const agentAAction = document.getElementById("agentAAction");
  const agentBAction = document.getElementById("agentBAction");
  const agentAHistory = document.getElementById("agentAHistory");
  const agentBHistory = document.getElementById("agentBHistory");
  const arenaChoicePlayed = document.getElementById("arenaChoicePlayed");
  const arenaClashFx = document.getElementById("arenaClashFx");

  let currentMatchId = null;
  let currentTrace = [];
  let renderedCount = 0;
  let playerHistoryTokens = [];
  let aiHistoryTokens = [];
  let playbackTimer = null;
  let paused = false;
  let eventSource = null;

  function setStatus(text) {
    arenaStatus.textContent = text;
  }

  function setProgress(progress) {
    const pct = Math.max(0, Math.min(100, Math.round((progress || 0) * 100)));
    arenaProgress.style.width = `${pct}%`;
  }

  function titleCase(value) {
    const token = String(value || "").trim();
    if (!token) {
      return "-";
    }
    return token[0].toUpperCase() + token.slice(1).toLowerCase();
  }

  function resetChip(node) {
    node.textContent = "-";
    node.classList.add("idle");
    node.classList.remove("reveal");
  }

  function flashChip(node, text) {
    node.classList.remove("idle");
    node.classList.remove("reveal");
    node.textContent = titleCase(text);
    void node.offsetWidth;
    node.classList.add("reveal");
  }

  function renderHistory(target, items) {
    target.innerHTML = "";
    items.slice(0, 8).forEach((item, index) => {
      const chip = document.createElement("div");
      chip.className = "history-icon latest";
      chip.style.opacity = String(index < 4 ? 1 - index * 0.15 : 0.3);
      chip.textContent = titleCase(item);
      target.appendChild(chip);
    });
  }

  function triggerClash() {
    if (!arenaClashFx) {
      return;
    }
    arenaClashFx.classList.remove("active");
    void arenaClashFx.offsetWidth;
    arenaClashFx.classList.add("active");
  }

  function clearPlayback() {
    currentTrace = [];
    renderedCount = 0;
    playerHistoryTokens = [];
    aiHistoryTokens = [];
    arenaRoundLog.innerHTML = "";
    renderHistory(agentAHistory, []);
    renderHistory(agentBHistory, []);
    resetChip(agentAAction);
    resetChip(agentBAction);
    arenaChoicePlayed.textContent = "Agent A: - | Agent B: -";
    arenaOutcomeBanner.textContent = "Start a match to begin playback.";
    arenaAgentAScore.textContent = "0";
    arenaAgentBScore.textContent = "0";
    arenaTieScore.textContent = "0";
    arenaRoundsShown.textContent = "0";
    setProgress(0);
  }

  function stopPlaybackTimer() {
    if (playbackTimer) {
      window.clearTimeout(playbackTimer);
      playbackTimer = null;
    }
  }

  function closeEvents() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  function safeJson(response) {
    return response.json().catch(() => ({}));
  }

  async function fetchAgents() {
    const response = await fetch(`${apiBase}/agents`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to load agents");
    }
    const names = body.agents.map((agent) => agent.name);
    [agentASelect, agentBSelect].forEach((select) => {
      select.innerHTML = "";
      names.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
    });
    if (names.length) {
      agentASelect.value = names[0];
      agentBSelect.value = names[Math.min(1, names.length - 1)];
    }
  }

  function renderFrame(frame) {
    flashChip(agentAAction, frame.agent_a_action_name);
    flashChip(agentBAction, frame.agent_b_action_name);
    triggerClash();
    playerHistoryTokens.unshift(frame.agent_a_action_name);
    aiHistoryTokens.unshift(frame.agent_b_action_name);
    renderHistory(agentAHistory, playerHistoryTokens);
    renderHistory(agentBHistory, aiHistoryTokens);
    arenaChoicePlayed.textContent = `Agent A: ${titleCase(frame.agent_a_action_name)} | Agent B: ${titleCase(frame.agent_b_action_name)}`;
    arenaAgentAScore.textContent = String(frame.score_agent_a);
    arenaAgentBScore.textContent = String(frame.score_agent_b);
    arenaTieScore.textContent = String(frame.score_ties);
    arenaRoundsShown.textContent = String(frame.round_index + 1);
    arenaOutcomeBanner.textContent = frame.winner === "tie"
      ? "Tie round."
      : frame.winner === "agent_a"
        ? "Agent A takes the round."
        : "Agent B takes the round.";
    const item = document.createElement("li");
    item.textContent = `Round ${frame.round_index + 1}: ${titleCase(frame.agent_a_action_name)} vs ${titleCase(frame.agent_b_action_name)} -> ${frame.winner}`;
    arenaRoundLog.prepend(item);
  }

  function schedulePlayback() {
    stopPlaybackTimer();
    if (paused || renderedCount >= currentTrace.length) {
      return;
    }
    playbackTimer = window.setTimeout(() => {
      const frame = currentTrace[renderedCount];
      if (frame) {
        renderFrame(frame);
        renderedCount += 1;
      }
      schedulePlayback();
    }, Number(speedSelect.value || 550));
  }

  function applyMatchPayload(match, resetPlayback) {
    currentMatchId = match.id;
    currentTrace = Array.isArray(match.trace) ? match.trace.slice() : [];
    if (resetPlayback) {
      clearPlayback();
      renderedCount = 0;
    }
    const summary = match.summary || {};
    agentALabel.textContent = summary.agent_a || match.agent_a || "Agent A";
    agentBLabel.textContent = summary.agent_b || match.agent_b || "Agent B";
    setStatus(`Arena match ${match.id}: ${match.status}`);
    setProgress(match.progress);
    if (match.status === "completed" && renderedCount >= currentTrace.length) {
      arenaOutcomeBanner.textContent = `Match winner: ${match.winner || "tie"}`;
    }
    schedulePlayback();
  }

  async function fetchMatches() {
    const response = await fetch(`${apiBase}/arena/matches`);
    const body = await safeJson(response);
    if (!response.ok) {
      throw new Error(body.error || "Failed to load arena matches");
    }
    arenaMatchesTableBody.innerHTML = "";
    body.matches.forEach((match) => {
      const tr = document.createElement("tr");
      const params = match.params || {};
      tr.innerHTML = `
        <td>${match.id}</td>
        <td>${match.status}</td>
        <td>${match.agent_a} vs ${match.agent_b}</td>
        <td>${match.winner || "-"}</td>
        <td>${params.rounds || "-"}</td>
        <td><button class="btn btn-secondary" type="button" data-match-id="${match.id}">Load</button></td>
      `;
      arenaMatchesTableBody.appendChild(tr);
    });
    arenaMatchesTableBody.querySelectorAll("button[data-match-id]").forEach((button) => {
      button.addEventListener("click", () => loadMatch(Number(button.getAttribute("data-match-id"))));
    });
  }

  async function loadMatch(matchId) {
    closeEvents();
    stopPlaybackTimer();
    const response = await fetch(`${apiBase}/arena/matches/${matchId}`);
    const body = await safeJson(response);
    if (!response.ok) {
      setStatus(`Failed to load match ${matchId}: ${body.error || "unknown error"}`);
      return;
    }
    applyMatchPayload(body.match, true);
  }

  function connectEvents(matchId) {
    closeEvents();
    if (!window.EventSource) {
      return;
    }
    eventSource = new EventSource(`${apiBase}/arena/matches/${matchId}/events`);
    eventSource.addEventListener("update", async (event) => {
      const match = JSON.parse(event.data);
      applyMatchPayload(match, false);
      if (match.status === "completed" || match.status === "failed") {
        await fetchMatches();
      }
    });
    eventSource.addEventListener("end", () => {
      closeEvents();
      fetchMatches().catch(() => null);
    });
    eventSource.onerror = () => {
      closeEvents();
    };
  }

  arenaForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearPlayback();
    closeEvents();
    stopPlaybackTimer();
    const payload = {
      agent_a: String(agentASelect.value || ""),
      agent_b: String(agentBSelect.value || ""),
      rounds: Number(roundsInput.value || 50),
      seed: Number(seedInput.value || 7),
    };
    const response = await fetch(`${apiBase}/arena/matches`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await safeJson(response);
    if (!response.ok) {
      setStatus(`Failed to start arena match: ${body.error || "unknown error"}`);
      return;
    }
    paused = false;
    pauseArenaBtn.textContent = "Pause";
    applyMatchPayload(body.match, true);
    connectEvents(body.match.id);
    await fetchMatches();
  });

  pauseArenaBtn.addEventListener("click", () => {
    paused = !paused;
    pauseArenaBtn.textContent = paused ? "Resume" : "Pause";
    if (paused) {
      stopPlaybackTimer();
    } else {
      schedulePlayback();
    }
  });

  refreshArenaMatchesBtn.addEventListener("click", () => {
    fetchMatches().catch((error) => setStatus(`Failed to refresh matches: ${String(error)}`));
  });

  fetchAgents()
    .then(fetchMatches)
    .catch((error) => setStatus(`Failed to initialize arena: ${String(error)}`));
})();
