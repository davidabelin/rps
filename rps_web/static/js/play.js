(function () {
  "use strict";

  const agentSelect = document.getElementById("agentSelect");
  const agentDetails = document.getElementById("agentDetails");
  const newGameBtn = document.getElementById("newGameBtn");
  const gameStatus = document.getElementById("gameStatus");
  const latencyStatus = document.getElementById("latencyStatus");
  const momentumStatus = document.getElementById("momentumStatus");
  const playerActionEl = document.getElementById("playerAction");
  const aiActionEl = document.getElementById("aiAction");
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

  function setStatus(message) {
    gameStatus.textContent = message;
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

  function animateAction(element, value) {
    element.textContent = value.toUpperCase();
    element.classList.remove("idle");
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
    momentumStatus.textContent = `Momentum: AI is on a ${currentStreakCount}-round streak.`;
  }

  function addLogRow(round) {
    const row = document.createElement("li");
    row.textContent = `#${round.round_index + 1}: you ${round.player_action_name}, ai ${round.ai_action_name}, outcome ${round.outcome}`;
    roundLog.prepend(row);
  }

  function updateAgentDetails() {
    const selected = String(agentSelect.value || "");
    const descriptor = agentsByName[selected];
    if (!descriptor) {
      agentDetails.textContent = "Choose an agent to view strategy details.";
      return;
    }
    if (selected === "active_model") {
      agentDetails.textContent = `${descriptor.description} Active model: ${activeModelSummary}.`;
      return;
    }
    agentDetails.textContent = `${descriptor.description}`;
  }

  async function fetchActiveModelSummary() {
    try {
      const response = await fetch("/api/v1/models");
      const body = await response.json();
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
    const response = await fetch("/api/v1/agents");
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || "Failed to fetch agents");
    }
    agentsByName = {};
    agentSelect.innerHTML = "";
    body.agents.forEach((agent) => {
      agentsByName[agent.name] = agent;
      const option = document.createElement("option");
      option.value = agent.name;
      option.textContent = `${agent.name} - ${agent.type}`;
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
      const response = await fetch("/api/v1/games", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent: agentSelect.value }),
      });
      const body = await response.json();
      if (!response.ok) {
        setStatus(`Failed to create game: ${body.error || "unknown error"}`);
        return;
      }
      currentGame = body.game;
      currentStreakSign = 0;
      currentStreakCount = 0;
      updateScore(currentGame);
      roundLog.innerHTML = "";
      playerActionEl.textContent = "-";
      aiActionEl.textContent = "-";
      playerActionEl.classList.add("idle");
      aiActionEl.classList.add("idle");
      playerActionEl.classList.remove("pending");
      aiActionEl.classList.remove("pending");
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
    setOutcome("AI thinking...", null);
    outcomeBanner.classList.add("pending");
    animateAction(playerActionEl, action);
    aiActionEl.textContent = "...";
    aiActionEl.classList.remove("idle");
    aiActionEl.classList.add("pending");
    const startedAt = window.performance.now();
    try {
      const response = await fetch(`/api/v1/games/${currentGame.game_id}/round`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const elapsedMs = Math.round(window.performance.now() - startedAt);
      latencyStatus.textContent = `Last round latency: ${elapsedMs} ms`;
      const body = await response.json();
      if (!response.ok) {
        setStatus(`Round error: ${body.error || "unknown error"}`);
        return;
      }
      currentGame = body.game;
      updateScore(currentGame);
      animateAction(playerActionEl, body.round.player_action_name);
      animateAction(aiActionEl, body.round.ai_action_name);
      if (body.round.outcome === "player") {
        setOutcome("You win this round.", "win");
      } else if (body.round.outcome === "ai") {
        setOutcome("AI wins this round.", "loss");
      } else {
        setOutcome("Tie round.", "tie");
      }
      updateMomentum(body.round.outcome);
      addLogRow(body.round);
    } catch (err) {
      setStatus(`Round error: ${String(err)}`);
      setOutcome("Round failed. Try again.", null);
      playerActionEl.classList.remove("pending");
      aiActionEl.classList.remove("pending");
    } finally {
      setRoundInteractionEnabled(true);
    }
  }

  newGameBtn.addEventListener("click", createGame);
  agentSelect.addEventListener("change", updateAgentDetails);
  actionButtons.forEach((button) => {
    button.addEventListener("click", () => playRound(button.dataset.action));
  });

  fetchAgents().catch((err) => {
    setStatus(`Unable to load agents: ${String(err)}`);
  });
})();
