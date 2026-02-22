(function () {
  "use strict";

  const agentSelect = document.getElementById("agentSelect");
  const newGameBtn = document.getElementById("newGameBtn");
  const gameStatus = document.getElementById("gameStatus");
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

  function setStatus(message) {
    gameStatus.textContent = message;
  }

  function setOutcome(text, kind) {
    outcomeBanner.textContent = text;
    outcomeBanner.classList.remove("win", "loss", "tie");
    if (kind) {
      outcomeBanner.classList.add(kind);
    }
  }

  function animateAction(element, value) {
    element.textContent = value.toUpperCase();
    element.classList.remove("idle");
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

  function addLogRow(round) {
    const row = document.createElement("li");
    row.textContent = `#${round.round_index + 1}: you ${round.player_action_name}, ai ${round.ai_action_name}, outcome ${round.outcome}`;
    roundLog.prepend(row);
  }

  async function fetchAgents() {
    const response = await fetch("/api/v1/agents");
    const body = await response.json();
    agentSelect.innerHTML = "";
    body.agents.forEach((agent) => {
      const option = document.createElement("option");
      option.value = agent.name;
      option.textContent = `${agent.name} - ${agent.type}`;
      agentSelect.appendChild(option);
    });
  }

  async function createGame() {
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
    updateScore(currentGame);
    roundLog.innerHTML = "";
    playerActionEl.textContent = "-";
    aiActionEl.textContent = "-";
    playerActionEl.classList.add("idle");
    aiActionEl.classList.add("idle");
    setOutcome("Game ready. Make your move.", null);
    setStatus(`Game ${currentGame.game_id} using ${currentGame.agent_name}`);
  }

  async function playRound(action) {
    if (!currentGame) {
      setStatus("Create a game first.");
      return;
    }
    const response = await fetch(`/api/v1/games/${currentGame.game_id}/round`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
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
    addLogRow(body.round);
  }

  newGameBtn.addEventListener("click", createGame);
  actionButtons.forEach((button) => {
    button.addEventListener("click", () => playRound(button.dataset.action));
  });

  fetchAgents().catch((err) => {
    setStatus(`Unable to load agents: ${String(err)}`);
  });
})();
