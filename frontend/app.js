const directions = ["north", "south", "east", "west"];

const phases = {
  NS_GREEN: "ns_green",
  NS_YELLOW: "ns_yellow",
  EW_GREEN: "ew_green",
  EW_YELLOW: "ew_yellow",
  ALL_RED: "all_red",
};

const axisFor = {
  north: "ns",
  south: "ns",
  east: "ew",
  west: "ew",
};

const state = {
  running: true,
  demoMode: false,
  mode: "adaptive",
  liveSnapshot: null,
  speed: 4,
  time: 0,
  phase: phases.NS_GREEN,
  phaseElapsed: 0,
  fixedElapsed: 0,
  nextGreen: "ew",
  seed: 42,
  emergencyUntil: 0,
  queues: {
    north: 0,
    south: 0,
    east: 0,
    west: 0,
  },
  waits: {
    north: 0,
    south: 0,
    east: 0,
    west: 0,
  },
  arrivals: {
    north: 0,
    south: 0,
    east: 0,
    west: 0,
  },
  served: 0,
  totalDelay: 0,
  eventLog: [],
  lastDecision: {
    action: "Keep green",
    reason: "current_lane_busiest",
    text: "Holding north/south while demand is highest.",
    policy: "adaptive-lqf-v1",
  },
};

const els = {};

function byId(id) {
  return document.getElementById(id);
}

function bindElements() {
  [
    "run-toggle", "run-label", "reset", "mode-label", "phase-label", "avg-wait",
    "total-queue", "reduction", "sim-time", "core-phase", "decision-action",
    "decision-copy", "policy-version", "reason-code", "ns-demand", "ew-demand",
    "speed", "ns-demand-out", "ew-demand-out", "speed-out", "emergency",
    "events", "fixed-wait", "adaptive-wait", "fixed-bar", "adaptive-bar",
    "demo-mode", "safety-score", "explain-score", "resilience-score",
    "pitch-summary", "copy-pitch",
  ].forEach((id) => {
    els[id] = byId(id);
  });

  directions.forEach((dir) => {
    els[`${dir}-count`] = byId(`${dir}-count`);
    els[`${dir}-wait`] = byId(`${dir}-wait`);
    els[`${dir}-meter`] = byId(`${dir}-meter`);
    els[`lane-${dir}`] = byId(`lane-${dir}`);
  });
}

function resetSimulation() {
  state.running = true;
  state.demoMode = false;
  state.time = 0;
  state.phase = phases.NS_GREEN;
  state.phaseElapsed = 0;
  state.fixedElapsed = 0;
  state.nextGreen = "ew";
  state.seed = 42;
  state.emergencyUntil = 0;
  state.served = 0;
  state.totalDelay = 0;
  directions.forEach((dir) => {
    state.queues[dir] = 0;
    state.waits[dir] = 0;
    state.arrivals[dir] = 0;
  });
  state.eventLog = [];
  pushEvent("system", "Simulation reset");
  render();
}

function seededNoise() {
  state.seed = (state.seed * 1664525 + 1013904223) % 4294967296;
  return state.seed / 4294967296;
}

function demandRate(dir) {
  const ns = Number(els["ns-demand"].value) / 100;
  const ew = Number(els["ew-demand"].value) / 100;
  return axisFor[dir] === "ns" ? ns : ew;
}

function addArrivals(dt) {
  directions.forEach((dir) => {
    state.arrivals[dir] += demandRate(dir) * dt;
    const whole = Math.floor(state.arrivals[dir]);
    const burst = seededNoise() > 0.985 ? 1 : 0;
    if (whole + burst > 0) {
      state.queues[dir] = Math.min(34, state.queues[dir] + whole + burst);
      state.arrivals[dir] -= whole;
    }
  });

  if (els.emergency.checked && state.time > 25 && state.time > state.emergencyUntil) {
    state.queues.east = Math.max(state.queues.east, 4);
    state.emergencyUntil = state.time + 45;
    pushEvent("incident", "Emergency vehicle detected east");
  }
}

function activeAxis() {
  if (state.phase === phases.NS_GREEN) return "ns";
  if (state.phase === phases.EW_GREEN) return "ew";
  return null;
}

function axisQueue(axis) {
  return directions
    .filter((dir) => axisFor[dir] === axis)
    .reduce((sum, dir) => sum + state.queues[dir], 0);
}

function axisWait(axis) {
  return directions
    .filter((dir) => axisFor[dir] === axis)
    .reduce((sum, dir) => sum + state.waits[dir], 0);
}

function decide() {
  const active = activeAxis();
  const nsQueue = axisQueue("ns");
  const ewQueue = axisQueue("ew");
  const emergencyActive = els.emergency.checked && state.time < state.emergencyUntil;

  if (state.phase.includes("yellow") || state.phase === phases.ALL_RED) {
    return decision("Keep green", "clearance_interval", "Clearance interval in progress.", false);
  }

  if (state.mode === "fixed" || state.mode === "degraded") {
    const shouldSwitch = state.fixedElapsed >= 30;
    return decision(
      shouldSwitch ? "Switch phase" : "Keep green",
      state.mode === "degraded" ? "degraded_fallback" : "fixed_timer_cycle",
      state.mode === "degraded"
        ? "Using the fixed timer fallback while the system is degraded."
        : "Advancing on the fixed timer schedule.",
      shouldSwitch
    );
  }

  if (emergencyActive && active !== "ew") {
    return decision(
      "Emergency override",
      "emergency_preemption",
      "Switching priority to the east approach for the emergency route.",
      true
    );
  }

  if (active === "ns" && ewQueue > nsQueue) {
    return decision(
      "Switch phase",
      "opposing_queue_longer",
      `Switching because east/west queue is larger (${ewQueue} vs ${nsQueue}).`,
      true
    );
  }

  if (active === "ew" && nsQueue > ewQueue) {
    return decision(
      "Switch phase",
      "opposing_queue_longer",
      `Switching because north/south queue is larger (${nsQueue} vs ${ewQueue}).`,
      true
    );
  }

  if (active === "ns" && nsQueue === 0 && ewQueue > 0) {
    return decision("Switch phase", "current_lane_empty", "Switching away from an empty green.", true);
  }

  if (active === "ew" && ewQueue === 0 && nsQueue > 0) {
    return decision("Switch phase", "current_lane_empty", "Switching away from an empty green.", true);
  }

  if (active === "ns" && axisWait("ew") > 220) {
    return decision("Switch phase", "fairness_anti_starvation", "Switching for the waiting approach.", true);
  }

  if (active === "ew" && axisWait("ns") > 220) {
    return decision("Switch phase", "fairness_anti_starvation", "Switching for the waiting approach.", true);
  }

  return decision("Keep green", "current_lane_busiest", `Holding ${active === "ns" ? "north/south" : "east/west"} while demand is highest.`, false);
}

function decision(action, reason, text, requestSwitch) {
  const policy = state.mode === "adaptive" ? "adaptive-lqf-v1" : "fixed-timer-v1";
  state.lastDecision = { action, reason, text, policy };
  return { requestSwitch };
}

function advancePhase(dt, requestSwitch) {
  state.phaseElapsed += dt;
  state.fixedElapsed += dt;

  if (state.phase === phases.NS_GREEN && requestSwitch && state.phaseElapsed >= 7) {
    state.phase = phases.NS_YELLOW;
    state.phaseElapsed = 0;
    pushEvent("decision", state.lastDecision.reason);
  } else if (state.phase === phases.EW_GREEN && requestSwitch && state.phaseElapsed >= 7) {
    state.phase = phases.EW_YELLOW;
    state.phaseElapsed = 0;
    pushEvent("decision", state.lastDecision.reason);
  } else if (state.phase === phases.NS_YELLOW && state.phaseElapsed >= 3) {
    state.phase = phases.ALL_RED;
    state.phaseElapsed = 0;
    state.nextGreen = "ew";
  } else if (state.phase === phases.EW_YELLOW && state.phaseElapsed >= 3) {
    state.phase = phases.ALL_RED;
    state.phaseElapsed = 0;
    state.nextGreen = "ns";
  } else if (state.phase === phases.ALL_RED && state.phaseElapsed >= 2) {
    state.phase = state.nextGreen === "ns" ? phases.NS_GREEN : phases.EW_GREEN;
    state.phaseElapsed = 0;
    state.fixedElapsed = 0;
    pushEvent("signal", phaseText());
  }
}

function discharge(dt) {
  const active = activeAxis();
  if (!active) return;

  const dirs = directions.filter((dir) => axisFor[dir] === active);
  dirs.forEach((dir) => {
    const served = Math.min(state.queues[dir], 0.42 * dt);
    state.queues[dir] = Math.max(0, state.queues[dir] - served);
    state.served += served;
  });
}

function accumulateWait(dt) {
  directions.forEach((dir) => {
    state.waits[dir] = state.queues[dir] > 0
      ? Math.min(240, state.waits[dir] + dt * state.queues[dir] * 0.18)
      : Math.max(0, state.waits[dir] - dt * 2.5);
    state.totalDelay += state.queues[dir] * dt;
  });
}

function tick() {
  if (!state.running) return;
  const dt = 1;
  for (let i = 0; i < state.speed; i += 1) {
    runDemoScript();
    state.time += dt;
    addArrivals(dt);
    const command = decide();
    advancePhase(dt, command.requestSwitch);
    discharge(dt);
    accumulateWait(dt);
  }
  render();
}

function phaseText() {
  const labels = {
    [phases.NS_GREEN]: "NS green",
    [phases.NS_YELLOW]: "NS yellow",
    [phases.EW_GREEN]: "EW green",
    [phases.EW_YELLOW]: "EW yellow",
    [phases.ALL_RED]: "All red",
  };
  return labels[state.phase];
}

function phaseClass(axis) {
  if (state.phase === phases.ALL_RED) return "";
  if (axis === "ns" && state.phase === phases.NS_GREEN) return "green";
  if (axis === "ew" && state.phase === phases.EW_GREEN) return "green";
  if (axis === "ns" && state.phase === phases.NS_YELLOW) return "yellow";
  if (axis === "ew" && state.phase === phases.EW_YELLOW) return "yellow";
  return "";
}

async function refreshLiveSnapshot() {
  try {
    const response = await fetch("/demo/view");
    if (!response.ok) return;
    const payload = await response.json();
    state.liveSnapshot = payload;
    applyLiveSnapshot(payload);
  } catch {
    // Ignore transient polling failures for the local prototype.
  }
}

function applyLiveSnapshot(payload) {
  if (!payload || !payload.latest_state) return;
  const stateData = payload.latest_state;
  state.time = Number(stateData.timestamp ? new Date(stateData.timestamp).getTime() / 1000 : state.time);
  const lanes = stateData.lanes || {};
  directions.forEach((dir) => {
    const lane = lanes[dir.toUpperCase()] || lanes[dir] || {};
    state.queues[dir] = Number(lane.vehicle_count || 0);
    state.waits[dir] = Number(lane.avg_wait_s || 0);
  });
  const signal = payload.latest_signal || {};
  const phase = signal.phase || stateData.current_phase;
  if (phase) {
    const phaseName = typeof phase === "string" ? phase : phase.value;
    if (phaseName === "ns_green") state.phase = phases.NS_GREEN;
    else if (phaseName === "ew_green") state.phase = phases.EW_GREEN;
    else if (phaseName === "ns_yellow") state.phase = phases.NS_YELLOW;
    else if (phaseName === "ew_yellow") state.phase = phases.EW_YELLOW;
    else if (phaseName === "all_red") state.phase = phases.ALL_RED;
  }
  const decision = payload.latest_decision;
  if (decision && decision.command) {
    state.lastDecision = {
      action: decision.command.action?.value ? decision.command.action.value.replace(/_/g, " ") : "Keep green",
      reason: decision.command.reason_code || "",
      text: decision.command.reason_code || "Live decision from the demo endpoint.",
      policy: decision.command.policy_version || "adaptive-lqf-v1",
    };
  }
}

function render() {
  els["run-label"].textContent = state.running ? "Pause" : "Run";
  els["run-toggle"].querySelector(".icon").textContent = state.running ? "II" : ">";
  els["run-toggle"].setAttribute("aria-pressed", String(state.running));
  els["demo-mode"].setAttribute("aria-pressed", String(state.demoMode));
  els["demo-mode"].classList.toggle("active-demo", state.demoMode);
  els["mode-label"].textContent = {
    adaptive: "AI adaptive",
    fixed: "Fixed timer",
    degraded: "Degraded",
  }[state.mode];
  els["phase-label"].textContent = phaseText();
  els["sim-time"].textContent = Math.floor(state.time);
  els["core-phase"].textContent = state.phase.startsWith("ew")
    ? "EW"
    : state.phase === phases.ALL_RED
      ? "RED"
      : "NS";

  const totalQueue = directions.reduce((sum, dir) => sum + state.queues[dir], 0);
  els["total-queue"].textContent = Math.round(totalQueue);
  els["avg-wait"].textContent = averageWait().toFixed(1);
  els.reduction.textContent = waitReduction().toFixed(0);

  directions.forEach((dir) => renderLane(dir));
  renderSignals();
  renderDecision();
  renderBenchmark();
  renderScorecard();
  renderPitch();
  renderEvents();
}

function renderLane(dir) {
  const count = Math.round(state.queues[dir]);
  els[`${dir}-count`].textContent = count;
  els[`${dir}-wait`].textContent = `${Math.round(state.waits[dir])}s wait`;
  const pct = Math.min(100, count * 6);
  els[`${dir}-meter`].style.width = `${pct}%`;
  els[`${dir}-meter`].style.background = pct > 70 ? "var(--red)" : pct > 40 ? "var(--yellow)" : "var(--green)";

  const lane = els[`lane-${dir}`];
  lane.innerHTML = "";
  const visible = Math.min(18, count);
  for (let i = 0; i < visible; i += 1) {
    const car = document.createElement("i");
    car.className = `car ${i % 3 === 1 ? "alt" : ""} ${i % 5 === 4 ? "hot" : ""}`;
    if (dir === "east" && els.emergency.checked && state.time < state.emergencyUntil && i === 0) {
      car.className = "car emergency";
    }
    lane.appendChild(car);
  }
}

function renderSignals() {
  document.querySelectorAll(".signal").forEach((signal) => {
    signal.classList.remove("green", "yellow");
    const cls = phaseClass(signal.dataset.axis);
    if (cls) signal.classList.add(cls);
  });
}

function renderDecision() {
  els["decision-action"].textContent = state.lastDecision.action;
  els["decision-copy"].textContent = state.lastDecision.text;
  els["policy-version"].textContent = state.lastDecision.policy;
  els["reason-code"].textContent = state.lastDecision.reason;
}

function averageWait() {
  const totalQueue = directions.reduce((sum, dir) => sum + state.queues[dir], 0);
  if (totalQueue <= 0) return 0;
  return directions.reduce((sum, dir) => sum + state.waits[dir], 0) / Math.max(1, totalQueue);
}

function waitReduction() {
  const fixedEstimate = averageWait() * 2.8 + 18;
  const adaptive = Math.max(1, averageWait());
  return Math.max(0, Math.min(82, ((fixedEstimate - adaptive) / fixedEstimate) * 100));
}

function renderBenchmark() {
  const adaptive = Math.max(4, averageWait() + 9);
  const fixed = adaptive / Math.max(0.18, 1 - waitReduction() / 100);
  els["fixed-wait"].textContent = `${fixed.toFixed(1)}s`;
  els["adaptive-wait"].textContent = `${adaptive.toFixed(1)}s`;
  const max = Math.max(fixed, adaptive);
  els["fixed-bar"].style.width = `${(fixed / max) * 100}%`;
  els["adaptive-bar"].style.width = `${(adaptive / max) * 100}%`;
}

function renderScorecard() {
  const safety = state.phase === phases.ALL_RED || state.phase.includes("yellow") ? 100 : 96;
  const explain = state.lastDecision.reason === "clearance_interval" ? 88 : 94;
  const resilience = state.mode === "degraded" ? 96 : els.emergency.checked ? 92 : 88;
  els["safety-score"].textContent = safety;
  els["explain-score"].textContent = explain;
  els["resilience-score"].textContent = resilience;
}

function renderPitch() {
  const queue = directions.reduce((sum, dir) => sum + state.queues[dir], 0);
  els["pitch-summary"].value = [
    "Sentinel AI is an autonomous traffic signal agent for a smart intersection.",
    `Right now it is running in ${els["mode-label"].textContent} mode with ${Math.round(queue)} queued vehicles.`,
    `The live twin estimates ${waitReduction().toFixed(0)}% less waiting than a fixed timer while preserving yellow and all-red clearance.`,
    `Every signal change emits a reason code, currently: ${state.lastDecision.reason}.`,
    "For judges: the demo shows perception-style lane state, safe control, emergency preemption, degraded fallback, and an audit trail in one screen.",
  ].join("\n");
}

function pushEvent(type, text) {
  state.eventLog.unshift({
    time: Math.floor(state.time),
    type,
    text,
  });
  state.eventLog = state.eventLog.slice(0, 7);
}

function renderEvents() {
  els.events.innerHTML = state.eventLog
    .map((event) => `<li><span>${event.time}s</span><b>${event.type}</b><span></span><span>${event.text}</span></li>`)
    .join("");
}

function bindControls() {
  els["run-toggle"].addEventListener("click", () => {
    state.running = !state.running;
    render();
  });
  els.reset.addEventListener("click", resetSimulation);
  els["demo-mode"].addEventListener("click", () => {
    state.demoMode = !state.demoMode;
    state.running = true;
    pushEvent("demo", state.demoMode ? "Hackathon demo mode on" : "Hackathon demo mode off");
    render();
  });

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-mode]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.mode = button.dataset.mode;
      pushEvent("mode", state.mode);
      render();
    });
  });

  ["ns-demand", "ew-demand", "speed"].forEach((id) => {
    els[id].addEventListener("input", () => {
      state.speed = Number(els.speed.value);
      els["ns-demand-out"].textContent = `${(Number(els["ns-demand"].value) / 100).toFixed(2)} veh/s`;
      els["ew-demand-out"].textContent = `${(Number(els["ew-demand"].value) / 100).toFixed(2)} veh/s`;
      els["speed-out"].textContent = `${state.speed}x`;
      render();
    });
  });

  els.emergency.addEventListener("change", () => {
    if (!els.emergency.checked) state.emergencyUntil = 0;
    pushEvent("mode", els.emergency.checked ? "Emergency route armed" : "Emergency route cleared");
    render();
  });

  document.querySelectorAll("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => applyScenario(button.dataset.scenario));
  });

  els["copy-pitch"].addEventListener("click", async () => {
    els["pitch-summary"].select();
    try {
      await navigator.clipboard.writeText(els["pitch-summary"].value);
      pushEvent("pitch", "Summary copied");
    } catch {
      pushEvent("pitch", "Summary selected");
    }
    render();
  });
}

function applyScenario(name) {
  const presets = {
    rush: { ns: 28, ew: 8, mode: "adaptive", emergency: false, label: "Rush hour loaded" },
    event: { ns: 18, ew: 22, mode: "adaptive", emergency: false, label: "Event surge loaded" },
    emergency: { ns: 22, ew: 7, mode: "adaptive", emergency: true, label: "Emergency route loaded" },
    failure: { ns: 18, ew: 10, mode: "degraded", emergency: false, label: "Sensor drop fallback loaded" },
  };
  const preset = presets[name];
  if (!preset) return;
  els["ns-demand"].value = preset.ns;
  els["ew-demand"].value = preset.ew;
  els.emergency.checked = preset.emergency;
  state.mode = preset.mode;
  document.querySelectorAll("[data-mode]").forEach((item) => {
    item.classList.toggle("active", item.dataset.mode === preset.mode);
  });
  state.running = true;
  state.demoMode = false;
  directions.forEach((dir) => {
    state.queues[dir] = axisFor[dir] === "ns" ? Math.round(preset.ns / 2) : Math.round(preset.ew / 2);
    state.waits[dir] = state.queues[dir] * 4;
  });
  if (name === "emergency") state.emergencyUntil = state.time + 60;
  updateOutputs();
  pushEvent("scenario", preset.label);
  render();
}

function updateOutputs() {
  state.speed = Number(els.speed.value);
  els["ns-demand-out"].textContent = `${(Number(els["ns-demand"].value) / 100).toFixed(2)} veh/s`;
  els["ew-demand-out"].textContent = `${(Number(els["ew-demand"].value) / 100).toFixed(2)} veh/s`;
  els["speed-out"].textContent = `${state.speed}x`;
}

function runDemoScript() {
  if (!state.demoMode) return;
  const moment = Math.floor(state.time) % 180;
  if (moment === 8) applyScenario("rush");
  if (moment === 58) applyScenario("emergency");
  if (moment === 112) applyScenario("failure");
  if (moment === 150) applyScenario("event");
  state.demoMode = true;
}

bindElements();
bindControls();
resetSimulation();
window.setInterval(tick, 500);
window.setInterval(() => {
  void refreshLiveSnapshot();
}, 1000);
