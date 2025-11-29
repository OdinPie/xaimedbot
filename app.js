const API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000/api"
    : "/api";


const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const inputLoader = document.getElementById("input-loader");
const globalLoader = document.getElementById("global-loader");

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const waitForBackendReady = async () => {
  const maxAttempts = 60;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      const res = await fetch(`${API_BASE}/ready`);
      if (!res.ok) throw new Error("not ready");
      const data = await res.json();
      if (data.ready) return true;
    } catch (error) {
      // keep waiting
    }
    await wait(1000);
  }
  return false;
};
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const statusLabel = document.getElementById("status-label");
const conditionsList = document.getElementById("conditions-list");
const notesPanel = document.getElementById("session-notes");
const finalSummaryModal = document.getElementById("final-summary-modal");
const finalSummaryContent = document.getElementById("final-summary-content");
const finalCloseBtn = document.getElementById("final-close-btn");
const quickAnswers = document.getElementById("quick-answers");
const quickYesBtn = document.getElementById("quick-yes");
const quickNoBtn = document.getElementById("quick-no");

let sessionId = null;
let sending = false;
let hasUserSentFirstMessage = false;


const sanitize = (text) => {
  if (!text) return "";
  // escape HTML first
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
  // then apply simple markdown-style bold (**text**)
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return safe;
};



const renderMessage = ({ role, content }) => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = sanitize(content);
  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
};

const updateConditions = (conditions = []) => {
  conditionsList.innerHTML = "";
  if (!conditions.length) {
    conditionsList.innerHTML = `<li>No ranked conditions yet.</li>`;
    return;
  }

  conditions.forEach((condition) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <div class="condition-head">
        <strong>${condition.name}</strong>
        <span class="confidence">${condition.confidence}% confidence</span>
      </div>
      <span class="meta">Matched: ${
        condition.matched_symptoms?.length
          ? condition.matched_symptoms.join(", ")
          : "gathering data"
      }</span>
      <span class="meta negative">Denied: ${
        condition.denied_symptoms?.length
          ? condition.denied_symptoms.join(", ")
          : "none"
      } • Penalty −${condition.penalty?.toFixed?.(2) ?? "0.00"}</span>
    `;
    conditionsList.appendChild(item);
  });
};

const updateStatus = (stage) => {
  const mapping = {
    collecting_symptoms: "Collecting symptoms",
    awaiting_phase2_consent: "Awaiting consent for diagnostic tests",
    collecting_tests: "Discussing diagnostic tests",
    completed: "Consultation complete",
  };
  statusLabel.textContent = mapping[stage] || "In session";
};

const setQuickVisible = (visible) => {
  if (!quickAnswers) return;
  if (visible) quickAnswers.classList.remove("hidden");
  else quickAnswers.classList.add("hidden");
};

const toggleSending = (state) => {
  sending = state;
  sendBtn.disabled = state;
  chatInput.disabled = state;

  // While sending, hide quick answers so the user can't spam Yes/No
  if (state) {
    setQuickVisible(false);
  } else {
    // After sending finishes, only show buttons if user has already sent first text
    if (hasUserSentFirstMessage) {
      setQuickVisible(true);
    } else {
      setQuickVisible(false);
    }
  }
};



const updateFinalSummary = (summary) => {
  if (!summary) {
    finalSummaryModal.classList.add("hidden");
    finalSummaryModal.setAttribute("aria-hidden", "true");
    finalSummaryContent.textContent = "";
    return;
  }
  finalSummaryContent.innerHTML = sanitize(summary);
  finalSummaryModal.classList.remove("hidden");
  finalSummaryModal.setAttribute("aria-hidden", "false");
};

const displayNotes = (messages) => {
  if (!messages?.length) return;
  const lastAssistant = messages.filter((msg) => msg.role === "assistant").pop();
  if (lastAssistant?.content.includes("Medical")) return;
  notesPanel.textContent = lastAssistant.content.replace(/\n+/g, " ");
};

const startSession = async () => {
  hasUserSentFirstMessage = false;
  setQuickVisible(false);
  toggleSending(true);
  chatLog.innerHTML = "";
  updateFinalSummary(null);
  try {
    const res = await fetch(`${API_BASE}/session`, { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
    data.messages.forEach(renderMessage);
    updateConditions(data.top_conditions || []);
    updateStatus(data.stage);
    notesPanel.textContent =
      "Share your symptoms to receive contextual questions, evidence-based reasoning, and guidance on what to discuss with your clinician.";
    globalLoader.classList.add("hidden");
  } catch (error) {
    console.error(error);
    renderMessage({
      role: "assistant",
      content: "Unable to start the session. Please ensure the backend server is running.",
    });
  } finally {
    toggleSending(false);
    chatInput.focus();
  }
};

const initializeApp = async () => {
  const ready = await waitForBackendReady();
  if (ready) {
    globalLoader.classList.add("hidden");
    startSession();
  } else {
    globalLoader.querySelector("p").textContent =
      "Still preparing the models… please keep this window open.";
  }
};

const setTyping = (active) => {
  if (active) {
    inputLoader.style.display = "block";
  } else {
    inputLoader.style.display = "none";
  }
};

const sendMessage = async (message) => {
  if (!message.trim() || sending || !sessionId) return;
  if (!hasUserSentFirstMessage) {
    hasUserSentFirstMessage = true;
  }

  renderMessage({ role: "user", content: message });
  chatInput.value = "";
  toggleSending(true);
  setTyping(true);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    if (!res.ok) {
      throw new Error("Server error");
    }
    const data = await res.json();
    data.messages.forEach(renderMessage);
    updateConditions(data.top_conditions || []);
    updateStatus(data.stage);
    updateFinalSummary(data.final_summary);
    displayNotes(data.messages);
  } catch (error) {
    console.error(error);
    renderMessage({
      role: "assistant",
      content: "Something went wrong while sending your message. Please try again.",
    });
  } finally {
    toggleSending(false);
    setTyping(false);
    chatInput.focus();
  }
};

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(chatInput.value);
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(chatInput.value);
  }
});

resetBtn.addEventListener("click", () => {
  startSession();
});

if (finalCloseBtn) {
  finalCloseBtn.addEventListener("click", () => {
    updateFinalSummary(null);
  });
}

if (quickYesBtn && quickNoBtn) {
  quickYesBtn.addEventListener("click", () => {
    if (!sending) sendMessage("Yes");
  });
  quickNoBtn.addEventListener("click", () => {
    if (!sending) sendMessage("No");
  });
}

initializeApp();

