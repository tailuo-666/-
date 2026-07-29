const state = {
  sessionId: null,
  quotedText: null,
  sending: false,
};

const sessionsNode = document.querySelector("#sessions");
const messagesNode = document.querySelector("#messages");
const sessionTitleNode = document.querySelector("#session-title");
const sessionStateNode = document.querySelector("#session-state");
const promptNode = document.querySelector("#prompt");
const composerNode = document.querySelector("#composer");
const quoteChipNode = document.querySelector("#quote-chip");
const quotePreviewNode = document.querySelector("#quote-preview");
const traceNode = document.querySelector("#trace-content");
const sendNode = document.querySelector("#send");
const messageTemplate = document.querySelector("#message-template");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "请求失败，请稍后重试。");
  }
  return payload;
}

function formatTime(value) {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function setSessionState(status) {
  const running = status === "running" || state.sending;
  sessionStateNode.textContent = running ? "执行中" : "待命";
  sessionStateNode.classList.toggle("is-running", running);
  promptNode.disabled = running;
  sendNode.disabled = running;
}

function clearQuote() {
  state.quotedText = null;
  quoteChipNode.hidden = true;
  quotePreviewNode.textContent = "";
}

function setQuote(text) {
  state.quotedText = text.trim();
  if (!state.quotedText) {
    clearQuote();
    return;
  }
  quotePreviewNode.textContent = state.quotedText.length > 96 ? `${state.quotedText.slice(0, 96)}...` : state.quotedText;
  quoteChipNode.hidden = false;
  promptNode.focus();
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    document.querySelector("#model-status").textContent = health.model_configured ? "模型已配置" : "缺少模型密钥";
    document.querySelector("#model-dot").classList.toggle("is-ready", health.model_configured);
  } catch (_) {
    document.querySelector("#model-status").textContent = "服务未连接";
  }
}

async function loadSessions(selectFirst = false) {
  const sessions = await api("/api/sessions");
  sessionsNode.replaceChildren();
  for (const session of sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.classList.toggle("is-active", session.id === state.sessionId);
    button.innerHTML = `<span class="session-title"></span><span class="session-time"></span>`;
    button.querySelector(".session-title").textContent = session.title;
    button.querySelector(".session-time").textContent = formatTime(session.updated_at);
    button.addEventListener("click", () => selectSession(session.id));
    sessionsNode.append(button);
  }
  if (selectFirst && sessions.length) {
    await selectSession(sessions[0].id);
  }
}

function renderMessages(messages) {
  messagesNode.replaceChildren();
  if (!messages.length) {
    messagesNode.innerHTML = `
      <div class="empty-state">
        <span class="empty-index">01</span>
        <h3>从一个问题开始</h3>
        <p>模型会在需要时调用本地工具，并将结果作为下一步观察。</p>
      </div>`;
    return;
  }
  for (const message of messages) {
    const fragment = messageTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".message");
    const role = fragment.querySelector(".message-role");
    const time = fragment.querySelector("time");
    const content = fragment.querySelector(".message-content");
    const traceButton = fragment.querySelector(".view-trace");
    article.classList.add(`is-${message.role}`);
    if (message.status === "failed") article.classList.add("is-failed");
    if (message.status === "running") article.classList.add("is-running");
    role.textContent = message.role === "user" ? "你" : "AGENT";
    time.textContent = formatTime(message.created_at);
    content.textContent = message.content || "正在思考与调用工具...";
    if (message.quoted_text) {
      const quote = document.createElement("blockquote");
      quote.textContent = message.quoted_text;
      content.prepend(quote);
    }
    if (message.role === "assistant" && message.trace_id) {
      traceButton.addEventListener("click", () => showTrace(message.trace_id));
    } else {
      traceButton.remove();
    }
    if (message.role === "assistant") {
      content.addEventListener("mouseup", () => {
        const selection = window.getSelection();
        const text = selection ? selection.toString().trim() : "";
        if (text) setQuote(text);
      });
    }
    messagesNode.append(fragment);
  }
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

async function selectSession(sessionId) {
  state.sessionId = sessionId;
  clearQuote();
  const detail = await api(`/api/sessions/${sessionId}`);
  sessionTitleNode.textContent = detail.title;
  setSessionState(detail.status);
  renderMessages(detail.messages);
  await loadSessions();
  if (detail.pending_message_id) {
    state.sending = true;
    setSessionState("running");
    try {
      await api(`/api/sessions/${sessionId}/resume`, { method: "POST" });
    } catch (error) {
      renderInlineError(error.message);
    } finally {
      state.sending = false;
      await selectSession(sessionId);
    }
  }
}

function renderInlineError(message) {
  const node = document.createElement("p");
  node.className = "inline-error";
  node.textContent = message;
  messagesNode.append(node);
}

async function showTrace(traceId) {
  traceNode.replaceChildren();
  const trace = await api(`/api/traces/${traceId}`);
  if (!trace.tool_spans.length) {
    traceNode.innerHTML = '<p class="trace-empty">本轮没有调用工具。</p>';
    return;
  }
  for (const span of trace.tool_spans) {
    const details = document.createElement("details");
    details.className = "trace-item";
    details.open = true;
    const summary = document.createElement("summary");
    summary.innerHTML = `<span>${span.name}</span><span class="span-status">${span.status === "completed" ? "完成" : "失败"}</span>`;
    const data = document.createElement("pre");
    data.textContent = JSON.stringify(
      span.error ? { error: span.error } : { input: span.input, output: span.output },
      null,
      2,
    );
    details.append(summary, data);
    traceNode.append(details);
  }
}

async function createSession() {
  const session = await api("/api/sessions", { method: "POST", body: JSON.stringify({}) });
  await loadSessions();
  await selectSession(session.id);
}

async function sendMessage(event) {
  event.preventDefault();
  const content = promptNode.value.trim();
  if (!content || !state.sessionId || state.sending) return;
  state.sending = true;
  setSessionState("running");
  const quotedText = state.quotedText;
  promptNode.value = "";
  clearQuote();
  try {
    await api(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content, quoted_text: quotedText }),
    });
    await selectSession(state.sessionId);
  } catch (error) {
    renderInlineError(error.message);
  } finally {
    state.sending = false;
    setSessionState("idle");
    promptNode.focus();
  }
}

document.querySelector("#new-session").addEventListener("click", createSession);
document.querySelector("#clear-quote").addEventListener("click", clearQuote);
composerNode.addEventListener("submit", sendMessage);
promptNode.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerNode.requestSubmit();
  }
});

(async function init() {
  await loadHealth();
  const sessions = await api("/api/sessions");
  if (sessions.length) {
    await loadSessions(true);
  } else {
    await createSession();
  }
})().catch((error) => renderInlineError(error.message));
