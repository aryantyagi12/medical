/* ─── MedAI Frontend Logic ─────────────────────────────────────────────── */

const API_BASE = '';  // same origin — FastAPI serves both frontend and API

// ── DOM refs ────────────────────────────────────────────────────────────────
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen      = document.getElementById('welcomeScreen');
const questionInput      = document.getElementById('questionInput');
const sendBtn            = document.getElementById('sendBtn');
const chatHistory        = document.getElementById('chatHistory');
const statusPill         = document.getElementById('statusPill');
const statusText         = document.getElementById('statusText');
const newChatBtn         = document.getElementById('newChatBtn');
const sidebarToggle      = document.getElementById('sidebarToggle');
const mobileSidebarBtn   = document.getElementById('mobileSidebarBtn');
const sidebar            = document.getElementById('sidebar');

// ── State ────────────────────────────────────────────────────────────────────
let isLoading   = false;
let chatSessions = [];   // [{id, title, messages:[]}]
let activeChatId = null;

// ── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  startNewChat();
  attachListeners();
})();

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.chain_loaded) {
      setStatus('ready', 'Ready');
    } else {
      setStatus('loading', 'Loading model…');
      // Poll until ready
      setTimeout(checkHealth, 3000);
    }
  } catch {
    setStatus('error', 'Cannot reach server');
  }
}

function setStatus(type, label) {
  statusPill.className = `status-pill ${type}`;
  statusText.textContent = label;
}

// ── Chat session management ───────────────────────────────────────────────────
function startNewChat() {
  const id = Date.now().toString();
  chatSessions.push({ id, title: 'New Chat', messages: [] });
  activeChatId = id;
  renderMessages();
  renderSidebarHistory();
}

function getActiveSession() {
  return chatSessions.find(s => s.id === activeChatId);
}

// ── Sidebar history ───────────────────────────────────────────────────────────
function renderSidebarHistory() {
  chatHistory.innerHTML = '';
  [...chatSessions].reverse().forEach(session => {
    const btn = document.createElement('button');
    btn.className = `history-item${session.id === activeChatId ? ' active' : ''}`;
    btn.textContent = session.title;
    btn.addEventListener('click', () => {
      activeChatId = session.id;
      renderMessages();
      renderSidebarHistory();
    });
    chatHistory.appendChild(btn);
  });
}

// ── Message rendering ─────────────────────────────────────────────────────────
function renderMessages() {
  const session = getActiveSession();
  // Clear non-welcome content
  const existingMessages = messagesContainer.querySelectorAll('.message-row');
  existingMessages.forEach(el => el.remove());

  if (session.messages.length === 0) {
    welcomeScreen.style.display = 'flex';
  } else {
    welcomeScreen.style.display = 'none';
    session.messages.forEach(msg => appendMessageToDOM(msg));
    scrollToBottom();
  }
}

function appendMessageToDOM(msg) {
  const row = buildMessageRow(msg);
  messagesContainer.appendChild(row);
}

function buildMessageRow(msg) {
  const row = document.createElement('div');
  row.className = `message-row ${msg.role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${msg.role}`;
  avatar.textContent = msg.role === 'user' ? 'You' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  content.textContent = msg.content;

  const time = document.createElement('div');
  time.className = 'bubble-time';
  time.textContent = formatTime(msg.ts);

  bubble.appendChild(content);
  bubble.appendChild(time);

  // Source citations for assistant messages
  if (msg.role === 'assistant' && msg.sources && msg.sources.length > 0) {
    const toggle = document.createElement('button');
    toggle.className = 'sources-toggle';
    toggle.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''}
    `;

    const list = document.createElement('div');
    list.className = 'sources-list';
    msg.sources.forEach(src => {
      const chip = document.createElement('div');
      chip.className = 'source-chip';
      const page   = src.page   ?? src.page_number ?? '?';
      const source = src.source ?? src.file_name   ?? 'Document';
      chip.innerHTML = `
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        Page ${page}
      `;
      list.appendChild(chip);
    });

    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      list.classList.toggle('open');
    });

    bubble.appendChild(toggle);
    bubble.appendChild(list);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

function buildTypingRow() {
  const row = document.createElement('div');
  row.className = 'message-row assistant typing-indicator';
  row.id = 'typingIndicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar assistant';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  content.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

  bubble.appendChild(content);
  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

// ── Send flow ─────────────────────────────────────────────────────────────────
async function sendQuestion(question) {
  if (isLoading || !question.trim()) return;

  isLoading = true;
  sendBtn.disabled = true;
  questionInput.value = '';
  autoResize();

  // Hide welcome screen
  welcomeScreen.style.display = 'none';

  const session = getActiveSession();

  // Update session title on first message
  if (session.messages.length === 0) {
    session.title = question.length > 36 ? question.slice(0, 33) + '…' : question;
    renderSidebarHistory();
  }

  // Add user message
  const userMsg = { role: 'user', content: question, ts: Date.now() };
  session.messages.push(userMsg);
  appendMessageToDOM(userMsg);
  scrollToBottom();

  // Show typing indicator
  const typingRow = buildTypingRow();
  messagesContainer.appendChild(typingRow);
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    typingRow.remove();

    const assistantMsg = {
      role: 'assistant',
      content: data.answer,
      sources: data.sources || [],
      ts: Date.now(),
    };
    session.messages.push(assistantMsg);
    appendMessageToDOM(assistantMsg);

  } catch (err) {
    typingRow.remove();
    const errMsg = {
      role: 'assistant',
      content: `⚠️ ${err.message || 'Something went wrong. Please try again.'}`,
      sources: [],
      ts: Date.now(),
    };
    session.messages.push(errMsg);
    appendMessageToDOM(errMsg);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    scrollToBottom();
    questionInput.focus();
  }
}

// ── Listeners ─────────────────────────────────────────────────────────────────
function attachListeners() {
  // Send on click
  sendBtn.addEventListener('click', () => {
    sendQuestion(questionInput.value.trim());
  });

  // Send on Enter (Shift+Enter = newline)
  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(questionInput.value.trim());
    }
  });

  // Enable/disable send button
  questionInput.addEventListener('input', () => {
    sendBtn.disabled = !questionInput.value.trim() || isLoading;
    autoResize();
  });

  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      sendQuestion(chip.dataset.q);
    });
  });

  // New chat
  newChatBtn.addEventListener('click', startNewChat);

  // Sidebar toggle (desktop)
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  // Mobile sidebar
  mobileSidebarBtn.addEventListener('click', () => {
    sidebar.classList.toggle('mobile-open');
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function autoResize() {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
