// RETER View — client-side application

// --- Theme management ---
const MERMAID_THEMES = { light: 'default', dark: 'dark' };
let currentTheme = localStorage.getItem('reter-theme') || 'light';
let lastMsg = null;
let activeHistoryId = null;

function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('reter-theme', theme);
  mermaid.initialize({ startOnLoad: false, theme: MERMAID_THEMES[theme] });
}

applyTheme(currentTheme);

document.getElementById('theme-toggle').addEventListener('click', async () => {
  applyTheme(currentTheme === 'light' ? 'dark' : 'light');
  if (lastMsg) await handleMessage(lastMsg);
});

// --- Sidebar toggle ---
function applySidebarState() {
  const collapsed = localStorage.getItem('reter-sidebar') === 'collapsed';
  document.documentElement.setAttribute('data-sidebar', collapsed ? 'collapsed' : 'expanded');
}
applySidebarState();

document.getElementById('sidebar-toggle').addEventListener('click', () => {
  const isCollapsed = document.documentElement.getAttribute('data-sidebar') === 'collapsed';
  const next = isCollapsed ? 'expanded' : 'collapsed';
  localStorage.setItem('reter-sidebar', next);
  document.documentElement.setAttribute('data-sidebar', next);
});

// --- Splitter drag-to-resize ---
(function() {
  const splitter = document.getElementById('splitter');
  const sidebar = document.getElementById('sidebar');
  const saved = localStorage.getItem('reter-sidebar-width');
  if (saved) sidebar.style.width = saved + 'px';

  let dragging = false;

  splitter.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    splitter.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const mainRect = document.getElementById('main').getBoundingClientRect();
    let w = e.clientX - mainRect.left;
    w = Math.max(160, Math.min(w, mainRect.width - 200));
    sidebar.style.width = w + 'px';
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    localStorage.setItem('reter-sidebar-width', parseInt(sidebar.style.width));
  });
})();

// --- Relative time formatting ---
function relativeTime(ts) {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  if (diff < 172800) return 'Yesterday';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// --- History sidebar ---
const historyList = document.getElementById('history-list');

function createHistoryItemEl(item) {
  const div = document.createElement('div');
  div.className = 'history-item';
  div.dataset.id = item.id;
  div.innerHTML =
    '<div class="hi-top">' +
      '<span class="hi-time">' + relativeTime(item.timestamp) + '</span>' +
      '<span class="hi-type">' + (item.content_type || item.type || '') + '</span>' +
    '</div>' +
    '<span class="hi-title">' + escapeHtml(item.title) + '</span>' +
    '<div class="hi-actions">' +
      '<button class="hi-save" title="Save to file">Save</button>' +
      '<span class="hi-saved-path"></span>' +
    '</div>';
  div.querySelector('.hi-title').addEventListener('click', () => loadHistoryItem(item.id));
  div.querySelector('.hi-top').addEventListener('click', () => loadHistoryItem(item.id));
  div.querySelector('.hi-save').addEventListener('click', (e) => {
    e.stopPropagation();
    saveHistoryItem(item.id, div);
  });
  div.querySelector('.hi-saved-path').addEventListener('click', (e) => {
    e.stopPropagation();
    const fp = e.target.dataset.filepath;
    if (fp) openFile(fp);
  });
  return div;
}

async function saveHistoryItem(id, el) {
  const btn = el.querySelector('.hi-save');
  const pathEl = el.querySelector('.hi-saved-path');
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const res = await fetch('/api/history/' + id + '/save', { method: 'POST' });
    if (!res.ok) { btn.textContent = 'Error'; return; }
    const data = await res.json();
    pathEl.textContent = data.path;
    pathEl.title = 'Click to open: ' + data.path;
    pathEl.dataset.filepath = data.path;
    pathEl.style.cursor = 'pointer';
    btn.textContent = 'Saved';
  } catch (e) {
    btn.textContent = 'Error';
    console.error('Save failed', e);
  } finally {
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Save'; }, 3000);
  }
}

async function openFile(filepath) {
  try {
    await fetch('/api/open-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: filepath }),
    });
  } catch (e) {
    console.error('Failed to open file', e);
  }
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function setActiveHistoryItem(id) {
  activeHistoryId = id;
  historyList.querySelectorAll('.history-item').forEach(el => {
    el.classList.toggle('active', String(el.dataset.id) === String(id));
  });
}

async function loadHistoryItem(id) {
  try {
    const res = await fetch('/api/history/' + id);
    if (!res.ok) return;
    const data = await res.json();
    lastMsg = { type: data.content_type, content: data.content, id: data.id };
    await handleMessage(lastMsg);
    setActiveHistoryItem(id);
  } catch (e) {
    console.error('Failed to load history item', e);
  }
}

async function loadHistoryList() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) return;
    const items = await res.json();
    historyList.innerHTML = '';
    items.forEach(item => historyList.appendChild(createHistoryItemEl(item)));
  } catch (e) {
    console.error('Failed to load history', e);
  }
}

function prependHistoryItem(msg) {
  if (!msg.id) return;
  // Skip if already in the list (e.g. cached _last_message after page load)
  if (historyList.querySelector('[data-id="' + msg.id + '"]')) {
    setActiveHistoryItem(msg.id);
    return;
  }
  const contentType = msg.type || 'html';
  // Extract title client-side (mirror server logic)
  let title = 'Untitled';
  const content = msg.content || '';
  if (contentType === 'markdown') {
    const m = content.match(/^#{1,6}\s+(.+)/m);
    if (m) title = m[1].trim().slice(0, 120);
  } else if (contentType === 'mermaid') {
    const first = content.trim().split('\n')[0] || '';
    title = first.slice(0, 120) || 'Mermaid diagram';
  } else {
    title = content.trim().slice(0, 80) || 'Untitled';
  }
  const item = {
    id: msg.id,
    timestamp: Date.now() / 1000,
    content_type: contentType,
    title: title,
  };
  const el = createHistoryItemEl(item);
  historyList.prepend(el);
  setActiveHistoryItem(msg.id);
}

// --- WebSocket connection ---
let ws, reconnectTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');

  ws.onopen = () => {
    document.getElementById('status-badge').className = 'connected';
    document.getElementById('status-badge').textContent = 'connected';
  };

  ws.onclose = () => {
    document.getElementById('status-badge').className = 'disconnected';
    document.getElementById('status-badge').textContent = 'disconnected';
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      lastMsg = msg;
      handleMessage(msg);
      prependHistoryItem(msg);
    } catch (e) {
      console.error('bad message', e);
    }
  };
}

// --- Mermaid zoom/pan ---
function wrapMermaidContainers() {
  document.querySelectorAll('.mermaid-container').forEach(container => {
    const wrapper = document.createElement('div');
    wrapper.className = 'mermaid-wrapper';

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'mermaid-toolbar';
    toolbar.innerHTML =
      '<button class="zoom-in" title="Zoom in">+</button>' +
      '<button class="zoom-out" title="Zoom out">&minus;</button>' +
      '<span class="mermaid-zoom-label">100%</span>' +
      '<button class="zoom-reset" title="Reset">&#8634;</button>';

    // Viewport + inner
    const viewport = document.createElement('div');
    viewport.className = 'mermaid-viewport';
    const inner = document.createElement('div');
    inner.className = 'mermaid-inner';

    // Move SVG content into inner
    while (container.firstChild) inner.appendChild(container.firstChild);
    viewport.appendChild(inner);
    wrapper.appendChild(toolbar);
    wrapper.appendChild(viewport);
    container.replaceWith(wrapper);

    // State
    let scale = 1, panX = 0, panY = 0, dragging = false, startX, startY;
    const MIN_SCALE = 0.1, MAX_SCALE = 5;
    const label = toolbar.querySelector('.mermaid-zoom-label');

    function applyTransform() {
      inner.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
      label.textContent = Math.round(scale * 100) + '%';
    }

    function zoomAt(cx, cy, factor) {
      const prev = scale;
      scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
      const ratio = scale / prev;
      panX = cx - ratio * (cx - panX);
      panY = cy - ratio * (cy - panY);
      applyTransform();
    }

    // Wheel zoom
    viewport.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      zoomAt(cx, cy, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });

    // Drag pan
    viewport.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      dragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      viewport.classList.add('grabbing');
    });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      applyTransform();
    });
    window.addEventListener('mouseup', () => {
      dragging = false;
      viewport.classList.remove('grabbing');
    });

    // Toolbar buttons
    toolbar.querySelector('.zoom-in').addEventListener('click', () => {
      const rect = viewport.getBoundingClientRect();
      zoomAt(rect.width / 2, rect.height / 2, 1.3);
    });
    toolbar.querySelector('.zoom-out').addEventListener('click', () => {
      const rect = viewport.getBoundingClientRect();
      zoomAt(rect.width / 2, rect.height / 2, 1 / 1.3);
    });
    toolbar.querySelector('.zoom-reset').addEventListener('click', () => {
      scale = 1; panX = 0; panY = 0;
      applyTransform();
    });
  });
}

// --- Message rendering ---
async function handleMessage(msg) {
  const el = document.getElementById('content');

  if (msg.type === 'markdown') {
    el.innerHTML = marked.parse(msg.content || '');
    // Find mermaid code blocks and convert them
    const codes = el.querySelectorAll('pre code.language-mermaid');
    for (const code of codes) {
      const container = document.createElement('div');
      container.className = 'mermaid-container';
      container.innerHTML = '<pre class="mermaid">' + code.textContent + '</pre>';
      code.parentElement.replaceWith(container);
    }
    if (el.querySelector('.mermaid')) {
      await mermaid.run({ nodes: el.querySelectorAll('.mermaid') });
      wrapMermaidContainers();
    }

  } else if (msg.type === 'mermaid') {
    el.innerHTML = '<div class="mermaid-container"><pre class="mermaid">'
      + (msg.content || '') + '</pre></div>';
    await mermaid.run({ nodes: el.querySelectorAll('.mermaid') });
    wrapMermaidContainers();

  } else if (msg.type === 'html') {
    el.innerHTML = msg.content || '';
  }
}

// --- Start ---
loadHistoryList();
connect();
