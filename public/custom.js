const STYLE_ID = "omniml-inline-view-style";
const HYDRATED_ATTR = "data-omniml-inline-hydrated";
const POLL_MS = 1200;

const anchorCache = new Map();

function log(message, extra) {
  if (typeof extra === "undefined") {
    console.info(`[OmniML inline] ${message}`);
    return;
  }
  console.info(`[OmniML inline] ${message}`, extra);
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .omniml-inline-card {
      width: 100%;
      margin: 16px 0 8px;
      border: 1px solid #21262d;
      border-radius: 18px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(10,12,18,0.98), rgba(13,17,23,0.98));
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }
    .omniml-inline-card__head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid #21262d;
      background: rgba(9, 12, 18, 0.92);
      color: #e6edf3;
      font-weight: 600;
    }
    .omniml-inline-card__meta {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .omniml-inline-card__title {
      font-size: 0.98rem;
      line-height: 1.25;
    }
    .omniml-inline-card__badge {
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid #30363d;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8b949e;
      white-space: nowrap;
    }
    .omniml-inline-card__frame {
      width: 100%;
      min-height: 420px;
      border: 0;
      display: block;
      background: #0d1117;
    }
    .omniml-inline-card[data-view="architecture_editor"] .omniml-inline-card__frame {
      min-height: 900px;
    }
    .omniml-inline-card[data-view="eda_dashboard"] .omniml-inline-card__frame,
    .omniml-inline-card[data-view="deployment_dashboard"] .omniml-inline-card__frame,
    .omniml-inline-card[data-view="training_config"] .omniml-inline-card__frame {
      min-height: 760px;
    }
    .omniml-inline-card[data-view="training_console"] .omniml-inline-card__frame,
    .omniml-inline-card[data-view="hpt_console"] .omniml-inline-card__frame {
      min-height: 560px;
    }
    .omniml-inline-card[data-view="pipeline_status"] .omniml-inline-card__frame {
      min-height: 320px;
    }
    @media (max-width: 900px) {
      .omniml-inline-card[data-view="architecture_editor"] .omniml-inline-card__frame {
        min-height: 780px;
      }
      .omniml-inline-card[data-view="eda_dashboard"] .omniml-inline-card__frame,
      .omniml-inline-card[data-view="deployment_dashboard"] .omniml-inline-card__frame,
      .omniml-inline-card[data-view="training_config"] .omniml-inline-card__frame {
        min-height: 640px;
      }
    }
  `;
  document.head.appendChild(style);
}

async function fetchAnchor(anchorId) {
  if (anchorCache.has(anchorId)) return anchorCache.get(anchorId);

  try {
    const response = await fetch(`/inline-view-state?anchor_id=${encodeURIComponent(anchorId)}`);
    if (!response.ok) return null;
    const payload = await response.json();
    if (!payload.ok) return null;
    anchorCache.set(anchorId, payload);
    return payload;
  } catch (error) {
    console.warn("[OmniML inline] Failed to fetch inline view state", anchorId, error);
    return null;
  }
}

function findMessageHost(node) {
  const selectors = [
    '[data-testid="message"]',
    '[data-testid="message-content"]',
    '[class*="message"]',
    "article",
    "li",
  ];

  for (const selector of selectors) {
    const match = node.closest(selector);
    if (match) return match;
  }
  return node.parentElement || node;
}

function findInsertionHost(link) {
  const localBlock = link.closest("p, li, blockquote, div");
  return localBlock || findMessageHost(link);
}

function buildCard(descriptor) {
  const card = document.createElement("section");
  card.className = "omniml-inline-card";
  card.dataset.view = descriptor.view;
  card.dataset.anchorId = descriptor.anchor_id;

  const head = document.createElement("div");
  head.className = "omniml-inline-card__head";

  const meta = document.createElement("div");
  meta.className = "omniml-inline-card__meta";

  const title = document.createElement("div");
  title.className = "omniml-inline-card__title";
  title.textContent = descriptor.title || "OmniML";

  const badge = document.createElement("div");
  badge.className = "omniml-inline-card__badge";
  badge.textContent = (descriptor.view || "view").replace(/_/g, " ");

  meta.appendChild(title);
  head.appendChild(meta);
  head.appendChild(badge);

  const frame = document.createElement("iframe");
  frame.className = "omniml-inline-card__frame";
  frame.setAttribute("title", descriptor.title || "OmniML inline view");
  frame.setAttribute("src", descriptor.url);

  card.appendChild(head);
  card.appendChild(frame);
  return card;
}

function extractAnchorId(link) {
  try {
    const parsed = new URL(link.href, window.location.origin);
    return parsed.searchParams.get("omniml_anchor");
  } catch (error) {
    return null;
  }
}

async function hydrateLink(link) {
  if (!link || link.getAttribute(HYDRATED_ATTR) === "true") return;

  const anchorId = extractAnchorId(link);
  if (!anchorId) return;

  const descriptor = await fetchAnchor(anchorId);
  if (!descriptor) return;

  const host = findInsertionHost(link);
  const messageHost = findMessageHost(link);

  if (messageHost && messageHost.querySelector(`.omniml-inline-card[data-anchor-id="${anchorId}"]`)) {
    link.setAttribute(HYDRATED_ATTR, "true");
    return;
  }

  const card = buildCard(descriptor);
  link.setAttribute(HYDRATED_ATTR, "true");
  if (host) {
    host.after(card);
  } else {
    link.after(card);
  }
  log(`hydrated ${descriptor.view}`, descriptor);
}

function findInlineLinks() {
  return Array.from(document.querySelectorAll('a[href*="omniml_anchor="]')).filter(
    (link) => link.getAttribute(HYDRATED_ATTR) !== "true"
  );
}

async function hydrateInlineViews() {
  ensureStyles();
  const links = findInlineLinks();
  for (const link of links) {
    await hydrateLink(link);
  }
}

function startInlineHydration() {
  ensureStyles();
  hydrateInlineViews();

  const observer = new MutationObserver(() => {
    hydrateInlineViews();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  window.setInterval(hydrateInlineViews, POLL_MS);
  log("initialized");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", startInlineHydration, { once: true });
} else {
  startInlineHydration();
}
