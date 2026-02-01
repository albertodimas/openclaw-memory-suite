import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  path: "",
  alwaysRecall: false,
  windowSize: 20,
  maxItems: 5,
  minMatch: 1
};

const POSITIVE = [
  "bien",
  "genial",
  "excelente",
  "feliz",
  "contento",
  "satisfecho",
  "gracias",
  "awesome",
  "great",
  "good",
  "happy",
  "love",
  "amazing",
  "nice"
];

const NEGATIVE = [
  "mal",
  "terrible",
  "triste",
  "enojado",
  "frustrado",
  "ansioso",
  "estresado",
  "bad",
  "angry",
  "sad",
  "annoyed",
  "problem",
  "issue",
  "fail",
  "broken"
];

function resolveDefaultPath() {
  return join(homedir(), ".openclaw", "memory", "sentiment.json");
}

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeConfig(raw) {
  const cfg = raw && typeof raw === "object" ? raw : {};
  return {
    enabled: cfg.enabled !== false,
    path: typeof cfg.path === "string" && cfg.path.trim() ? cfg.path.trim() : resolveDefaultPath(),
    alwaysRecall: cfg.alwaysRecall === true,
    windowSize: Math.max(5, Math.floor(toNumber(cfg.windowSize, DEFAULTS.windowSize))),
    maxItems: Math.max(1, Math.floor(toNumber(cfg.maxItems, DEFAULTS.maxItems))),
    minMatch: Math.max(0, Math.floor(toNumber(cfg.minMatch, DEFAULTS.minMatch)))
  };
}

function extractTextContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9áéíóúñü\s]/gi, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function sentimentScore(text) {
  const tokens = tokenize(text);
  let pos = 0;
  let neg = 0;
  for (const token of tokens) {
    if (POSITIVE.includes(token)) pos += 1;
    if (NEGATIVE.includes(token)) neg += 1;
  }
  const total = pos + neg;
  const score = total > 0 ? (pos - neg) / total : 0;
  let label = "neutral";
  if (score >= 0.2) label = "positive";
  if (score <= -0.2) label = "negative";
  return { score, label, matches: total };
}

function loadState(path) {
  try {
    if (!fs.existsSync(path)) return { entries: [] };
    const raw = fs.readFileSync(path, "utf8");
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return { entries: [] };
    if (!Array.isArray(data.entries)) data.entries = [];
    return data;
  } catch {
    return { entries: [] };
  }
}

function saveState(path, state) {
  try {
    fs.writeFileSync(path, JSON.stringify(state, null, 2));
  } catch {
    // best effort
  }
}

function shouldRecall(prompt) {
  const p = String(prompt || "").toLowerCase();
  return /sentiment|emocional|estado de animo|estado de ánimo|mood|feeling|como me siento|cómo me siento/.test(p);
}

const memorySentimentPlugin = {
  id: "memory-sentiment",
  name: "Memory (Sentiment)",
  description: "Track recent sentiment signals",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const path = api.resolvePath(cfg.path);
    fs.mkdirSync(join(path, ".."), { recursive: true });

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        if (!cfg.alwaysRecall && !shouldRecall(event.prompt)) return;

        const state = loadState(path);
        if (!state.entries.length) return;

        const recent = state.entries.slice(-cfg.windowSize);
        const last = recent[recent.length - 1];
        const avg = recent.reduce((acc, item) => acc + (item.score || 0), 0) / recent.length;

        const tail = recent.slice(-5);
        const prev = recent.slice(-10, -5);
        const tailAvg = tail.length ? tail.reduce((acc, item) => acc + (item.score || 0), 0) / tail.length : 0;
        const prevAvg = prev.length ? prev.reduce((acc, item) => acc + (item.score || 0), 0) / prev.length : 0;
        const trend = tailAvg - prevAvg;

        const lines = [];
        lines.push(`Last: ${last.label} (${last.score.toFixed(2)})`);
        lines.push(`Avg(${recent.length}): ${avg.toFixed(2)}`);
        lines.push(`Trend: ${trend >= 0 ? "+" : ""}${trend.toFixed(2)}`);

        const body = lines.join("\n");
        recordRouting("sentiment", body.length);
        return { prependContext: `<sentiment-memory>\n${body}\n</sentiment-memory>` };
      },
      { priority: 42 }
    );

    api.on("agent_end", async (event) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;

      const userTexts = [];
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "user") continue;
        const text = extractTextContent(msg.content);
        if (text) userTexts.push(text);
      }

      if (!userTexts.length) return;

      const combined = userTexts.join("\n");
      const scored = sentimentScore(combined);
      if (scored.matches < cfg.minMatch) return;

      const state = loadState(path);
      state.entries.push({
        ts: Date.now(),
        label: scored.label,
        score: scored.score,
        sample: combined.slice(0, 160)
      });

      if (state.entries.length > 500) {
        state.entries = state.entries.slice(-500);
      }

      saveState(path, state);
    });
  }
};

export default memorySentimentPlugin;
