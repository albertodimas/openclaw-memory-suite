import * as lancedb from "@lancedb/lancedb";
import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  path: "",
  ltmDbPath: "",
  alwaysRecall: false,
  maxChars: 900
};

function resolveDefaultPath() {
  return join(homedir(), ".openclaw", "memory", "meta.json");
}

function resolveDefaultLtmPath() {
  return join(homedir(), ".openclaw", "memory", "lancedb");
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
    ltmDbPath: typeof cfg.ltmDbPath === "string" && cfg.ltmDbPath.trim() ? cfg.ltmDbPath.trim() : resolveDefaultLtmPath(),
    alwaysRecall: cfg.alwaysRecall === true,
    maxChars: Math.max(200, Math.floor(toNumber(cfg.maxChars, DEFAULTS.maxChars)))
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

function loadMeta(path) {
  try {
    if (!fs.existsSync(path)) return {};
    const raw = fs.readFileSync(path, "utf8");
    const data = JSON.parse(raw);
    return data && typeof data === "object" ? data : {};
  } catch {
    return {};
  }
}

function saveMeta(path, meta) {
  try {
    fs.writeFileSync(path, JSON.stringify(meta, null, 2));
  } catch {
    // best effort
  }
}

function countToolCalls(messages) {
  let count = 0;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role !== "assistant") continue;
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block && typeof block === "object" && block.type === "toolCall") {
          count += 1;
        }
      }
    }
  }
  return count;
}

function countToolErrors(messages) {
  let count = 0;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role !== "toolResult" && msg.role !== "tool") continue;
    if (msg.isError) {
      count += 1;
      continue;
    }
    const details = msg.details || {};
    if (typeof details.exitCode === "number" && details.exitCode !== 0) {
      count += 1;
    }
  }
  return count;
}

function countMemoryCommands(texts) {
  const regex = /(remember|recuerda|guardar|guarda|save memory|olvida|forget|memoriza|memoria)/i;
  let count = 0;
  for (const text of texts) {
    if (regex.test(text)) count += 1;
  }
  return count;
}

async function countLtm(dbPath) {
  try {
    const conn = await lancedb.connect(dbPath);
    const tables = await conn.tableNames();
    if (!tables.includes("memories")) return null;
    const table = await conn.openTable("memories");
    return await table.countRows();
  } catch {
    return null;
  }
}

function shouldRecall(prompt) {
  const p = String(prompt || "").toLowerCase();
  return /meta\s*memory|memory stats|estado de memoria|memoria|stats|estado del sistema/.test(p);
}

function startOfWeek(ts) {
  const dt = new Date(ts);
  const day = dt.getUTCDay();
  const diff = (day + 6) % 7;
  dt.setUTCDate(dt.getUTCDate() - diff);
  dt.setUTCHours(0, 0, 0, 0);
  return dt.getTime();
}

function parseFeedbacks(text) {
  const feedbacks = [];
  if (!text || typeof text !== "string") return feedbacks;
  const regex = /(memory-feedback|memoria-feedback)\s*[: ]\s*layer\s*=\s*([a-z0-9_-]+)\s*(?:,|\s)+\s*useful\s*=\s*(true|false|1|0)/gi;
  let match;
  while ((match = regex.exec(text)) !== null) {
    const layer = match[2];
    const raw = String(match[3]).toLowerCase();
    const useful = raw === "true" || raw === "1";
    feedbacks.push({ layer, useful });
  }
  return feedbacks;
}

function applyFeedback(meta, feedbacks) {
  if (!feedbacks || !feedbacks.length) return;
  const stats = meta.routing_stats || (meta.routing_stats = {});
  const layers = stats.layers || (stats.layers = {});
  for (const fb of feedbacks) {
    if (!fb || !fb.layer) continue;
    const entry = layers[fb.layer] || (layers[fb.layer] = {
      activations: 0,
      chars_injected: 0,
      useful_up: 0,
      useful_down: 0,
      last_activated_at: null
    });
    if (fb.useful) {
      entry.useful_up = (entry.useful_up || 0) + 1;
    } else {
      entry.useful_down = (entry.useful_down || 0) + 1;
    }
    const total = (entry.useful_up || 0) + (entry.useful_down || 0);
    if (total > 0) {
      entry.useful_rate = Number((entry.useful_up / total).toFixed(2));
    }
    entry.last_feedback_at = Date.now();
  }
  stats.last_updated_at = Date.now();
}

function finalizeRoutingSession(meta) {
  const stats = meta.routing_stats || (meta.routing_stats = {});
  const sessionChars = stats.current_session_chars || 0;
  const sessionActs = stats.current_session_activations || 0;
  stats.sessions = (stats.sessions || 0) + 1;
  stats.total_session_chars = (stats.total_session_chars || 0) + sessionChars;
  stats.total_session_activations = (stats.total_session_activations || 0) + sessionActs;
  stats.after_routing_avg = stats.sessions ? Math.round(stats.total_session_chars / stats.sessions) : 0;
  stats.last_session_chars = sessionChars;
  stats.last_session_activations = sessionActs;
  stats.current_session_chars = 0;
  stats.current_session_activations = 0;
}

function updateTokenSavings(meta) {
  const stats = meta.routing_stats;
  if (!stats) return;
  const token = meta.token_savings || (meta.token_savings = {});
  if (stats.after_routing_avg !== undefined) {
    token.after_routing_avg = stats.after_routing_avg;
  }
  const before = Number(token.before_routing_avg);
  if (!Number.isFinite(before)) return;
  const last = Number(stats.last_session_chars || 0);
  const saved = Math.max(0, Math.round(before - last));
  token.saved_last_session = saved;
  token.saved_total = (token.saved_total || 0) + saved;
  const now = Date.now();
  const weekStart = Number.isFinite(token.week_start) ? token.week_start : startOfWeek(now);
  if (now - weekStart >= 7 * 24 * 60 * 60 * 1000) {
    token.week_start = startOfWeek(now);
    token.saved_this_week = 0;
  }
  token.saved_this_week = (token.saved_this_week || 0) + saved;
}

function formatRoutingLines(meta) {
  const stats = meta.routing_stats;
  if (!stats || !stats.layers) return [];
  const entries = Object.entries(stats.layers);
  entries.sort((a, b) => (b[1].activations || 0) - (a[1].activations || 0));
  const lines = [];
  if (stats.after_routing_avg !== undefined) {
    lines.push(`Routing avg chars/session: ${stats.after_routing_avg}`);
  }
  for (const [layer, entry] of entries) {
    const activations = entry.activations || 0;
    const avgChars = activations ? Math.round((entry.chars_injected || 0) / activations) : 0;
    const useful = entry.useful_rate !== undefined ? entry.useful_rate : null;
    const usefulStr = useful === null ? "" : `, useful=${useful}`;
    lines.push(`${layer}: act=${activations}, avg_chars=${avgChars}${usefulStr}`);
  }
  return lines;
}

const memoryMetaPlugin = {
  id: "memory-meta",
  name: "Memory (Meta)",
  description: "Memory about memory health/stats",
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

        const meta = loadMeta(path);
        const lines = [];
        if (meta.sessions !== undefined) lines.push(`Sessions: ${meta.sessions}`);
        if (meta.lastSessionAt) lines.push(`Last session: ${new Date(meta.lastSessionAt).toISOString()}`);
        if (meta.toolCalls !== undefined || meta.toolErrors !== undefined) {
          lines.push(`Tool calls: ${meta.toolCalls || 0} | errors: ${meta.toolErrors || 0}`);
        }
        if (meta.memoryCommands !== undefined) {
          lines.push(`Memory commands: ${meta.memoryCommands || 0}`);
        }
        if (meta.ltmCount !== undefined && meta.ltmCount !== null) {
          lines.push(`LTM count: ${meta.ltmCount}`);
        }
        const routingLines = formatRoutingLines(meta);
        if (routingLines.length) {
          lines.push("Routing stats:");
          lines.push(...routingLines);
        }
        if (!lines.length) return;
        const body = lines.join("\n").slice(0, cfg.maxChars);
        recordRouting("meta", body.length);
        return { prependContext: `<meta-memory>\n${body}\n</meta-memory>` };
      },
      { priority: 41 }
    );

    api.on("agent_end", async (event) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;

      const meta = loadMeta(path);
      meta.sessions = (meta.sessions || 0) + 1;
      meta.lastSessionAt = Date.now();

      const toolCalls = countToolCalls(event.messages);
      const toolErrors = countToolErrors(event.messages);
      meta.toolCalls = (meta.toolCalls || 0) + toolCalls;
      meta.toolErrors = (meta.toolErrors || 0) + toolErrors;

      const texts = [];
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "user") continue;
        const text = extractTextContent(msg.content);
        if (text) texts.push(text);
      }
      const memCmds = countMemoryCommands(texts);
      if (memCmds > 0) {
        meta.memoryCommands = (meta.memoryCommands || 0) + memCmds;
        meta.lastMemoryCommandAt = Date.now();
      }

      const ltmCount = await countLtm(cfg.ltmDbPath);
      if (ltmCount !== null) {
        meta.ltmCount = ltmCount;
        meta.ltmUpdatedAt = Date.now();
      }

      const feedbacks = [];
      for (const text of texts) {
        feedbacks.push(...parseFeedbacks(text));
      }
      if (feedbacks.length) {
        applyFeedback(meta, feedbacks);
      }

      finalizeRoutingSession(meta);
      updateTokenSavings(meta);

      saveMeta(path, meta);
    });
  }
};

export default memoryMetaPlugin;
