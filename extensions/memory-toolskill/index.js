import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  dbPath: "",
  alwaysRecall: false,
  recallLimit: 3,
  minTotal: 2,
  minSuccessRate: 0.6,
  maxExamples: 3,
  maxChars: 900,
  redaction: {
    enabled: true
  }
};

function resolveDefaultDbPath() {
  return join(homedir(), ".openclaw", "memory", "toolskill");
}

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeConfig(raw) {
  const cfg = raw && typeof raw === "object" ? raw : {};
  const redaction = cfg.redaction && typeof cfg.redaction === "object" ? cfg.redaction : {};

  return {
    enabled: cfg.enabled !== false,
    dbPath: typeof cfg.dbPath === "string" && cfg.dbPath.trim() ? cfg.dbPath.trim() : resolveDefaultDbPath(),
    alwaysRecall: cfg.alwaysRecall === true,
    recallLimit: Math.max(1, Math.floor(toNumber(cfg.recallLimit, DEFAULTS.recallLimit))),
    minTotal: Math.max(1, Math.floor(toNumber(cfg.minTotal, DEFAULTS.minTotal))),
    minSuccessRate: Math.max(0, Math.min(1, toNumber(cfg.minSuccessRate, DEFAULTS.minSuccessRate))),
    maxExamples: Math.max(1, Math.floor(toNumber(cfg.maxExamples, DEFAULTS.maxExamples))),
    maxChars: Math.max(200, Math.floor(toNumber(cfg.maxChars, DEFAULTS.maxChars))),
    redaction: {
      enabled: redaction.enabled !== false
    }
  };
}

function redactSensitive(text, enabled) {
  if (!enabled || !text) return text || "";
  let out = text;
  const patterns = [
    /sk-[A-Za-z0-9]{20,}/g,
    /(api[_-]?key\s*[:=]\s*\S+)/gi,
    /(token\s*[:=]\s*\S+)/gi,
    /(bearer\s+\S+)/gi,
    /(authorization\s*[:=]\s*\S+)/gi
  ];
  for (const pattern of patterns) {
    out = out.replace(pattern, "[redacted]");
  }
  return out;
}

function truncate(text, maxChars) {
  if (!text || typeof text !== "string") return "";
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trim() + "...";
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

function collectToolContext(messages) {
  const toolCalls = new Map();
  const toolResults = [];

  if (!Array.isArray(messages)) return { toolCalls, toolResults };

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const role = msg.role;

    if (role === "assistant" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (!block || typeof block !== "object") continue;
        if (block.type === "toolCall") {
          const id = block.id;
          const name = block.name;
          const args = block.arguments;
          if (id && name) toolCalls.set(id, { name, args });
        }
      }
    }

    if (role === "toolResult" || role === "tool") {
      toolResults.push({
        toolName: msg.toolName || msg.name || msg.tool_name,
        toolCallId: msg.toolCallId || msg.tool_call_id,
        isError: Boolean(msg.isError),
        details: msg.details || {},
        contentText: extractTextContent(msg.content)
      });
    }
  }

  return { toolCalls, toolResults };
}

function summarizeArgs(toolName, args, redactEnabled) {
  if (!args || typeof args !== "object") return "";
  if (toolName === "exec" || toolName === "bash") {
    const cmd = args.command || args.cmd || "";
    if (!cmd) return "";
    return truncate(redactSensitive(String(cmd), redactEnabled), 140);
  }
  if (typeof args.path === "string") return truncate(redactSensitive(args.path, redactEnabled), 120);
  if (typeof args.file === "string") return truncate(redactSensitive(args.file, redactEnabled), 120);
  if (typeof args.url === "string") return truncate(redactSensitive(args.url, redactEnabled), 140);
  return "";
}

function patternKeyFor(toolName, argsSummary) {
  if (!toolName) return "unknown";
  if (toolName === "exec" || toolName === "bash") {
    if (!argsSummary) return `${toolName}:cmd`;
    const token = String(argsSummary).trim().split(/\s+/)[0] || "cmd";
    return `${toolName}:${token}`;
  }
  return toolName;
}

function ensureStatsShape(stats) {
  if (!stats || typeof stats !== "object") return { tools: {} };
  if (!stats.tools || typeof stats.tools !== "object") stats.tools = {};
  return stats;
}

function loadStats(statsPath) {
  try {
    if (!fs.existsSync(statsPath)) return { tools: {} };
    const raw = fs.readFileSync(statsPath, "utf8");
    return ensureStatsShape(JSON.parse(raw));
  } catch {
    return { tools: {} };
  }
}

function saveStats(statsPath, stats) {
  try {
    fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2));
  } catch {
    // best effort
  }
}

function updateStats(stats, toolName, patternKey, success, example, maxExamples) {
  if (!toolName) return stats;
  const now = Date.now();
  if (!stats.tools[toolName]) {
    stats.tools[toolName] = { total: 0, success: 0, lastUsedAt: 0, patterns: {} };
  }
  const tool = stats.tools[toolName];
  tool.total += 1;
  if (success) tool.success += 1;
  tool.lastUsedAt = now;

  if (!tool.patterns[patternKey]) {
    tool.patterns[patternKey] = { total: 0, success: 0, lastUsedAt: 0, examples: [] };
  }
  const pattern = tool.patterns[patternKey];
  pattern.total += 1;
  if (success) pattern.success += 1;
  pattern.lastUsedAt = now;
  if (success && example) {
    pattern.examples = pattern.examples || [];
    pattern.examples.unshift(example);
    pattern.examples = Array.from(new Set(pattern.examples)).slice(0, maxExamples);
  }

  return stats;
}

function calcRate(success, total) {
  if (!total) return 0;
  return success / total;
}

function selectToolsForPrompt(prompt, stats, alwaysRecall) {
  const tools = Object.keys(stats.tools || {});
  if (alwaysRecall || !prompt) return tools;
  const lower = prompt.toLowerCase();
  const matched = new Set();

  for (const toolName of tools) {
    if (lower.includes(toolName.toLowerCase())) matched.add(toolName);
    const patterns = stats.tools[toolName]?.patterns || {};
    for (const key of Object.keys(patterns)) {
      const token = key.includes(":") ? key.split(":")[1] : key;
      if (token && lower.includes(token.toLowerCase())) matched.add(toolName);
    }
  }

  return Array.from(matched);
}

function formatToolSummary(toolName, tool, cfg) {
  if (!tool) return "";
  const total = tool.total || 0;
  const success = tool.success || 0;
  if (total < cfg.minTotal) return "";
  const rate = calcRate(success, total);
  if (rate < cfg.minSuccessRate) return "";

  const patterns = tool.patterns || {};
  const patternEntries = Object.entries(patterns).map(([key, value]) => {
    const pTotal = value.total || 0;
    const pSuccess = value.success || 0;
    const pRate = calcRate(pSuccess, pTotal);
    return { key, pTotal, pSuccess, pRate, examples: value.examples || [] };
  });

  patternEntries.sort((a, b) => b.pRate - a.pRate || b.pTotal - a.pTotal);
  const top = patternEntries.slice(0, cfg.recallLimit);

  const lines = [];
  lines.push(`Tool: ${toolName} | success ${success}/${total} (${Math.round(rate * 100)}%)`);
  for (const entry of top) {
    const ex = entry.examples && entry.examples.length ? ` | ex: ${entry.examples[0]}` : "";
    lines.push(`- ${entry.key}: ${entry.pSuccess}/${entry.pTotal} (${Math.round(entry.pRate * 100)}%)${ex}`);
  }
  return lines.join("\n");
}

const memoryToolSkillPlugin = {
  id: "memory-toolskill",
  name: "Memory (Tool/Skill)",
  description: "Tool usage learning with success rates and patterns",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });
    const statsPath = join(resolvedDbPath, "toolskill.json");

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        const stats = loadStats(statsPath);
        const tools = selectToolsForPrompt(event.prompt, stats, cfg.alwaysRecall);
        if (!tools.length) return;

        const summaries = [];
        for (const toolName of tools) {
          const summary = formatToolSummary(toolName, stats.tools[toolName], cfg);
          if (summary) summaries.push(summary);
        }

        if (!summaries.length) return;
        const body = truncate(summaries.join("\n\n"), cfg.maxChars);
        recordRouting("tool_skill", body.length);
        return { prependContext: `<tool-skill-memories>\n${body}\n</tool-skill-memories>` };
      },
      { priority: 40 }
    );

    api.on("agent_end", async (event) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;
      const { toolCalls, toolResults } = collectToolContext(event.messages);
      if (!toolResults.length) return;

      let stats = loadStats(statsPath);
      for (const result of toolResults) {
        const toolName = result.toolName;
        if (!toolName) continue;
        const call = result.toolCallId ? toolCalls.get(result.toolCallId) : null;
        const argsSummary = summarizeArgs(toolName, call ? call.args : null, cfg.redaction.enabled);
        const example = argsSummary ? `${toolName}: ${argsSummary}` : toolName;
        const success = !result.isError && (typeof result.details?.exitCode !== "number" || result.details.exitCode === 0);
        const patternKey = patternKeyFor(toolName, argsSummary);
        stats = updateStats(stats, toolName, patternKey, success, example, cfg.maxExamples);
      }

      saveStats(statsPath, stats);
    });
  }
};

export default memoryToolSkillPlugin;
