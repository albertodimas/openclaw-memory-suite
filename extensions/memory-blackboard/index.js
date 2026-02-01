import fs from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  path: "",
  maxItems: 20,
  alwaysRecall: true,
  redaction: {
    enabled: true
  }
};

function resolveDefaultPath() {
  return join(homedir(), ".openclaw", "memory", "blackboard.json");
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
    path: typeof cfg.path === "string" && cfg.path.trim() ? cfg.path.trim() : resolveDefaultPath(),
    maxItems: Math.max(1, Math.floor(toNumber(cfg.maxItems, DEFAULTS.maxItems))),
    alwaysRecall: cfg.alwaysRecall !== false,
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

function shouldRecall(prompt) {
  const p = String(prompt || "").toLowerCase();
  return /\bbb:|blackboard|pizarra|tablero|tareas|todo|decision|decisión|plan|roadmap/.test(p);
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

function loadBoard(path) {
  try {
    if (!fs.existsSync(path)) return { items: [] };
    const raw = fs.readFileSync(path, "utf8");
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return { items: [] };
    if (!Array.isArray(data.items)) data.items = [];
    return data;
  } catch {
    return { items: [] };
  }
}

function saveBoard(path, board) {
  try {
    fs.writeFileSync(path, JSON.stringify(board, null, 2));
  } catch {
    // best effort
  }
}

function parseBlackboardLines(text) {
  if (!text || typeof text !== "string") return { items: [], clear: false };
  const items = [];
  let clear = false;
  const lines = text.split(/\r?\n/);
  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line) continue;
    if (/^(bb|blackboard)\s*(clear|reset|limpiar)\b/i.test(line)) {
      clear = true;
      continue;
    }
    const match = line.match(/^(bb|blackboard|decision|decisión|todo|tarea|task|risk|riesgo|fact|hecho|note|nota|question|pregunta)\s*[:\-]\s*(.+)$/i);
    if (match) {
      const rawType = match[1].toLowerCase();
      const textValue = match[2].trim();
      const typeMap = {
        bb: "note",
        blackboard: "note",
        decision: "decision",
        "decisión": "decision",
        todo: "todo",
        tarea: "todo",
        task: "todo",
        risk: "risk",
        riesgo: "risk",
        fact: "fact",
        hecho: "fact",
        note: "note",
        nota: "note",
        question: "question",
        pregunta: "question"
      };
      const type = typeMap[rawType] || "note";
      if (textValue) items.push({ type, text: textValue });
    }
  }
  return { items, clear };
}

function upsertItem(items, newItem) {
  const key = `${newItem.type.toLowerCase()}::${newItem.text.toLowerCase()}`;
  const existingIndex = items.findIndex((item) => `${item.type.toLowerCase()}::${item.text.toLowerCase()}` === key);
  const now = Date.now();
  if (existingIndex >= 0) {
    items[existingIndex].updatedAt = now;
  } else {
    items.push({ ...newItem, updatedAt: now });
  }
}

const memoryBlackboardPlugin = {
  id: "memory-blackboard",
  name: "Memory (Collaborative Blackboard)",
  description: "Shared multi-agent blackboard",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const boardPath = api.resolvePath(cfg.path);
    fs.mkdirSync(dirname(boardPath), { recursive: true });

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 2) return;
        const prompt = event.prompt;
        if (!cfg.alwaysRecall && !shouldRecall(prompt)) return;
        const board = loadBoard(boardPath);
        if (!board.items || board.items.length === 0) return;
        const sorted = board.items
          .slice()
          .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
          .slice(0, cfg.maxItems);
        const lines = sorted.map((item) => `- [${item.type}] ${item.text}`);
        const body = lines.join("\n");
        recordRouting("blackboard", body.length);
        return { prependContext: `<collab-blackboard>\n${body}\n</collab-blackboard>` };
      },
      { priority: 60 }
    );

    api.on("agent_end", async (event) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;
      const board = loadBoard(boardPath);
      const texts = [];
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "user" && msg.role !== "assistant") continue;
        const text = extractTextContent(msg.content);
        if (text) texts.push(text);
      }
      if (!texts.length) return;

      let didChange = false;
      for (const raw of texts) {
        const parsed = parseBlackboardLines(raw);
        if (parsed.clear) {
          board.items = [];
          didChange = true;
        }
        for (const item of parsed.items) {
          const cleaned = {
            type: item.type,
            text: redactSensitive(item.text, cfg.redaction.enabled)
          };
          upsertItem(board.items, cleaned);
          didChange = true;
        }
      }

      if (didChange) {
        saveBoard(boardPath, board);
      }
    });
  }
};

export default memoryBlackboardPlugin;
