import * as lancedb from "@lancedb/lancedb";
import OpenAI from "openai";
import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  dbPath: "",
  embedding: {
    apiKey: "local",
    model: "text-embedding-3-large"
  },
  recallLimit: 4,
  minScore: 0.5,
  halfLifeDays: 30,
  alwaysRecall: false,
  captureMode: "explicit_or_pattern",
  maxChars: 1200,
  redaction: {
    enabled: true
  }
};

const EMBEDDING_DIMENSIONS = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072
};

function resolveDefaultDbPath() {
  return join(homedir(), ".openclaw", "memory", "goals");
}

function resolveEnvVars(value) {
  if (!value || typeof value !== "string") return value;
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeCaptureMode(mode) {
  const raw = typeof mode === "string" ? mode.toLowerCase().trim() : "";
  if (raw === "explicit" || raw === "explicit_or_pattern" || raw === "pattern") {
    return raw;
  }
  return DEFAULTS.captureMode;
}

function normalizeConfig(raw) {
  const cfg = raw && typeof raw === "object" ? raw : {};
  const embedding = cfg.embedding && typeof cfg.embedding === "object" ? cfg.embedding : {};
  const redaction = cfg.redaction && typeof cfg.redaction === "object" ? cfg.redaction : {};

  const model = typeof embedding.model === "string" ? embedding.model : DEFAULTS.embedding.model;
  if (!EMBEDDING_DIMENSIONS[model]) {
    throw new Error(`Unsupported embedding model: ${model}`);
  }

  const apiKeyRaw = typeof embedding.apiKey === "string" ? embedding.apiKey : DEFAULTS.embedding.apiKey;
  const apiKey = resolveEnvVars(apiKeyRaw);

  return {
    enabled: cfg.enabled !== false,
    dbPath: typeof cfg.dbPath === "string" && cfg.dbPath.trim() ? cfg.dbPath.trim() : resolveDefaultDbPath(),
    embedding: {
      apiKey: apiKey || DEFAULTS.embedding.apiKey,
      model
    },
    recallLimit: Math.max(1, Math.floor(toNumber(cfg.recallLimit, DEFAULTS.recallLimit))),
    minScore: Math.max(0, Math.min(1, toNumber(cfg.minScore, DEFAULTS.minScore))),
    halfLifeDays: Math.max(1, toNumber(cfg.halfLifeDays, DEFAULTS.halfLifeDays)),
    alwaysRecall: cfg.alwaysRecall === true,
    captureMode: normalizeCaptureMode(cfg.captureMode),
    maxChars: Math.max(200, Math.floor(toNumber(cfg.maxChars, DEFAULTS.maxChars))),
    redaction: {
      enabled: redaction.enabled !== false
    }
  };
}

class Embeddings {
  constructor(apiKey, model) {
    this.model = model;
    this.client = new OpenAI({ apiKey });
  }

  async embed(text) {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text
    });
    return response.data[0].embedding;
  }
}

class VectorTable {
  constructor(dbPath, vectorDim, tableName) {
    this.dbPath = dbPath;
    this.vectorDim = vectorDim;
    this.tableName = tableName;
    this.db = null;
    this.table = null;
    this.initPromise = null;
  }

  async ensureInitialized() {
    if (this.table) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  async doInitialize() {
    this.db = await lancedb.connect(this.dbPath);
    const tables = await this.db.tableNames();
    if (tables.includes(this.tableName)) {
      this.table = await this.db.openTable(this.tableName);
    } else {
      this.table = await this.db.createTable(this.tableName, [
        {
          id: "__schema__",
          text: "",
          vector: new Array(this.vectorDim).fill(0),
          createdAt: 0,
          updatedAt: 0,
          status: "",
          priority: "",
          owner: "",
          meta: "{}"
        }
      ]);
      await this.table.delete('id = "__schema__"');
    }
  }

  async store(entry) {
    await this.ensureInitialized();
    const now = Date.now();
    const fullEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: entry.createdAt || now,
      updatedAt: entry.updatedAt || now
    };
    await this.table.add([fullEntry]);
    return fullEntry;
  }

  async search(vector, limit = 5) {
    await this.ensureInitialized();
    const results = await this.table.vectorSearch(vector).limit(limit).toArray();
    return results.map((row) => {
      const distance = row._distance ?? 0;
      const score = 1 / (1 + distance);
      return {
        entry: {
          id: row.id,
          text: row.text,
          vector: row.vector,
          createdAt: row.createdAt,
          updatedAt: row.updatedAt,
          status: row.status,
          priority: row.priority,
          owner: row.owner,
          meta: row.meta
        },
        score
      };
    });
  }
}

function truncate(text, maxChars) {
  if (!text || typeof text !== "string") return "";
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trim() + "...";
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

function stripInjected(text) {
  if (!text) return text;
  return text
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/gi, "")
    .replace(/<episodic-memories>[\s\S]*?<\/episodic-memories>/gi, "")
    .replace(/<procedural-memories>[\s\S]*?<\/procedural-memories>/gi, "")
    .replace(/<tool-skill-memories>[\s\S]*?<\/tool-skill-memories>/gi, "")
    .replace(/<entity-memories>[\s\S]*?<\/entity-memories>/gi, "")
    .replace(/<causal-graph>[\s\S]*?<\/causal-graph>/gi, "")
    .replace(/<collab-blackboard>[\s\S]*?<\/collab-blackboard>/gi, "")
    .replace(/<goal-intent>[\s\S]*?<\/goal-intent>/gi, "");
}

function normalizeStatus(text) {
  const lower = String(text || "").toLowerCase();
  if (/done|completed|resuelto|hecho|terminado|ok\b/.test(lower)) return "done";
  if (/cancel|cancelado|abandonado/.test(lower)) return "cancelled";
  return "active";
}

function normalizePriority(text) {
  const lower = String(text || "").toLowerCase();
  if (/high|alto|urgent|urgente|crit/.test(lower)) return "high";
  if (/low|bajo/.test(lower)) return "low";
  if (/medium|medio/.test(lower)) return "medium";
  return "";
}

function parseGoals(text, cfg) {
  if (!text || typeof text !== "string") return [];
  const cleaned = stripInjected(text);
  if (!cleaned) return [];

  const lines = cleaned.split(/\r?\n/);
  const prefixRegex = /^\s*(goal|objective|objetivo|meta|intent|plan|task)\s*[:\-]\s*(.+)$/i;
  const statusRegex = /^\s*status\s*[:\-]\s*(.+)$/i;
  const priorityRegex = /^\s*priority\s*[:\-]\s*(.+)$/i;
  const ownerRegex = /^\s*(owner|due|responsable)\s*[:\-]\s*(.+)$/i;
  const goals = [];
  let current = null;

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      if (current) {
        goals.push(current);
        current = null;
      }
      continue;
    }

    const match = line.match(prefixRegex);
    if (match) {
      if (current) goals.push(current);
      current = {
        goal: match[2].trim(),
        status: normalizeStatus(match[2]),
        priority: normalizePriority(match[2]),
        owner: "",
        details: []
      };
      continue;
    }

    if (current) {
      const statusMatch = line.match(statusRegex);
      if (statusMatch) {
        current.status = normalizeStatus(statusMatch[1]);
        continue;
      }
      const priorityMatch = line.match(priorityRegex);
      if (priorityMatch) {
        current.priority = normalizePriority(priorityMatch[1]);
        continue;
      }
      const ownerMatch = line.match(ownerRegex);
      if (ownerMatch) {
        current.owner = ownerMatch[2].trim();
        continue;
      }
      current.details.push(line);
    }
  }

  if (current) goals.push(current);

  if (goals.length > 0 || cfg.captureMode === "explicit") {
    return goals;
  }

  if (cfg.captureMode === "pattern" || cfg.captureMode === "explicit_or_pattern") {
    const patternRegex = /(quiero|necesito|vamos a|we need to|we want to|goal is to|plan to)\s+([^\n]{4,120})/i;
    const match = cleaned.match(patternRegex);
    if (match) {
      goals.push({
        goal: match[2].trim(),
        status: "active",
        priority: "",
        owner: "",
        details: []
      });
    }
  }

  return goals;
}

function shouldRecallGoals(prompt) {
  const p = String(prompt || "").toLowerCase();
  return /goal|objetivo|meta|intent|plan|task|prioridad|priority/.test(p);
}

function computeDecayScore(score, updatedAt, halfLifeDays) {
  const ageMs = Math.max(0, Date.now() - (updatedAt || Date.now()));
  const halfLifeMs = halfLifeDays * 24 * 60 * 60 * 1000;
  if (halfLifeMs <= 0) return score;
  const decay = Math.exp(-ageMs / halfLifeMs);
  return score * decay;
}

const memoryGoalPlugin = {
  id: "memory-goal",
  name: "Memory (Goal/Intent)",
  description: "Goal and intent tracking",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });

    const vectorDim = EMBEDDING_DIMENSIONS[cfg.embedding.model];
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model);
    const table = new VectorTable(resolvedDbPath, vectorDim, "goals");

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        if (!cfg.alwaysRecall && !shouldRecallGoals(event.prompt)) return;

        try {
          const vector = await embeddings.embed(event.prompt);
          const results = await table.search(vector, cfg.recallLimit * 3);
          const rescored = results
            .map((item) => ({
              ...item,
              adjusted: computeDecayScore(item.score, item.entry.updatedAt, cfg.halfLifeDays)
            }))
            .filter((item) => item.adjusted >= cfg.minScore)
            .sort((a, b) => {
              const aActive = String(a.entry.status || "").toLowerCase() === "active";
              const bActive = String(b.entry.status || "").toLowerCase() === "active";
              if (aActive !== bActive) return aActive ? -1 : 1;
              return b.adjusted - a.adjusted;
            })
            .slice(0, cfg.recallLimit);

          if (!rescored.length) return;
          const lines = rescored.map((item, idx) => {
            const status = item.entry.status ? ` (${item.entry.status})` : "";
            const priority = item.entry.priority ? ` [${item.entry.priority}]` : "";
            return `${idx + 1}. ${truncate(item.entry.text || "", 400)}${priority}${status}`;
          });
          const body = truncate(lines.join("\n"), cfg.maxChars);
        recordRouting("goal", body.length);
          return { prependContext: `<goal-intent>\n${body}\n</goal-intent>` };
        } catch (err) {
          api.logger?.warn?.(`memory-goal: recall failed: ${String(err)}`);
        }
      },
      { priority: 44 }
    );

    api.on("agent_end", async (event) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;
      const texts = [];
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "user" && msg.role !== "assistant") continue;
        const text = extractTextContent(msg.content);
        if (text) texts.push(text);
      }

      if (!texts.length) return;
      const seen = new Set();

      for (const raw of texts) {
        const goals = parseGoals(raw, cfg);
        for (const goal of goals) {
          if (!goal.goal) continue;
          const key = goal.goal.toLowerCase();
          if (seen.has(key)) continue;
          seen.add(key);

          const details = goal.details && goal.details.length ? goal.details.join(" ") : "";
          const summary = redactSensitive(
            truncate(
              `Goal: ${goal.goal}\nStatus: ${goal.status || "active"}\nPriority: ${goal.priority || ""}\nOwner: ${goal.owner || ""}\nDetails: ${details}`,
              cfg.maxChars
            ),
            cfg.redaction.enabled
          );

          try {
            const vector = await embeddings.embed(summary);
            await table.store({
              text: summary,
              vector,
              status: goal.status || "active",
              priority: goal.priority || "",
              owner: goal.owner || "",
              meta: JSON.stringify({
                goal: goal.goal,
                status: goal.status || "active",
                priority: goal.priority || "",
                owner: goal.owner || "",
                details,
                capturedAt: Date.now()
              })
            });
          } catch (err) {
            api.logger?.warn?.(`memory-goal: capture failed: ${String(err)}`);
          }
        }
      }
    });
  }
};

export default memoryGoalPlugin;
