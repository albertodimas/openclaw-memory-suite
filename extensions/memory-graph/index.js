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
  minScore: 0.4,
  halfLifeDays: 30,
  alwaysRecall: false,
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
  return join(homedir(), ".openclaw", "memory", "graph");
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
          subject: "",
          relation: "",
          object: "",
          meta: "{}"
        }
      ]);
      await this.table.delete('id = "__schema__"');
    }
  }

  async store(entry) {
    await this.ensureInitialized();
    const fullEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: Date.now()
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
          subject: row.subject,
          relation: row.relation,
          object: row.object,
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

function summarizeArgs(toolName, args, redactEnabled) {
  if (!args || typeof args !== "object") return "";
  if (toolName === "exec" || toolName === "bash") {
    const cmd = args.command || args.cmd || "";
    if (!cmd) return "";
    return truncate(redactSensitive(String(cmd), redactEnabled), 120);
  }
  if (typeof args.path === "string") return truncate(redactSensitive(args.path, redactEnabled), 120);
  if (typeof args.file === "string") return truncate(redactSensitive(args.file, redactEnabled), 120);
  if (typeof args.url === "string") return truncate(redactSensitive(args.url, redactEnabled), 140);
  return "";
}

function parseEdgeParts(raw) {
  if (!raw) return null;
  const cleaned = raw.trim();
  if (!cleaned) return null;
  let parts = [];
  if (cleaned.includes("|")) {
    parts = cleaned.split("|").map((p) => p.trim()).filter(Boolean);
  } else if (cleaned.includes("->")) {
    parts = cleaned.split("->").map((p) => p.trim()).filter(Boolean);
  }
  if (parts.length === 3) {
    return { subject: parts[0], relation: parts[1], object: parts[2] };
  }
  if (parts.length === 2) {
    return { subject: parts[0], relation: "related_to", object: parts[1] };
  }
  return null;
}

function extractExplicitEdges(text) {
  if (!text || typeof text !== "string") return [];
  const edges = [];
  const lines = text.split(/\r?\n/);
  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line) continue;
    const prefixMatch = line.match(/^(rel(ation)?|graph|causal)\s*[:\-]\s*(.+)$/i);
    if (prefixMatch) {
      const parts = parseEdgeParts(prefixMatch[3]);
      if (parts) edges.push({ ...parts, source: "explicit" });
      continue;
    }
    const arrowMatch = line.match(/^(.+?)\s*--?\s*([\w\s-]{2,})\s*--?>\s*(.+)$/);
    if (arrowMatch) {
      edges.push({ subject: arrowMatch[1].trim(), relation: arrowMatch[2].trim(), object: arrowMatch[3].trim(), source: "explicit" });
    }
  }
  return edges;
}

function shouldRecallGraph(prompt) {
  const p = prompt.toLowerCase();
  return /quien|quién|who|causa|causal|relacion|relación|relation|dependency|dependencia|impacto|impact|responsable|hizo|did|root cause|why/.test(p);
}

function computeDecayScore(score, createdAt, halfLifeDays) {
  const ageMs = Math.max(0, Date.now() - createdAt);
  const halfLifeMs = halfLifeDays * 24 * 60 * 60 * 1000;
  if (halfLifeMs <= 0) return score;
  const decay = Math.exp(-ageMs / halfLifeMs);
  return score * decay;
}

const memoryGraphPlugin = {
  id: "memory-graph",
  name: "Memory (Causal/Graph)",
  description: "Causal/graph relations with who-did-what",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });

    const vectorDim = EMBEDDING_DIMENSIONS[cfg.embedding.model];
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model);
    const table = new VectorTable(resolvedDbPath, vectorDim, "edges");

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        if (!cfg.alwaysRecall && !shouldRecallGraph(event.prompt)) return;

        try {
          const vector = await embeddings.embed(event.prompt);
          const results = await table.search(vector, cfg.recallLimit * 3);
          const rescored = results
            .map((item) => ({
              ...item,
              adjusted: computeDecayScore(item.score, item.entry.createdAt, cfg.halfLifeDays)
            }))
            .filter((item) => item.adjusted >= cfg.minScore)
            .sort((a, b) => b.adjusted - a.adjusted)
            .slice(0, cfg.recallLimit);

          if (!rescored.length) return;

          const lines = rescored.map((item, idx) => {
            const text = truncate(item.entry.text || "", 400);
            return `${idx + 1}. ${text}`;
          });

          const body = truncate(lines.join("\n"), cfg.maxChars);
        recordRouting("graph", body.length);
          return { prependContext: `<causal-graph>\n${body}\n</causal-graph>` };
        } catch (err) {
          api.logger?.warn?.(`memory-graph: recall failed: ${String(err)}`);
        }
      },
      { priority: 43 }
    );

    api.on("agent_end", async (event, ctx) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;

      const edges = [];
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "user" && msg.role !== "assistant") continue;
        const text = extractTextContent(msg.content);
        if (!text) continue;
        edges.push(...extractExplicitEdges(text));
      }

      const toolCalls = new Map();
      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role === "assistant" && Array.isArray(msg.content)) {
          for (const block of msg.content) {
            if (!block || typeof block !== "object") continue;
            if (block.type === "toolCall") {
              if (block.id && block.name) {
                toolCalls.set(block.id, { name: block.name, args: block.arguments });
              }
            }
          }
        }
      }

      for (const msg of event.messages) {
        if (!msg || typeof msg !== "object") continue;
        if (msg.role !== "toolResult" && msg.role !== "tool") continue;
        if (msg.isError) continue;
        const details = msg.details || {};
        if (typeof details.exitCode === "number" && details.exitCode !== 0) continue;
        const toolName = msg.toolName || msg.name || msg.tool_name;
        if (!toolName) continue;
        const call = msg.toolCallId ? toolCalls.get(msg.toolCallId) : null;
        const argsSummary = summarizeArgs(toolName, call ? call.args : null, cfg.redaction.enabled);
        const object = argsSummary ? `${toolName}: ${argsSummary}` : toolName;
        edges.push({
          subject: ctx?.agentId || "agent",
          relation: "used",
          object,
          source: "tool"
        });
      }

      if (!edges.length) return;

      for (const edge of edges) {
        if (!edge.subject || !edge.object) continue;
        const text = redactSensitive(
          truncate(`${edge.subject} --${edge.relation || "related_to"}--> ${edge.object}`, cfg.maxChars),
          cfg.redaction.enabled
        );
        const vector = await embeddings.embed(text);
        await table.store({
          text,
          vector,
          subject: edge.subject,
          relation: edge.relation || "related_to",
          object: edge.object,
          meta: JSON.stringify({
            source: edge.source || "unknown",
            capturedAt: Date.now()
          })
        });
      }
    });
  }
};

export default memoryGraphPlugin;
