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
  minScore: 0.45,
  halfLifeDays: 45,
  alwaysRecall: false,
  maxChars: 1200
};

const EMBEDDING_DIMENSIONS = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072
};

function resolveDefaultDbPath() {
  return join(homedir(), ".openclaw", "memory", "timeline");
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
    maxChars: Math.max(200, Math.floor(toNumber(cfg.maxChars, DEFAULTS.maxChars)))
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
          occurredAt: 0,
          recordedAt: 0,
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
      createdAt: entry.createdAt || now
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
          occurredAt: row.occurredAt,
          recordedAt: row.recordedAt,
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

function parseDateFromText(text) {
  if (!text) return null;
  const iso = text.match(/\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?/);
  if (iso) {
    const year = Number(iso[1]);
    const month = Number(iso[2]) - 1;
    const day = Number(iso[3]);
    const hour = Number(iso[4] || 0);
    const minute = Number(iso[5] || 0);
    const second = Number(iso[6] || 0);
    const dt = new Date(Date.UTC(year, month, day, hour, minute, second));
    if (!Number.isNaN(dt.getTime())) return dt.getTime();
  }
  const parsed = Date.parse(text);
  if (!Number.isNaN(parsed)) return parsed;
  return null;
}

function extractEvents(text) {
  if (!text || typeof text !== "string") return [];
  const lines = text.split(/\r?\n/);
  const events = [];
  const prefixRegex = /^\s*(event|evento|timeline|time|fecha|when|incident|incidente|deploy|release)\s*[:\-]\s*(.+)$/i;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const match = line.match(prefixRegex);
    if (match) {
      events.push(match[2].trim());
    }
  }
  return events;
}

function shouldRecall(prompt) {
  const p = String(prompt || "").toLowerCase();
  return /timeline|historial|cuando|fecha|ultima vez|última vez|event|evento|incident|incidente/.test(p);
}

function computeDecayScore(score, occurredAt, halfLifeDays) {
  const ageMs = Math.max(0, Date.now() - (occurredAt || Date.now()));
  const halfLifeMs = halfLifeDays * 24 * 60 * 60 * 1000;
  if (halfLifeMs <= 0) return score;
  const decay = Math.exp(-ageMs / halfLifeMs);
  return score * decay;
}

function formatIso(ts) {
  try {
    return new Date(ts).toISOString();
  } catch {
    return "";
  }
}

const memoryTimelinePlugin = {
  id: "memory-timeline",
  name: "Memory (Timeline)",
  description: "Bi-temporal timeline memory",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });

    const vectorDim = EMBEDDING_DIMENSIONS[cfg.embedding.model];
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model);
    const table = new VectorTable(resolvedDbPath, vectorDim, "timeline");

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        if (!cfg.alwaysRecall && !shouldRecall(event.prompt)) return;

        try {
          const vector = await embeddings.embed(event.prompt);
          const results = await table.search(vector, cfg.recallLimit * 3);
          const rescored = results
            .map((item) => ({
              ...item,
              adjusted: computeDecayScore(item.score, item.entry.occurredAt, cfg.halfLifeDays)
            }))
            .filter((item) => item.adjusted >= cfg.minScore)
            .sort((a, b) => b.adjusted - a.adjusted)
            .slice(0, cfg.recallLimit);

          if (!rescored.length) return;
          const lines = rescored.map((item) => {
            const when = item.entry.occurredAt ? formatIso(item.entry.occurredAt) : "";
            const recorded = item.entry.recordedAt ? formatIso(item.entry.recordedAt) : "";
            const prefix = when ? `[${when}]` : "[unknown]";
            const rec = recorded ? ` (recorded ${recorded})` : "";
            return `${prefix} ${truncate(item.entry.text || "", 360)}${rec}`;
          });
          const body = truncate(lines.join("\n"), cfg.maxChars);
        recordRouting("timeline", body.length);
          return { prependContext: `<timeline>\n${body}\n</timeline>` };
        } catch (err) {
          api.logger?.warn?.(`memory-timeline: recall failed: ${String(err)}`);
        }
      },
      { priority: 42 }
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

      for (const raw of texts) {
        const events = extractEvents(raw);
        for (const evt of events) {
          if (!evt) continue;
          const occurredAt = parseDateFromText(evt) || Date.now();
          const recordedAt = Date.now();
          const summary = truncate(`Event: ${evt}\nOccurred: ${formatIso(occurredAt)}\nRecorded: ${formatIso(recordedAt)}`, cfg.maxChars);
          try {
            const vector = await embeddings.embed(summary);
            await table.store({
              text: summary,
              vector,
              occurredAt,
              recordedAt,
              meta: JSON.stringify({
                event: evt,
                occurredAt,
                recordedAt
              })
            });
          } catch (err) {
            api.logger?.warn?.(`memory-timeline: capture failed: ${String(err)}`);
          }
        }
      }
    });
  }
};

export default memoryTimelinePlugin;
