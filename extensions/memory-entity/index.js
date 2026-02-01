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
  minScore: 0.6,
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

const TYPE_MAP = {
  entity: "entity",
  perfil: "entity",
  profile: "entity",
  client: "client",
  cliente: "client",
  staff: "staff",
  equipo: "staff",
  team: "staff",
  service: "service",
  svc: "service",
  vendor: "vendor",
  proveedor: "vendor",
  partner: "partner",
  project: "project",
  producto: "product",
  product: "product",
  app: "service",
  sistema: "service"
};

function resolveDefaultDbPath() {
  return join(homedir(), ".openclaw", "memory", "entities");
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
          name: "",
          type: "",
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

  async search(vector, limit = 5, minScore = 0.5) {
    await this.ensureInitialized();
    const results = await this.table.vectorSearch(vector).limit(limit).toArray();
    const mapped = results.map((row) => {
      const distance = row._distance ?? 0;
      const score = 1 / (1 + distance);
      return {
        entry: {
          id: row.id,
          text: row.text,
          vector: row.vector,
          createdAt: row.createdAt,
          name: row.name,
          type: row.type,
          meta: row.meta
        },
        score
      };
    });
    return mapped.filter((item) => item.score >= minScore);
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

function stripInjected(text) {
  if (!text) return text;
  return text
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/gi, "")
    .replace(/<episodic-memories>[\s\S]*?<\/episodic-memories>/gi, "")
    .replace(/<procedural-memories>[\s\S]*?<\/procedural-memories>/gi, "")
    .replace(/<tool-skill-memories>[\s\S]*?<\/tool-skill-memories>/gi, "")
    .replace(/<entity-memories>[\s\S]*?<\/entity-memories>/gi, "");
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

function normalizeType(raw) {
  const key = String(raw || "").toLowerCase().trim();
  return TYPE_MAP[key] || "entity";
}

function extractEntitiesFromText(text, cfg) {
  if (!text || typeof text !== "string") return [];
  const cleaned = stripInjected(text);
  if (cleaned.length < 4) return [];

  const entities = [];
  const lines = cleaned.split(/\r?\n/);
  const prefixRegex = /^\s*(entity|perfil|profile|client|cliente|staff|equipo|team|service|svc|vendor|proveedor|partner|project|producto|product|app|sistema)\s*[:\-]\s*(.+)$/i;
  let current = null;

  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line) {
      if (current) {
        entities.push(current);
        current = null;
      }
      continue;
    }

    const match = line.match(prefixRegex);
    if (match) {
      if (current) entities.push(current);
      current = {
        type: normalizeType(match[1]),
        name: match[2].trim(),
        details: []
      };
      continue;
    }

    if (current) {
      current.details.push(line);
    }
  }

  if (current) entities.push(current);

  if (entities.length > 0 || cfg.captureMode === "explicit") {
    return entities;
  }

  if (cfg.captureMode === "pattern" || cfg.captureMode === "explicit_or_pattern") {
    const inlineRegex = /\b(client|cliente|staff|service|svc|vendor|proveedor|partner|project|producto|product|app|sistema)\s+([A-Za-z0-9][\w ._-]{2,60})/gi;
    let match;
    while ((match = inlineRegex.exec(cleaned))) {
      const type = normalizeType(match[1]);
      const name = match[2].trim();
      if (name.length < 2) continue;
      entities.push({
        type,
        name,
        details: [cleaned.slice(0, 200)]
      });
      if (entities.length >= 5) break;
    }
  }

  return entities;
}

function loadIndex(indexPath) {
  try {
    if (!fs.existsSync(indexPath)) return { entities: {} };
    const raw = fs.readFileSync(indexPath, "utf8");
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return { entities: {} };
    if (!data.entities || typeof data.entities !== "object") data.entities = {};
    return data;
  } catch {
    return { entities: {} };
  }
}

function saveIndex(indexPath, index) {
  try {
    fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  } catch {
    // best effort
  }
}

function shouldRecallEntities(prompt) {
  const p = prompt.toLowerCase();
  return /cliente|client|perfil|profile|entity|staff|equipo|team|service|svc|vendor|proveedor|partner|project|producto|product|account|cuenta/.test(p);
}

const memoryEntityPlugin = {
  id: "memory-entity",
  name: "Memory (Entity)",
  description: "Rich entity profiles for clients/staff/services",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });

    const indexPath = join(resolvedDbPath, "entities.json");
    const vectorDim = EMBEDDING_DIMENSIONS[cfg.embedding.model];
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model);
    const table = new VectorTable(resolvedDbPath, vectorDim, "entities");

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || event.prompt.length < 3) return;
        const prompt = event.prompt;
        const index = loadIndex(indexPath);

        const promptLower = prompt.toLowerCase();
        const matched = [];
        for (const key of Object.keys(index.entities || {})) {
          const entry = index.entities[key];
          if (!entry || !entry.name) continue;
          if (promptLower.includes(String(entry.name).toLowerCase())) {
            matched.push(entry);
          }
        }

        const allowVector = cfg.alwaysRecall || shouldRecallEntities(prompt);
        const vectorResults = [];
        if (allowVector) {
          try {
            const vector = await embeddings.embed(prompt);
            const results = await table.search(vector, cfg.recallLimit, cfg.minScore);
            for (const item of results) {
              vectorResults.push({
                name: item.entry.name,
                type: item.entry.type,
                summary: item.entry.text
              });
            }
          } catch (err) {
            api.logger?.warn?.(`memory-entity: recall failed: ${String(err)}`);
          }
        }

        const combined = [];
        const seen = new Set();
        for (const entry of [...matched, ...vectorResults]) {
          if (!entry) continue;
          const key = `${String(entry.type || "entity").toLowerCase()}::${String(entry.name || "").toLowerCase()}`;
          if (seen.has(key)) continue;
          seen.add(key);
          combined.push(entry);
          if (combined.length >= cfg.recallLimit) break;
        }

        if (!combined.length) return;
        const lines = combined.map((entry, idx) => {
          const summary = entry.summary || "";
          const label = entry.type ? `[${entry.type}]` : "[entity]";
          return `${idx + 1}. ${label} ${truncate(summary, 400)}`;
        });
        const body = truncate(lines.join("\n"), cfg.maxChars);
        recordRouting("entity", body.length);
        return { prependContext: `<entity-memories>\n${body}\n</entity-memories>` };
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

      const index = loadIndex(indexPath);
      const created = [];

      for (const raw of texts) {
        const entities = extractEntitiesFromText(raw, cfg);
        for (const entity of entities) {
          if (!entity.name) continue;
          const details = entity.details && entity.details.length ? entity.details.join(" ") : "";
          const summary = redactSensitive(
            truncate(`Entity: ${entity.name}\nType: ${entity.type}\nDetails: ${details}`, cfg.maxChars),
            cfg.redaction.enabled
          );

          const vector = await embeddings.embed(summary);
          await table.store({
            text: summary,
            vector,
            name: entity.name,
            type: entity.type,
            meta: JSON.stringify({
              name: entity.name,
              type: entity.type,
              details,
              capturedAt: Date.now()
            })
          });

          const key = `${String(entity.type).toLowerCase()}::${String(entity.name).toLowerCase()}`;
          index.entities[key] = {
            name: entity.name,
            type: entity.type,
            summary,
            updatedAt: Date.now()
          };
          created.push(key);
        }
      }

      if (created.length) {
        saveIndex(indexPath, index);
      }
    });
  }
};

export default memoryEntityPlugin;
