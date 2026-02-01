import * as lancedb from "@lancedb/lancedb";
import OpenAI from "openai";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { recordRouting } from "../_shared/meta-routing.js";

const DEFAULTS = {
  enabled: true,
  dbPath: "",
  embedding: {
    apiKey: "local",
    model: "text-embedding-3-large"
  },
  episodic: {
    enabled: true,
    alwaysRecall: false,
    recallLimit: 3,
    minScore: 0.45,
    halfLifeDays: 14,
    maxChars: 1200
  },
  procedural: {
    enabled: true,
    alwaysRecall: false,
    recallLimit: 3,
    minScore: 0.4,
    halfLifeDays: 60,
    minSteps: 2,
    maxSteps: 8,
    maxChars: 1400
  },
  capture: {
    maxUserChars: 600,
    maxAssistantChars: 600,
    maxTools: 8
  },
  redaction: {
    enabled: true
  }
};

const EMBEDDING_DIMENSIONS = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072
};

function resolveDefaultDbPath() {
  return join(homedir(), ".openclaw", "memory", "epiproc");
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
  const episodic = cfg.episodic && typeof cfg.episodic === "object" ? cfg.episodic : {};
  const procedural = cfg.procedural && typeof cfg.procedural === "object" ? cfg.procedural : {};
  const capture = cfg.capture && typeof cfg.capture === "object" ? cfg.capture : {};
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
    episodic: {
      enabled: episodic.enabled !== false,
      alwaysRecall: episodic.alwaysRecall === true,
      recallLimit: Math.max(1, Math.floor(toNumber(episodic.recallLimit, DEFAULTS.episodic.recallLimit))),
      minScore: Math.max(0, Math.min(1, toNumber(episodic.minScore, DEFAULTS.episodic.minScore))),
      halfLifeDays: Math.max(1, toNumber(episodic.halfLifeDays, DEFAULTS.episodic.halfLifeDays)),
      maxChars: Math.max(200, Math.floor(toNumber(episodic.maxChars, DEFAULTS.episodic.maxChars)))
    },
    procedural: {
      enabled: procedural.enabled !== false,
      alwaysRecall: procedural.alwaysRecall === true,
      recallLimit: Math.max(1, Math.floor(toNumber(procedural.recallLimit, DEFAULTS.procedural.recallLimit))),
      minScore: Math.max(0, Math.min(1, toNumber(procedural.minScore, DEFAULTS.procedural.minScore))),
      halfLifeDays: Math.max(1, toNumber(procedural.halfLifeDays, DEFAULTS.procedural.halfLifeDays)),
      minSteps: Math.max(1, Math.floor(toNumber(procedural.minSteps, DEFAULTS.procedural.minSteps))),
      maxSteps: Math.max(1, Math.floor(toNumber(procedural.maxSteps, DEFAULTS.procedural.maxSteps))),
      maxChars: Math.max(200, Math.floor(toNumber(procedural.maxChars, DEFAULTS.procedural.maxChars)))
    },
    capture: {
      maxUserChars: Math.max(100, Math.floor(toNumber(capture.maxUserChars, DEFAULTS.capture.maxUserChars))),
      maxAssistantChars: Math.max(100, Math.floor(toNumber(capture.maxAssistantChars, DEFAULTS.capture.maxAssistantChars))),
      maxTools: Math.max(1, Math.floor(toNumber(capture.maxTools, DEFAULTS.capture.maxTools)))
    },
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

const DB_CONNECTIONS = new Map();

async function getConnection(dbPath) {
  if (DB_CONNECTIONS.has(dbPath)) return DB_CONNECTIONS.get(dbPath);
  const conn = await lancedb.connect(dbPath);
  DB_CONNECTIONS.set(dbPath, conn);
  return conn;
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
    this.db = await getConnection(this.dbPath);
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
          kind: "",
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
          kind: row.kind,
          meta: row.meta
        },
        score
      };
    });
  }

  async count() {
    await this.ensureInitialized();
    return this.table.countRows();
  }
}

function truncate(text, maxChars) {
  if (!text || typeof text !== "string") return "";
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trim() + "...";
}

function stripInjected(text) {
  if (!text) return text;
  return text
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/gi, "")
    .replace(/<episodic-memories>[\s\S]*?<\/episodic-memories>/gi, "")
    .replace(/<procedural-memories>[\s\S]*?<\/procedural-memories>/gi, "");
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

function collectContext(messages) {
  const userTexts = [];
  const assistantTexts = [];
  const toolCalls = new Map();
  const toolResults = [];
  const toolNames = new Set();

  if (!Array.isArray(messages)) {
    return { userTexts, assistantTexts, toolCalls, toolResults, toolNames };
  }

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const role = msg.role;

    if (role === "user") {
      const text = extractTextContent(msg.content);
      if (text) userTexts.push(text);
      continue;
    }

    if (role === "assistant") {
      const text = extractTextContent(msg.content);
      if (text) assistantTexts.push(text);
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (!block || typeof block !== "object") continue;
          if (block.type === "toolCall") {
            const id = block.id;
            const name = block.name;
            const args = block.arguments;
            if (id && name) {
              toolCalls.set(id, { name, args });
              toolNames.add(name);
            }
          }
        }
      }
      continue;
    }

    if (role === "toolResult" || role === "tool") {
      const toolName = msg.toolName || msg.name;
      if (toolName) toolNames.add(toolName);
      toolResults.push({
        toolName,
        toolCallId: msg.toolCallId,
        isError: Boolean(msg.isError),
        details: msg.details || {},
        contentText: extractTextContent(msg.content)
      });
    }
  }

  return { userTexts, assistantTexts, toolCalls, toolResults, toolNames };
}

function summarizeArgs(toolName, args, redactEnabled) {
  if (!args || typeof args !== "object") return "";
  if (toolName === "exec" || toolName === "bash") {
    const cmd = args.command || args.cmd || "";
    if (!cmd) return "";
    const cleaned = redactSensitive(String(cmd), redactEnabled);
    return truncate(cleaned, 120);
  }
  if (typeof args.path === "string") return truncate(redactSensitive(args.path, redactEnabled), 80);
  if (typeof args.paths === "string") return truncate(redactSensitive(args.paths, redactEnabled), 80);
  if (Array.isArray(args.paths) && args.paths.length > 0) {
    return truncate(redactSensitive(args.paths.join(", "), redactEnabled), 120);
  }
  if (typeof args.file === "string") return truncate(redactSensitive(args.file, redactEnabled), 80);
  if (typeof args.url === "string") return truncate(redactSensitive(args.url, redactEnabled), 120);
  return "";
}

function patternPart(toolName, argsSummary) {
  if (!toolName) return "unknown";
  if (toolName === "exec" || toolName === "bash") {
    if (!argsSummary) return `${toolName}:cmd`;
    const token = argsSummary.split(/\s+/)[0];
    return `${toolName}:${token}`;
  }
  return toolName;
}

function shouldRecallEpisodic(prompt) {
  const p = prompt.toLowerCase();
  return (
    /que paso|que ha pasado|que ocurrió|que ocurrio|cuando|ayer|hace un|hace una|hace poco|hace \d+|la otra vez|ultima vez|última vez|reciente|recent|recently|semana pasada|mes pasado|la vez pasada|last week|yesterday|previous|previously|earlier|the other day|historial|historia|registro|log|evento|timeline/.test(p)
  );
}

function shouldRecallProcedural(prompt) {
  const p = prompt.toLowerCase();
  return (
    /como|cómo|how|procedimiento|pasos|paso a paso|instrucciones|guia|guía|tutorial|checklist|check list|comandos|comando|runbook|playbook|workflow|orquest|arregl|resolv|fix|solved|solucion|solución|diagnostic|diagnostico|diagnóstico|setup|instal|configur|verificar|verificacion|verificación/.test(p)
  );
}

function formatDate(ts) {
  try {
    return new Date(ts).toISOString();
  } catch {
    return String(ts);
  }
}

function buildEpisode(context, event, cfg, ctxMeta) {
  const userText = context.userTexts.length ? context.userTexts[context.userTexts.length - 1] : "";
  const assistantText = context.assistantTexts.length ? context.assistantTexts[context.assistantTexts.length - 1] : "";
  const tools = Array.from(context.toolNames.values());

  if (!userText && !assistantText && tools.length === 0) return null;

  const summaryUser = truncate(stripInjected(userText), cfg.capture.maxUserChars);
  const summaryAssistant = truncate(stripInjected(assistantText), cfg.capture.maxAssistantChars);
  const toolList = tools.length ? tools.join(", ") : "none";

  const text = [
    `When: ${formatDate(Date.now())}`,
    `Agent: ${ctxMeta.agentId || "main"}`,
    `User asked: ${summaryUser || "(no user text)"}`,
    `Actions: ${toolList}`,
    `Outcome: ${summaryAssistant || (event.success ? "completed" : "failed")}`,
    `Success: ${event.success ? "yes" : "no"}`
  ].join("\n");

  const meta = {
    agentId: ctxMeta.agentId || "main",
    sessionKey: ctxMeta.sessionKey || "",
    durationMs: event.durationMs || 0,
    success: Boolean(event.success),
    tools,
    userText: summaryUser,
    assistantText: summaryAssistant
  };

  return { text, meta };
}

function buildProcedure(context, event, cfg) {
  const steps = [];
  const patternParts = [];

  for (const result of context.toolResults) {
    const toolName = result.toolName || "";
    if (!toolName) continue;
    if (result.isError) continue;
    if (typeof result.details?.exitCode === "number" && result.details.exitCode !== 0) continue;

    const call = result.toolCallId ? context.toolCalls.get(result.toolCallId) : null;
    const argsSummary = summarizeArgs(toolName, call ? call.args : null, cfg.redaction.enabled);
    const stepText = argsSummary ? `${toolName}: ${argsSummary}` : toolName;
    steps.push(stepText);
    patternParts.push(patternPart(toolName, argsSummary));
    if (steps.length >= cfg.procedural.maxSteps) break;
  }

  if (steps.length < cfg.procedural.minSteps) return null;

  const userText = context.userTexts.length ? context.userTexts[context.userTexts.length - 1] : "";
  const summaryUser = truncate(stripInjected(userText), cfg.capture.maxUserChars);
  const patternKey = patternParts.join(" -> ") || "unknown";
  const success = Boolean(event.success);

  const text = [
    `Procedure pattern: ${patternKey}`,
    `Context: ${summaryUser || "(no user text)"}`,
    `Steps:`,
    ...steps.map((step, idx) => `${idx + 1}. ${step}`),
    `Success: ${success ? "yes" : "no"}`
  ].join("\n");

  const meta = {
    patternKey,
    steps,
    success,
    toolCount: steps.length
  };

  return { text, meta, patternKey, success };
}

function loadStats(statsPath) {
  try {
    if (!fs.existsSync(statsPath)) return {};
    const raw = fs.readFileSync(statsPath, "utf8");
    const data = JSON.parse(raw);
    return data && typeof data === "object" ? data : {};
  } catch {
    return {};
  }
}

function saveStats(statsPath, stats) {
  try {
    fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2));
  } catch {
    // best effort
  }
}

function updateStats(stats, patternKey, success) {
  if (!patternKey) return stats;
  const entry = stats[patternKey] || { total: 0, success: 0, lastUsedAt: 0 };
  entry.total += 1;
  if (success) entry.success += 1;
  entry.lastUsedAt = Date.now();
  stats[patternKey] = entry;
  return stats;
}

function computeDecayScore(score, createdAt, halfLifeDays) {
  const ageMs = Math.max(0, Date.now() - createdAt);
  const halfLifeMs = halfLifeDays * 24 * 60 * 60 * 1000;
  if (halfLifeMs <= 0) return score;
  const decay = Math.exp(-ageMs / halfLifeMs);
  return score * decay;
}

async function recallEntries(table, embeddings, query, cfg) {
  const vector = await embeddings.embed(query);
  const raw = await table.search(vector, cfg.recallLimit * 3);
  const scored = raw.map((item) => {
    const adjusted = computeDecayScore(item.score, item.entry.createdAt, cfg.halfLifeDays);
    return { ...item, adjustedScore: adjusted };
  });

  scored.sort((a, b) => b.adjustedScore - a.adjustedScore);
  const filtered = scored.filter((item) => item.adjustedScore >= cfg.minScore);
  return filtered.slice(0, cfg.recallLimit);
}

function formatEpisodicContext(entries) {
  if (!entries.length) return "";
  const lines = entries.map((item, idx) => {
    const text = truncate(item.entry.text || "", 400);
    return `${idx + 1}. ${text}`;
  });
  return `<episodic-memories>\n${lines.join("\n")}\n</episodic-memories>`;
}

function formatProceduralContext(entries, stats) {
  if (!entries.length) return "";
  const lines = entries.map((item, idx) => {
    let meta = {};
    try {
      meta = item.entry.meta ? JSON.parse(item.entry.meta) : {};
    } catch {
      meta = {};
    }
    const patternKey = meta.patternKey || "";
    const stat = patternKey && stats[patternKey] ? stats[patternKey] : null;
    const rate = stat ? `${stat.success}/${stat.total}` : "n/a";
    const text = truncate(item.entry.text || "", 420);
    return `${idx + 1}. ${text}\n   Success rate: ${rate}`;
  });
  return `<procedural-memories>\n${lines.join("\n")}\n</procedural-memories>`;
}

const memoryEpiProcPlugin = {
  id: "memory-epiproc",
  name: "Memory (Episodic + Procedural)",
  description: "Episodic and procedural memory with recency decay",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    const resolvedDbPath = api.resolvePath(cfg.dbPath);
    fs.mkdirSync(resolvedDbPath, { recursive: true });

    const vectorDim = EMBEDDING_DIMENSIONS[cfg.embedding.model];
    const embeddings = new Embeddings(cfg.embedding.apiKey, cfg.embedding.model);

    const episodicTable = new VectorTable(resolvedDbPath, vectorDim, "episodes");
    const proceduralTable = new VectorTable(resolvedDbPath, vectorDim, "procedures");
    const statsPath = join(resolvedDbPath, "procedures-stats.json");

    api.logger?.info?.(`memory-epiproc: initialized (db: ${resolvedDbPath})`);

    api.on(
      "before_agent_start",
      async (event, ctx) => {
        if (!event?.prompt || event.prompt.length < 5) return;
        const prompt = event.prompt;

        const episodicAllowed = cfg.episodic.enabled && (cfg.episodic.alwaysRecall || shouldRecallEpisodic(prompt));
        const proceduralAllowed = cfg.procedural.enabled && (cfg.procedural.alwaysRecall || shouldRecallProcedural(prompt));
        if (!episodicAllowed && !proceduralAllowed) return;

        const parts = [];
        const stats = loadStats(statsPath);

        if (episodicAllowed) {
          try {
            const episodic = await recallEntries(episodicTable, embeddings, prompt, cfg.episodic);
            const formatted = formatEpisodicContext(episodic);
            if (formatted) {
              recordRouting("episodic", formatted.length);
              parts.push(formatted);
            }
          } catch (err) {
            api.logger?.warn?.(`memory-epiproc: episodic recall failed: ${String(err)}`);
          }
        }

        if (proceduralAllowed) {
          try {
            const procedural = await recallEntries(proceduralTable, embeddings, prompt, cfg.procedural);
            const formatted = formatProceduralContext(procedural, stats);
            if (formatted) {
              recordRouting("procedural", formatted.length);
              parts.push(formatted);
            }
          } catch (err) {
            api.logger?.warn?.(`memory-epiproc: procedural recall failed: ${String(err)}`);
          }
        }

        if (!parts.length) return;
        return { prependContext: parts.join("\n\n") };
      },
      { priority: 45 }
    );

    api.on("agent_end", async (event, ctx) => {
      if (!event || !Array.isArray(event.messages) || event.messages.length === 0) return;

      const context = collectContext(event.messages);
      const meta = { agentId: ctx?.agentId || "main", sessionKey: ctx?.sessionKey || "" };

      if (cfg.episodic.enabled) {
        try {
          const episode = buildEpisode(context, event, cfg, meta);
          if (episode) {
            const text = redactSensitive(episode.text, cfg.redaction.enabled);
            const vector = await embeddings.embed(text);
            await episodicTable.store({
              text: truncate(text, cfg.episodic.maxChars),
              vector,
              createdAt: Date.now(),
              kind: "episodic",
              meta: JSON.stringify(episode.meta)
            });
          }
        } catch (err) {
          api.logger?.warn?.(`memory-epiproc: episodic capture failed: ${String(err)}`);
        }
      }

      if (cfg.procedural.enabled) {
        try {
          const procedure = buildProcedure(context, event, cfg);
          if (procedure) {
            const text = redactSensitive(procedure.text, cfg.redaction.enabled);
            const vector = await embeddings.embed(text);
            await proceduralTable.store({
              text: truncate(text, cfg.procedural.maxChars),
              vector,
              createdAt: Date.now(),
              kind: "procedural",
              meta: JSON.stringify(procedure.meta)
            });

            let stats = loadStats(statsPath);
            stats = updateStats(stats, procedure.patternKey, procedure.success);
            saveStats(statsPath, stats);
          }
        } catch (err) {
          api.logger?.warn?.(`memory-epiproc: procedural capture failed: ${String(err)}`);
        }
      }
    });
  }
};

export default memoryEpiProcPlugin;
