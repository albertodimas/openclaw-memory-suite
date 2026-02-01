const DEFAULTS = {
  enabled: true,
  ollamaUrl: "http://127.0.0.1:11434",
  rerankModel: "dengcao/Qwen3-Reranker-8B:Q5_K_M",
  ltmLimit: 25,
  maxDocuments: 8,
  minScore: 0.82,
  minGap: 0.08,
  timeoutMs: 8000,
  maxTotalMs: 20000,
  maxDocChars: 800,
  promptTemplate: ""
};

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeConfig(raw) {
  const cfg = { ...DEFAULTS, ...(raw && typeof raw === "object" ? raw : {}) };
  cfg.enabled = cfg.enabled !== false;
  cfg.ollamaUrl = typeof cfg.ollamaUrl === "string" ? cfg.ollamaUrl.trim() : DEFAULTS.ollamaUrl;
  cfg.rerankModel = typeof cfg.rerankModel === "string" ? cfg.rerankModel.trim() : DEFAULTS.rerankModel;
  cfg.ltmLimit = Math.max(1, toNumber(cfg.ltmLimit, DEFAULTS.ltmLimit));
  cfg.maxDocuments = Math.max(1, toNumber(cfg.maxDocuments, DEFAULTS.maxDocuments));
  cfg.minScore = Math.max(0, Math.min(1, toNumber(cfg.minScore, DEFAULTS.minScore)));
  cfg.minGap = Math.max(0, Math.min(1, toNumber(cfg.minGap, DEFAULTS.minGap)));
  cfg.timeoutMs = Math.max(1000, toNumber(cfg.timeoutMs, DEFAULTS.timeoutMs));
  cfg.maxTotalMs = Math.max(cfg.timeoutMs, toNumber(cfg.maxTotalMs, DEFAULTS.maxTotalMs));
  cfg.maxDocChars = Math.max(200, toNumber(cfg.maxDocChars, DEFAULTS.maxDocChars));
  cfg.promptTemplate = typeof cfg.promptTemplate === "string" ? cfg.promptTemplate : "";
  return cfg;
}

function extractJsonArray(output) {
  if (!output) return null;
  const start = output.indexOf("[");
  const end = output.lastIndexOf("]");
  if (start == -1 || end == -1 || end <= start) return null;
  const jsonText = output.slice(start, end + 1);
  try {
    return JSON.parse(jsonText);
  } catch {
    return null;
  }
}

function truncate(text, maxChars) {
  if (!text || typeof text !== "string") return "";
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trim() + "…";
}

function applyTemplate(template, query, doc) {
  return template
    .replace(/\{\{\s*query\s*\}\}/gi, query)
    .replace(/\{\{\s*document\s*\}\}/gi, doc);
}

function buildPrompt(query, doc, template) {
  if (template && template.trim()) {
    return applyTemplate(template, query, doc);
  }
  return (
    "You are a reranker. Score the relevance between QUERY and DOCUMENT. " +
    "Return ONLY a single number between 0 and 100.\n\n" +
    "QUERY:\n" +
    query +
    "\n\nDOCUMENT:\n" +
    doc +
    "\n"
  );
}

async function scoreCandidate({ query, doc, cfg }) {
  if (!cfg.rerankModel || !cfg.ollamaUrl) return null;
  const url = cfg.ollamaUrl.replace(/\/$/, "") + "/api/generate";
  const prompt = buildPrompt(query, doc, cfg.promptTemplate);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), cfg.timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: cfg.rerankModel,
        prompt,
        stream: false,
        options: { temperature: 0, num_predict: 32 }
      }),
      signal: controller.signal
    });
    if (!res.ok) {
      return null;
    }
    const data = await res.json();
    const text = String(data.response ?? "").trim();
    const match = text.match(/-?\d+(?:\.\d+)?/);
    if (!match) return null;
    let score = parseFloat(match[0]);
    if (!Number.isFinite(score)) return null;
    if (score > 1) score = score / 100;
    if (score < 0) score = 0;
    if (score > 1) score = 1;
    return score;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchCandidates(api, query, cfg) {
  const argv = [
    "/usr/bin/openclaw",
    "--no-color",
    "ltm",
    "search",
    query,
    "--limit",
    String(cfg.ltmLimit)
  ];
  const result = await api.runtime.system.runCommandWithTimeout(argv, {
    timeoutMs: cfg.maxTotalMs
  });
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  const parsed = extractJsonArray(output);
  if (!Array.isArray(parsed)) return [];
  return parsed.filter((item) => item && typeof item.text === "string");
}

function selectTopCluster(sorted, cfg) {
  const selected = [];
  for (const item of sorted) {
    if (item.score < cfg.minScore) {
      continue;
    }
    if (selected.length === 0) {
      selected.push(item);
      continue;
    }
    const prev = selected[selected.length - 1];
    if (prev.score - item.score <= cfg.minGap) {
      selected.push(item);
    } else {
      break;
    }
    if (selected.length >= cfg.maxDocuments) {
      break;
    }
  }
  if (selected.length === 0) {
    return sorted.slice(0, cfg.maxDocuments);
  }
  return selected.slice(0, cfg.maxDocuments);
}

const memoryRerankPlugin = {
  id: "memory-rerank",
  name: "Memory Rerank",
  description: "Rerank memory recall results using a local model",
  version: "1.0.0",
  register(api) {
    const cfg = normalizeConfig(api.pluginConfig);
    if (!cfg.enabled) return;

    api.on(
      "before_agent_start",
      async (event) => {
        if (!event?.prompt || typeof event.prompt !== "string" || event.prompt.length < 5) {
          return;
        }
        const query = event.prompt;
        let candidates = [];
        try {
          candidates = await fetchCandidates(api, query, cfg);
        } catch (err) {
          api.logger?.warn?.(`memory-rerank: ltm search failed: ${String(err)}`);
          return;
        }
        if (!candidates.length) {
          return;
        }

        const scored = [];
        const start = Date.now();
        for (const item of candidates) {
          if (Date.now() - start > cfg.maxTotalMs) break;
          const doc = truncate(item.text, cfg.maxDocChars);
          const rerankScore = await scoreCandidate({ query, doc, cfg });
          const fallbackScore = typeof item.score === "number" ? item.score : 0;
          const score = rerankScore ?? fallbackScore;
          scored.push({
            text: item.text,
            category: item.category ?? "other",
            score
          });
        }

        if (!scored.length) {
          return;
        }

        scored.sort((a, b) => b.score - a.score);
        const selected = selectTopCluster(scored, cfg);
        const memoryContext = selected
          .map((r) => `- [${r.category}] ${truncate(r.text, 400)}`)
          .join("\n");

        return {
          prependContext:
            `<relevant-memories>\n` +
            `The following memories may be relevant to this conversation:\n` +
            `${memoryContext}\n` +
            `</relevant-memories>`
        };
      },
      { priority: 50 }
    );
  }
};

export default memoryRerankPlugin;
