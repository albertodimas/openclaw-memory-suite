import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const DEFAULT_META_PATH = join(homedir(), ".openclaw", "memory", "meta.json");

function resolveMetaPath() {
  return process.env.OPENCLAW_META_PATH || DEFAULT_META_PATH;
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
    fs.mkdirSync(join(path, ".."), { recursive: true });
    fs.writeFileSync(path, JSON.stringify(meta, null, 2));
  } catch {
    // best effort
  }
}

function ensureLayer(stats, layer) {
  const layers = stats.layers || (stats.layers = {});
  const entry = layers[layer] || (layers[layer] = {
    activations: 0,
    chars_injected: 0,
    useful_up: 0,
    useful_down: 0,
    last_activated_at: null
  });
  return entry;
}

export function recordRouting(layer, chars) {
  if (!layer) return;
  const path = resolveMetaPath();
  const meta = loadMeta(path);
  const stats = meta.routing_stats || (meta.routing_stats = {});
  const entry = ensureLayer(stats, layer);

  const safeChars = Number.isFinite(Number(chars)) ? Number(chars) : 0;
  entry.activations = (entry.activations || 0) + 1;
  entry.chars_injected = (entry.chars_injected || 0) + safeChars;
  entry.last_activated_at = Date.now();

  stats.total_activations = (stats.total_activations || 0) + 1;
  stats.total_chars_injected = (stats.total_chars_injected || 0) + safeChars;
  stats.current_session_chars = (stats.current_session_chars || 0) + safeChars;
  stats.current_session_activations = (stats.current_session_activations || 0) + 1;
  stats.last_updated_at = Date.now();

  saveMeta(path, meta);
}

export function recordRoutingFeedback(layer, useful) {
  if (!layer) return;
  const path = resolveMetaPath();
  const meta = loadMeta(path);
  const stats = meta.routing_stats || (meta.routing_stats = {});
  const entry = ensureLayer(stats, layer);

  if (useful) {
    entry.useful_up = (entry.useful_up || 0) + 1;
  } else {
    entry.useful_down = (entry.useful_down || 0) + 1;
  }

  const total = (entry.useful_up || 0) + (entry.useful_down || 0);
  if (total > 0) {
    entry.useful_rate = Number((entry.useful_up / total).toFixed(2));
  }
  entry.last_feedback_at = Date.now();

  stats.last_updated_at = Date.now();
  saveMeta(path, meta);
}
