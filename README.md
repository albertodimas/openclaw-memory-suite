# OpenClaw Memory Suite (Routing + Meta Stats)

[![Release](https://img.shields.io/github/v/release/albertodimas/openclaw-memory-suite)](https://github.com/albertodimas/openclaw-memory-suite/releases)
[![License](https://img.shields.io/github/license/albertodimas/openclaw-memory-suite)](https://github.com/albertodimas/openclaw-memory-suite/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/albertodimas/openclaw-memory-suite?style=social)](https://github.com/albertodimas/openclaw-memory-suite)

Coleccion de plugins de memoria para OpenClaw con routing selectivo, estadisticas por capa y meta-memory. Diseñado para minimizar ruido en contexto y optimizar latencia/uso de tokens sin perder precision.

## Incluye
- memory-epiproc: Episodic + Procedural con decaimiento temporal.
- memory-entity: Perfiles de entidades (clientes/staff/servicios).
- memory-graph: Relaciones causales (quien-hizo-que).
- memory-blackboard: Blackboard multi-agente compartido.
- memory-toolskill: Aprendizaje de herramientas / patrones exitosos.
- memory-goal: Objetivos e intencion.
- memory-timeline: Linea de tiempo bi-temporal.
- memory-sentiment: Senales de estado/tono.
- memory-meta: Meta-memory + metricas de routing.
- memory-rerank: Reranker local (Ollama) para LTM.
- _shared/meta-routing.js: helper comun para metricas por capa.

Nota: memory-lancedb-strict se incluye como plugin opcional. Si prefieres lo estandar, usa el plugin oficial memory-lancedb y aplica la configuracion de LTM en tu openclaw.json.

## Requisitos
- OpenClaw 2026.x
- Node.js 18+
- Un proveedor de embeddings (OpenAI, Ollama/vLLM, etc.)

## Instalacion (rapida)
1) Copia las extensiones a tu OpenClaw:
```
cp -r extensions/* ~/.openclaw/extensions/
```

2) Dependencias (elige una opcion):
- Opcion A (simple): instala deps por plugin
```
cd ~/.openclaw/extensions/memory-epiproc && npm install
cd ~/.openclaw/extensions/memory-entity && npm install
cd ~/.openclaw/extensions/memory-graph && npm install
cd ~/.openclaw/extensions/memory-goal && npm install
cd ~/.openclaw/extensions/memory-timeline && npm install
cd ~/.openclaw/extensions/memory-meta && npm install
```
- Opcion B (symlink a OpenClaw global): si OpenClaw esta en /usr/lib/node_modules/openclaw
```
ln -s /usr/lib/node_modules/@lancedb ~/.openclaw/extensions/memory-epiproc/node_modules/@lancedb
ln -s /usr/lib/node_modules/openclaw/node_modules/openai ~/.openclaw/extensions/memory-epiproc/node_modules/openai
# Repite para memory-entity, memory-graph, memory-goal, memory-timeline, memory-meta
```

3) Configura plugins en ~/.openclaw/openclaw.json (ver ejemplo abajo).

## Configuracion minima (ejemplo)
Recomendado usar embeddings locales via Ollama/vLLM:
```
# .env (ejemplo)
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=local
```

Ejemplo de plugins.entries (recorta/ajusta segun tus capas):
```jsonc
"plugins": {
  "slots": { "memory": "memory-lancedb" },
  "entries": {
    "memory-lancedb": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/lancedb",
        "autoCapture": true,
        "autoRecall": true
      }
    },
    "memory-rerank": {
      "enabled": true,
      "config": {
        "ollamaUrl": "http://127.0.0.1:11434",
        "rerankModel": "dengcao/Qwen3-Reranker-8B:Q5_K_M",
        "maxDocChars": 1200
      }
    },
    "memory-epiproc": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/epiproc",
        "episodic": { "enabled": true, "recallLimit": 3, "minScore": 0.5, "halfLifeDays": 14 },
        "procedural": { "enabled": true, "recallLimit": 3, "minScore": 0.4, "halfLifeDays": 60 }
      }
    },
    "memory-entity": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/entities",
        "recallLimit": 4,
        "minScore": 0.6,
        "alwaysRecall": false
      }
    },
    "memory-graph": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/graph",
        "recallLimit": 4,
        "minScore": 0.4,
        "alwaysRecall": false
      }
    },
    "memory-blackboard": {
      "enabled": true,
      "config": { "path": "~/.openclaw/memory/blackboard.json", "maxItems": 20, "alwaysRecall": false }
    },
    "memory-toolskill": {
      "enabled": true,
      "config": { "dbPath": "~/.openclaw/memory/toolskill", "alwaysRecall": false }
    },
    "memory-goal": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/goals",
        "alwaysRecall": false
      }
    },
    "memory-timeline": {
      "enabled": true,
      "config": {
        "embedding": { "apiKey": "${OPENAI_API_KEY}", "model": "text-embedding-3-large" },
        "dbPath": "~/.openclaw/memory/timeline",
        "alwaysRecall": false
      }
    },
    "memory-sentiment": {
      "enabled": true,
      "config": { "path": "~/.openclaw/memory/sentiment.json", "alwaysRecall": false }
    },
    "memory-meta": {
      "enabled": true,
      "config": { "path": "~/.openclaw/memory/meta.json", "alwaysRecall": false }
    }
  }
}
```

## Routing stats (auto)
Cada capa que inyecta contexto registra:
- activations
- chars_injected
- useful_rate (si hay feedback)

Se guarda en: ~/.openclaw/memory/meta.json bajo routing_stats.

### Feedback manual (useful rate)
Puedes mandar feedback en texto:
```
memory-feedback layer=episodic useful=1
memory-feedback layer=procedural useful=0
```

### Ahorro de tokens
Define un baseline (antes de routing) en meta.json:
```json
"token_savings": { "before_routing_avg": 8500 }
```
Luego el sistema calcula:
- after_routing_avg
- saved_last_session
- saved_total
- saved_this_week

## Publicacion
1) Inicializa repo Git en esta carpeta:
```
cd openclaw-memory-suite
git init
```
2) Revisa el LICENSE y personaliza autor/organizacion.
3) Publica en GitHub/GitLab.

## Licencia
MIT (c) 2026 Alberto Dimas. Puedes cambiarlo en LICENSE.

## Autor
Alberto Dimas (NexoDash)
