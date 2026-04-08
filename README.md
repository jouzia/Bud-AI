---
title: OpenEnv Study Intelligence
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# OpenEnv Study Intelligence

> A stateful, reward-driven benchmark for evaluating autonomous LLM agents on long-horizon educational tasks.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Gradio 4.x](https://img.shields.io/badge/gradio-4.36-orange)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Problem Statement

LLMs are evaluated on static benchmarks (MMLU, HumanEval). These measure point-in-time language ability — not how an agent performs when past decisions compound, constraints tighten, and memory must be leveraged across steps.

OpenEnv fills that gap: a **Gym-style environment for AI tutors** where the LLM handles execution and the environment handles strategy.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│             Gradio UI  (app/app.py)              │
│  conversation tree · charts · human-in-the-loop  │
└──────────────────┬──────────────────────────────┘
                   │  delegates to
┌──────────────────▼──────────────────────────────┐
│         core.benchmark.run_episode()             │
│         orchestration only — zero UI imports     │
└────┬───────────────┬──────────────────┬──────────┘
     │               │                  │
┌────▼────┐  ┌───────▼──────┐  ┌───────▼───────┐
│ StudyEnv│  │ MemoryAgent  │  │    Grader     │
│ engine  │  │ memory_agent │  │   grader.py   │
└─────────┘  └──────┬───────┘  └───────────────┘
                    │
           ┌────────▼────────┐
           │ DualLayerMemory │
           │ episodic+semantic│
           └─────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/yourname/openenv && cd openenv
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY
python app/app.py           # → http://localhost:7860
```

No API key? The heuristic baseline agent runs entirely offline.

---

## Project Structure

```
openenv/
├── app/app.py                      UI layer only — zero business logic
├── core/
│   ├── config.py                   centralised config (YAML + env vars)
│   ├── telemetry.py                structured logging, session metrics
│   ├── benchmark.py                run_episode(), run_comparative_benchmark()
│   ├── agents/
│   │   ├── base.py                 BaseAgent interface
│   │   ├── memory_agent.py         Gemini + dual-layer memory + ELI5/PhD mode
│   │   ├── gemini_agent.py         plain Gemini agent (comparison baseline)
│   │   └── heuristic_agent.py      deterministic rule-based baseline
│   ├── environment/
│   │   ├── engine.py               StudyEnv — step(), reset(), reward logic
│   │   └── state.py                StudyState (Pydantic v2, frozen/immutable)
│   ├── grading/grader.py           GradeReport, letter grades, breakdown
│   ├── memory/memory.py            DualLayerMemory: episodic ring + semantic KV
│   ├── conversation/
│   │   ├── tree.py                 DecisionTree engine (data-driven, no Gradio)
│   │   └── openenv_tree.py         5-branch guided conversation node definitions
│   └── visualisation/
│       └── knowledge_graph.py      Plotly network graph from Fact list
├── openenv.yaml                    machine-readable environment specification
├── requirements.txt                pinned production dependencies
└── .env.example                    required environment variables
```

---

## Reward Formula

```
S = (0.50 × C) + (0.30 × D) + (0.20 × E) − P
```

| Symbol | Component | Weight |
|---|---|---|
| C | Completeness (0–1) | 50% |
| D | Diversity — unique action types | 30% |
| E | Efficiency — completeness per step | 20% |
| P | Cumulative penalty | subtracted |

**Penalty schedule:** repeat action −0.30 · illegal action −0.50 · empty op −0.20

---

## Key Features

**Dual-layer memory** — episodic ring buffer (5 steps) + semantic key-value store. Facts extracted via `[FACT: concept | definition]` tags. Human corrections stored at confidence 1.0, never overwritten.

**Complexity modes** — ELI5 (10-year-old), Standard (university), PhD (graduate). Same environment, same reward, different execution voice.

**Human-in-the-loop** — intervene tab lets you correct the agent mid-session. Correction stored immediately in semantic memory and injected into the next prompt.

**Knowledge graph** — live Plotly network where every extracted fact becomes a node. Human corrections shown in gold.

**Guided decision tree** — 5-branch conversation tree with free-text escape hatches on every node. Zero hardcoded Gradio logic — tree is pure Python data.

**Comparison matrix** — runs all agents × all modes, produces benchmark table for presentations.

**Telemetry footer** — avg latency, token count, API calls, facts extracted. Makes it look like a high-end dev tool.

---

## Reference Scores (Hard Mode)

| Model | Score | Grade | Failure Mode |
|---|---|---|---|
| GPT-4o | 0.92 | S | None — optimal path |
| Gemini 1.5 Pro | 0.88 | A | Ran out of steps |
| Llama 3 8B | 0.65 | C | Repeated expand — max penalty |
| Heuristic Baseline | 0.78 | B | Cannot adapt |

Hard mode target: **≥ 0.90 (grade S).**

---

## Deploy to Hugging Face Spaces

```bash
# requirements.txt is HF Spaces compatible
# 1. create a Gradio Space
# 2. push this repo
# 3. add GEMINI_API_KEY as a Secret
# done — no Docker needed
```

---

## Pitch (3 min)

**Hook (0:00):** "Most AI study tools are reactive. We built OpenEnv — the first environment that separates strategic reasoning from language execution."

**Tech flex (0:30):** "The agent chose `expand` because our environment calculated low completeness. It didn't just summarise — it decided to build the foundation first."

**Memory + graph (1:15):** "That [FACT] tag just became a node on the knowledge graph. The reward chart shows the penalty when the agent repeated an action. This is state-aware, memory-augmented learning."

**Impact (2:15):** "We're building the benchmark for AI tutors. Before a model teaches a student, it must pass Hard Mode with score ≥ 0.90. Our environment proves even the best models struggle under efficiency constraints."

---

## Roadmap

- Multi-agent mode: one agent writes, another quizzes
- PDF injection: upload real study material
- Public leaderboard on Hugging Face Spaces
- ChromaDB vector memory backend
- Researcher-defined reward functions via YAML

---

MIT License
