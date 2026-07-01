# baselines/external/ — upstream baseline model checkouts

Clone each baseline's **upstream** repository here, then point the matching
`*_ROOT` variable in `baselines/.env` at it. The benchmark runners load the
upstream policy/agent code from these paths at runtime — they are **not vendored**
into this repo (license + size), only orchestrated.

```bash
cd baselines/external

git clone https://github.com/JiuTian-VL/GUI-explorer.git GUI-explorer
git clone https://github.com/<pg-agent-upstream>/PG-Agent.git PG-Agent      # arXiv 2509.03536
git clone https://github.com/TencentQQGYLab/AppAgent.git AppAgent
git clone --recurse-submodules https://github.com/ai-agents-2030/SPA-Bench.git SPA-Bench
```

This directory is git-ignored except for this README — clones stay local.

> KG-RAG is not included: its upstream depends on private, lab-internal services
> and ships no English-app configs, so it is **not runnable end-to-end** from a
> fresh clone. See `../droidtask_nav/README.md` for what we did run (our own
> agent on the DroidTask benchmark) and how.
