import type { AppConfig, Candidate, CheckResult, ConfigOption } from "./types";

const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export async function fetchDefaultConfig(): Promise<{ config: AppConfig; configId: string }> {
  const data = await request<{ config: AppConfig; config_id: string }>("/api/config/default");
  return { config: data.config, configId: data.config_id };
}

export async function fetchConfigList(): Promise<{ configs: ConfigOption[]; activeConfigId: string }> {
  const data = await request<{ configs: ConfigOption[]; active_config_id: string }>("/api/configs");
  return { configs: data.configs, activeConfigId: data.active_config_id };
}

export async function selectConfig(configId: string): Promise<{ config: AppConfig; configId: string }> {
  const data = await request<{ config: AppConfig; config_id: string }>("/api/config/select", {
    method: "POST",
    body: JSON.stringify({ config_id: configId }),
  });
  return { config: data.config, configId: data.config_id };
}

export async function fetchConfig(): Promise<AppConfig> {
  const data = await request<{ config: AppConfig }>("/api/config");
  return data.config;
}

export async function saveConfig(config: AppConfig): Promise<void> {
  await request("/api/config", { method: "PUT", body: JSON.stringify({ config }) });
}

export async function runChecks(): Promise<{ agent: CheckResult; device: CheckResult }> {
  return request("/api/checks/run", { method: "POST" });
}

export async function connectDevice(): Promise<void> {
  await request("/api/device/connect", { method: "POST" });
}

export async function disconnectDevice(): Promise<void> {
  await request("/api/device/disconnect", { method: "POST" });
}

export async function retrieve(query: string): Promise<{ candidates: Candidate[]; use_memory: boolean }> {
  return request("/api/retrieve", { method: "POST", body: JSON.stringify({ query }) });
}

export async function selectNode(nodeId: string): Promise<void> {
  await request("/api/select-node", { method: "POST", body: JSON.stringify({ node_id: nodeId }) });
}

export async function fetchNavigationMemory(query: string, nodeId?: string): Promise<{ memory: string }> {
  return request("/api/navigation-memory", {
    method: "POST",
    body: JSON.stringify({
      query,
      ...(nodeId ? { node_id: nodeId } : {}),
    }),
  });
}

export async function executeStart(
  query: string,
  nodeId?: string,
  resetToStartPage: boolean = true,
  modelThinking: boolean = true,
  navigationMemory?: string
): Promise<void> {
  await request("/api/execute/start", {
    method: "POST",
    body: JSON.stringify({
      query,
      node_id: nodeId,
      reset_to_start_page: resetToStartPage,
      model_thinking: modelThinking,
      ...(navigationMemory != null ? { navigation_memory: navigationMemory } : {}),
    }),
  });
}

export async function executeStop(): Promise<void> {
  await request("/api/execute/stop", { method: "POST" });
}

export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${proto}//${host}${path}`;
}

export function subscribeLoadProgress(
  onProgress: (key: string, label: string, status: string) => void,
  onComplete: () => void,
  onError: (msg: string) => void
): () => void {
  const es = new EventSource("/api/resources/load/stream");

  es.addEventListener("progress", (e) => {
    const data = JSON.parse(e.data);
    onProgress(data.key, data.label, data.status);
  });
  es.addEventListener("complete", () => {
    es.close();
    onComplete();
  });
  es.addEventListener("error", (e: Event) => {
    if (e instanceof MessageEvent && e.data) {
      try {
        const data = JSON.parse(e.data);
        onError(data.message || "Resource load failed");
      } catch {
        onError("Resource load failed");
      }
    }
    es.close();
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    onError("Resource load stream disconnected");
    es.close();
  };

  return () => es.close();
}
