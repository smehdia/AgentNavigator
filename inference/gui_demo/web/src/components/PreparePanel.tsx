import type { CheckResult, LoadItem, PreparePhase } from "../types";

interface Props {
  agent: CheckResult | null;
  device: CheckResult | null;
  imageSrc: string | null;
  deviceConnected: boolean;
  loadItems: LoadItem[];
  loadError: string | null;
  resourcesLoaded: boolean;
  phase: PreparePhase;
  error: string | null;
  onStart: () => void;
  onDisconnect: () => void;
}

const PHASE_LABELS: Record<PreparePhase, string> = {
  idle: "Ready to start",
  checking: "Running pre-flight checks…",
  connecting: "Connecting to device…",
  loading: "Loading resources…",
  done: "Setup complete",
  error: "Setup failed",
};

function CheckCard({ title, result, active }: { title: string; result: CheckResult | null; active: boolean }) {
  const ok = result?.ok;
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-semibold text-sm">{title}</h3>
          {active && !result && <p className="text-xs text-slate-400 mt-1 animate-pulse">Checking…</p>}
          {result && (
            <p className={`text-xs mt-1 ${ok ? "text-emerald-600" : "text-red-500"}`}>{result.message}</p>
          )}
          {!active && !result && <p className="text-xs text-slate-400 mt-1">Pending</p>}
        </div>
        <div
          className={`w-9 h-9 shrink-0 rounded-full flex items-center justify-center text-lg ${
            active && !result
              ? "bg-slate-100 text-slate-400"
              : ok
              ? "bg-emerald-100 text-emerald-600"
              : result
              ? "bg-red-100 text-red-500"
              : "bg-slate-100 text-slate-300"
          }`}
        >
          {active && !result ? "…" : ok ? "✓" : result ? "✗" : "○"}
        </div>
      </div>
    </div>
  );
}

function LoadStatusIcon({ status }: { status: LoadItem["status"] }) {
  if (status === "done") return <span className="w-6 h-6 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center text-xs">✓</span>;
  if (status === "loading") return <span className="w-6 h-6 rounded-full bg-indigo-100 text-accent flex items-center justify-center animate-spin text-xs">↻</span>;
  if (status === "error") return <span className="w-6 h-6 rounded-full bg-red-100 text-red-500 flex items-center justify-center text-xs">✗</span>;
  return <span className="w-6 h-6 rounded-full bg-slate-100 text-slate-400 flex items-center justify-center text-xs">○</span>;
}

const DEFAULT_LOAD_ITEMS: LoadItem[] = [
  { key: "graph", label: "Graph (graph.json)", status: "pending" },
  { key: "node_intents", label: "User intents", status: "pending" },
  { key: "node_level_information", label: "Node level information", status: "pending" },
  { key: "node_navigation_plans", label: "Navigation plans", status: "pending" },
  { key: "embeddings", label: "Embedding matrix", status: "pending" },
  { key: "localizer", label: "Localizer + OOD", status: "pending" },
  { key: "bge_model", label: "BGE-M3 model", status: "pending" },
  { key: "vlm", label: "VLM client", status: "pending" },
  { key: "agent", label: "Agent server", status: "pending" },
];

export default function PreparePanel({
  agent,
  device,
  imageSrc,
  deviceConnected,
  loadItems,
  loadError,
  resourcesLoaded,
  phase,
  error,
  onStart,
  onDisconnect,
}: Props) {
  const running = phase === "checking" || phase === "connecting" || phase === "loading";
  const displayLoad = loadItems.length ? loadItems : DEFAULT_LOAD_ITEMS;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-xl font-bold">Prepare Session</h2>
          <p className="text-sm text-slate-500 mt-1">
            Run checks, connect the device, and load graph resources in one step
          </p>
        </div>
        <div className="flex gap-2">
          {deviceConnected && (
            <button
              onClick={onDisconnect}
              disabled={running}
              className="px-4 py-2.5 bg-slate-200 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-300 disabled:opacity-50"
            >
              Disconnect
            </button>
          )}
          <button
            onClick={onStart}
            disabled={running || resourcesLoaded}
            className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50"
          >
            {running ? "Starting…" : resourcesLoaded ? "Ready ✓" : "Start"}
          </button>
        </div>
      </div>

      <div
        className={`px-4 py-3 rounded-lg text-sm font-medium border ${
          phase === "error"
            ? "bg-red-50 border-red-200 text-red-700"
            : phase === "done"
            ? "bg-emerald-50 border-emerald-200 text-emerald-700"
            : running
            ? "bg-indigo-50 border-indigo-200 text-indigo-700"
            : "bg-slate-50 border-slate-200 text-slate-600"
        }`}
      >
        {error || PHASE_LABELS[phase]}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          <section>
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Pre-flight checks</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <CheckCard title="Agent Server" result={agent} active={phase === "checking"} />
              <CheckCard title="Device Connection" result={device} active={phase === "checking"} />
            </div>
          </section>

          <section>
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Resource loading</h3>
            <div className="bg-white rounded-xl border border-slate-200 divide-y divide-slate-100 max-h-72 overflow-y-auto">
              {displayLoad.map((item) => (
                <div key={item.key} className="flex items-center gap-3 px-4 py-3">
                  <LoadStatusIcon status={item.status} />
                  <span className="flex-1 text-sm">{item.label}</span>
                  <span className="text-xs text-slate-400 capitalize">{item.status}</span>
                </div>
              ))}
            </div>
            {(loadError || error) && phase === "error" && (
              <p className="mt-2 text-xs text-red-600">{loadError || error}</p>
            )}
          </section>
        </div>

        <section>
          <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Device preview</h3>
          <div className="flex justify-center bg-slate-100 rounded-xl border border-slate-200 py-8">
            <div className="relative bg-slate-900 rounded-[2.5rem] p-3 shadow-2xl border-4 border-slate-800">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-24 h-6 bg-slate-900 rounded-b-2xl z-10" />
              <div className="w-[260px] h-[520px] bg-slate-800 rounded-[2rem] overflow-hidden flex items-center justify-center">
                {imageSrc ? (
                  <img src={imageSrc} alt="Device screenshot" className="w-full h-full object-contain" />
                ) : (
                  <div className="text-slate-500 text-sm text-center px-6">
                    {phase === "connecting"
                      ? "Connecting…"
                      : deviceConnected
                      ? "Waiting for frames…"
                      : "Preview appears after connect"}
                  </div>
                )}
              </div>
              {deviceConnected && (
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs text-emerald-600 font-medium">Live</span>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
