import type { CheckResult } from "../types";

interface Props {
  agent: CheckResult | null;
  device: CheckResult | null;
  loading: boolean;
  onRun: () => void;
}

function Card({ title, result, loading }: { title: string; result: CheckResult | null; loading: boolean }) {
  const ok = result?.ok;
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 animate-fade-in">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-lg">{title}</h3>
          {loading && <p className="text-sm text-slate-400 mt-2 animate-pulse-ring">Checking…</p>}
          {!loading && result && (
            <p className={`text-sm mt-2 ${ok ? "text-emerald-600" : "text-red-500"}`}>{result.message}</p>
          )}
          {!loading && !result && <p className="text-sm text-slate-400 mt-2">Not checked yet</p>}
        </div>
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${
            loading ? "bg-slate-100" : ok ? "bg-emerald-100 text-emerald-600" : result ? "bg-red-100 text-red-500" : "bg-slate-100 text-slate-400"
          }`}
        >
          {loading ? "…" : ok ? "✓" : result ? "✗" : "?"}
        </div>
      </div>
      {result?.model_id && (
        <div className="mt-3 px-3 py-1.5 bg-slate-50 rounded-lg text-xs font-mono text-slate-600 inline-block">
          {result.model_id}
        </div>
      )}
    </div>
  );
}

export default function StatusCard({ agent, device, loading, onRun }: Props) {
  const allOk = agent?.ok && device?.ok;
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Pre-flight Checks</h2>
          <p className="text-sm text-slate-500 mt-1">Verify agent server and device before connecting</p>
        </div>
        <button
          onClick={onRun}
          disabled={loading}
          className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50"
        >
          {loading ? "Running…" : "Run Checks"}
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Agent Server" result={agent} loading={loading} />
        <Card title="Device Connection" result={device} loading={loading} />
      </div>
      {allOk && (
        <div className="px-4 py-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-700 text-sm font-medium">
          All checks passed — proceed to connect your device.
        </div>
      )}
    </div>
  );
}
