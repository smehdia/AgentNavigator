import type { LoadItem } from "../types";

interface Props {
  items: LoadItem[];
  loading: boolean;
  loaded: boolean;
  error: string | null;
  onLoad: () => void;
}

const DEFAULT_ITEMS: LoadItem[] = [
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

export default function LoadProgress({ items, loading, loaded, error, onLoad }: Props) {
  const display = items.length ? items : DEFAULT_ITEMS;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Load Resources</h2>
          <p className="text-sm text-slate-500 mt-1">Preload graph, embeddings, models into memory</p>
        </div>
        <button
          onClick={onLoad}
          disabled={loading || loaded}
          className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50"
        >
          {loading ? "Loading…" : loaded ? "Loaded ✓" : "Load Resources"}
        </button>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 divide-y divide-slate-100">
        {display.map((item) => (
          <div key={item.key} className="flex items-center gap-4 px-5 py-4">
            <StatusIcon status={item.status} />
            <span className="flex-1 text-sm font-medium">{item.label}</span>
            <span className="text-xs text-slate-400 capitalize">{item.status}</span>
          </div>
        ))}
      </div>

      {error && (
        <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">{error}</div>
      )}
      {loaded && (
        <div className="px-4 py-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-700 text-sm font-medium">
          All resources loaded — ready for inference.
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: LoadItem["status"] }) {
  if (status === "done") return <span className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center text-sm">✓</span>;
  if (status === "loading") return <span className="w-8 h-8 rounded-full bg-indigo-100 text-accent flex items-center justify-center animate-spin text-sm">↻</span>;
  if (status === "error") return <span className="w-8 h-8 rounded-full bg-red-100 text-red-500 flex items-center justify-center text-sm">✗</span>;
  return <span className="w-8 h-8 rounded-full bg-slate-100 text-slate-400 flex items-center justify-center text-sm">○</span>;
}
