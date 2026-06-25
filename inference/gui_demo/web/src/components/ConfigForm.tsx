import type { AppConfig, ConfigOption } from "../types";

interface Props {
  config: AppConfig;
  configOptions: ConfigOption[];
  selectedConfigId: string;
  configListError?: string | null;
  onSelectConfig: (configId: string) => void;
  selectingConfig?: boolean;
  onChange: (config: AppConfig) => void;
  onSave: () => void;
  saving: boolean;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details open className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <summary className="px-5 py-3 font-semibold text-sm cursor-pointer bg-slate-50 border-b border-slate-100">
        {title}
      </summary>
      <div className="p-5 space-y-4">{children}</div>
    </details>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`flex items-center justify-between gap-4 w-full px-4 py-2.5 rounded-lg border transition-colors ${
        checked ? "border-accent bg-indigo-50" : "border-slate-200 bg-white"
      }`}
    >
      <span className="text-sm font-medium text-left">{label}</span>
      <span
        aria-hidden="true"
        className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors ${
          checked ? "bg-accent" : "bg-slate-300"
        }`}
      >
        <span
          className={`inline-block h-5 w-5 rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </span>
    </button>
  );
}

const inputCls =
  "w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent";

export default function ConfigForm({
  config,
  configOptions,
  selectedConfigId,
  configListError = null,
  onSelectConfig,
  selectingConfig = false,
  onChange,
  onSave,
  saving,
}: Props) {
  const set = (path: string[], value: unknown) => {
    const next = structuredClone(config);
    let obj: Record<string, unknown> = next as Record<string, unknown>;
    for (let i = 0; i < path.length - 1; i++) {
      if (!obj[path[i]]) obj[path[i]] = {};
      obj = obj[path[i]] as Record<string, unknown>;
    }
    obj[path[path.length - 1]] = value;
    onChange(next);
  };

  const selectedLabel =
    configOptions.find((option) => option.id === selectedConfigId)?.label ?? selectedConfigId;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold">Configuration</h2>
          <p className="text-sm text-slate-500 mt-1">
            App preset: <span className="font-medium text-slate-700">{selectedLabel || "—"}</span>
          </p>
        </div>
        <button
          onClick={onSave}
          disabled={saving || selectingConfig}
          className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50"
        >
          {saving ? "Saving…" : "Apply Config"}
        </button>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <Field label="App / config preset">
          <select
            className={inputCls}
            value={selectedConfigId || (configOptions[0]?.id ?? "")}
            disabled={selectingConfig || (configOptions.length === 0 && !configListError)}
            onChange={(e) => onSelectConfig(e.target.value)}
          >
            {configOptions.length === 0 ? (
              <option value="">{configListError ? "Failed to load configs" : "Loading configs…"}</option>
            ) : (
              configOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label} ({option.filename})
                </option>
              ))
            )}
          </select>
        </Field>
        {selectingConfig ? (
          <p className="text-xs text-indigo-600 mt-2">Loading preset…</p>
        ) : null}
        {configListError ? (
          <p className="text-xs text-red-600 mt-2">
            Could not load config list: {configListError}. Restart the GUI server after updating the backend.
          </p>
        ) : (
          <p className="text-xs text-slate-500 mt-2">
            Switching apps loads a new YAML from <code className="text-slate-600">inference/configs/</code> and resets
            device, resources, and navigation state.
          </p>
        )}
      </div>

      <Section title="Driver">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Device ID">
            <input className={inputCls} value={config.driver?.device_id ?? ""} onChange={(e) => set(["driver", "device_id"], e.target.value)} />
          </Field>
          <Field label="OS">
            <select className={inputCls} value={config.driver?.os_name ?? "android"} onChange={(e) => set(["driver", "os_name"], e.target.value)}>
              <option value="android">Android</option>
              <option value="harmony">HarmonyOS</option>
            </select>
          </Field>
          <Field label="App Package">
            <input className={inputCls} value={config.driver?.appPackage ?? ""} onChange={(e) => set(["driver", "appPackage"], e.target.value)} />
          </Field>
          <Field label="App Activity">
            <input className={inputCls} value={config.driver?.appActivity ?? ""} onChange={(e) => set(["driver", "appActivity"], e.target.value)} />
          </Field>
        </div>
        <Field label="Reset Instruction">
          <textarea className={`${inputCls} min-h-[80px]`} value={config.driver?.reset_instruction ?? ""} onChange={(e) => set(["driver", "reset_instruction"], e.target.value)} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Toggle label="Use launcher intent" checked={!!config.driver?.use_launcher_intent} onChange={(v) => set(["driver", "use_launcher_intent"], v)} />
          <Toggle label="Skip when wrong app package" checked={!!config.driver?.skip_exploration_for_no_app_package} onChange={(v) => set(["driver", "skip_exploration_for_no_app_package"], v)} />
        </div>
      </Section>

      <Section title="Agent">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Server URL">
            <input className={inputCls} value={config.agent?.url ?? ""} onChange={(e) => set(["agent", "url"], e.target.value)} />
          </Field>
          <Field label="Model">
            <select className={inputCls} value={config.agent?.model_name ?? "mai_ui"} onChange={(e) => set(["agent", "model_name"], e.target.value)}>
              <option value="mai_ui">MAI-UI</option>
              <option value="ui_tars">UI-TARS</option>
            </select>
          </Field>
          <Field label="History N">
            <input type="number" className={inputCls} value={config.agent?.settings?.history_n ?? 3} onChange={(e) => set(["agent", "settings", "history_n"], +e.target.value)} />
          </Field>
          <Field label="Resize Factor">
            <input type="number" step="0.05" className={inputCls} value={config.agent?.settings?.resize_factor ?? 0.75} onChange={(e) => set(["agent", "settings", "resize_factor"], +e.target.value)} />
          </Field>
          <Field label="Model Thinking (default)">
            <Toggle
              label="model_thinking"
              checked={config.agent?.settings?.model_thinking ?? true}
              onChange={(v) => set(["agent", "settings", "model_thinking"], v)}
            />
          </Field>
          <Field label="Max Steps">
            <input type="number" className={inputCls} value={config.agent?.max_steps ?? 10} onChange={(e) => set(["agent", "max_steps"], +e.target.value)} />
          </Field>
        </div>
      </Section>

      <Section title="VLM">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Alibaba API Key">
            <input type="password" className={inputCls} value={config.vlm?.alibaba_api_key ?? ""} onChange={(e) => set(["vlm", "alibaba_api_key"], e.target.value)} />
          </Field>
          <Field label="Yibu API Key">
            <input type="password" className={inputCls} value={config.vlm?.yibu_api_key ?? ""} onChange={(e) => set(["vlm", "yibu_api_key"], e.target.value)} />
          </Field>
        </div>
        <Toggle label="Use Yibu API" checked={!!config.vlm?.use_yibu_api} onChange={(v) => set(["vlm", "use_yibu_api"], v)} />
      </Section>

      <Section title="Logs & Retrieval">
        <Field label="Exploration logs root">
          <input className={inputCls} value={config.logs?.root ?? ""} onChange={(e) => set(["logs", "root"], e.target.value)} />
        </Field>
        <Toggle label="Use memory for navigation" checked={!!config.use_memory_for_navigation} onChange={(v) => set(["use_memory_for_navigation"], v)} />
        <div className="grid grid-cols-2 gap-4">
          <Field label="Stage-1 top K">
            <input type="range" min={5} max={50} value={config.top_k_in_first_stage_retrieval ?? 30} onChange={(e) => set(["top_k_in_first_stage_retrieval"], +e.target.value)} className="w-full" />
            <span className="text-xs text-slate-500">{config.top_k_in_first_stage_retrieval ?? 30}</span>
          </Field>
          <Field label="Stage-2 top K">
            <input type="range" min={1} max={10} value={config.top_k_retrieval_in_stage_2 ?? 5} onChange={(e) => set(["top_k_retrieval_in_stage_2"], +e.target.value)} className="w-full" />
            <span className="text-xs text-slate-500">{config.top_k_retrieval_in_stage_2 ?? 5}</span>
          </Field>
        </div>
      </Section>
    </div>
  );
}
