import type { WizardStep } from "../types";
import { WIZARD_STEPS } from "../types";

interface Props {
  current: WizardStep;
  unlocked: Set<WizardStep>;
  onSelect: (step: WizardStep) => void;
}

export default function Sidebar({ current, unlocked, onSelect }: Props) {
  return (
    <aside className="w-56 min-h-screen bg-sidebar text-white flex flex-col shrink-0">
      <div className="p-5 border-b border-slate-700">
        <h1 className="text-lg font-bold tracking-tight">AgentNavigator</h1>
        <p className="text-xs text-slate-400 mt-1">Inference Demo</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {WIZARD_STEPS.map((step, idx) => {
          const active = current === step.id;
          const enabled = unlocked.has(step.id);
          return (
            <button
              key={step.id}
              disabled={!enabled}
              onClick={() => enabled && onSelect(step.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-accent text-white"
                  : enabled
                  ? "text-slate-300 hover:bg-sidebar-hover"
                  : "text-slate-600 cursor-not-allowed"
              }`}
            >
              <span className="w-6 h-6 flex items-center justify-center rounded-full text-xs bg-slate-700/50">
                {idx + 1}
              </span>
              <span>{step.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
