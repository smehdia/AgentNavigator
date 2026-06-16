import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { Candidate, ChatMessage, NavigatePhase } from "../types";

interface Props {
  messages: ChatMessage[];
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
  sending: boolean;
  phase: NavigatePhase;
  onSelectCandidate: (nodeId: string) => void;
  onExecute: () => void;
  onStop: () => void;
  selectingNodeId: string | null;
  resetToStartPage: boolean;
  onResetToStartPageChange: (v: boolean) => void;
}

function CandidateCards({
  candidates,
  selectedId,
  onSelect,
  disabled,
}: {
  candidates: Candidate[];
  selectedId: string | null;
  onSelect: (nodeId: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
      {candidates.map((c, idx) => {
        const selected = selectedId === c.node_id;
        return (
          <button
            key={c.node_id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(c.node_id)}
            className={`text-left rounded-xl border-2 overflow-hidden transition-all ${
              disabled ? "opacity-60 cursor-not-allowed" : "hover:shadow-md"
            } ${selected ? "border-accent ring-2 ring-accent/20" : "border-slate-200 bg-white"}`}
          >
            <div className="relative aspect-[9/16] max-h-48 bg-slate-100">
              <img src={c.screenshot_url} alt={c.node_id} className="w-full h-full object-contain" />
              <div className="absolute top-1.5 left-1.5 flex gap-1">
                <span className="px-1.5 py-0.5 bg-black/60 text-white text-[10px] rounded-full">#{idx + 1}</span>
                <span className="px-1.5 py-0.5 bg-accent/90 text-white text-[10px] rounded-full">
                  {c.score.toFixed(3)}
                </span>
              </div>
            </div>
            <div className="p-2">
              <p className="text-xs font-medium line-clamp-2 text-slate-800">
                {c.page_label || c.page_purpose}
              </p>
              {c.vlm_reasoning && (
                <p className="text-[10px] text-slate-500 mt-1 line-clamp-2 italic">{c.vlm_reasoning}</p>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`flex items-center justify-between gap-4 w-full px-3 py-2 rounded-lg border transition-colors ${
        disabled ? "opacity-60 cursor-not-allowed" : ""
      } ${checked ? "border-accent bg-indigo-50" : "border-slate-200 bg-white"}`}
    >
      <span className="text-xs font-medium text-left text-slate-700">{label}</span>
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

export default function ChatPanel({
  messages,
  input,
  onInputChange,
  onSend,
  sending,
  phase,
  onSelectCandidate,
  onExecute,
  onStop,
  selectingNodeId,
  resetToStartPage,
  onResetToStartPageChange,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (phase === "input" && !sending) onSend();
    }
  };

  const inputDisabled = phase !== "input" || sending;
  const showExecute = phase === "ready";
  const showStop = phase === "executing";

  return (
    <div className="flex flex-col h-full min-h-[400px] bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-3 pt-3 pb-2 border-b border-slate-100 shrink-0">
        <Toggle
          label="reset_to_start_page"
          checked={resetToStartPage}
          onChange={onResetToStartPageChange}
          disabled={phase === "executing"}
        />
        <p className="text-[10px] text-slate-400 mt-1 px-1">
          When on, resets the app to the start screen before executing navigation.
        </p>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-400 py-12 text-sm">
            Type a navigation goal — retrieval, selection, memory, and execution all happen here.
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
          >
            <div
              className={`max-w-[95%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-accent text-white max-w-[85%]"
                  : msg.role === "step"
                  ? "bg-slate-50 border border-slate-200"
                  : msg.role === "candidates"
                  ? "bg-slate-50 border border-slate-200 w-full max-w-full"
                  : "bg-slate-100 text-slate-800"
              }`}
            >
              {msg.role === "step" && msg.step != null && (
                <div className="text-xs font-semibold text-accent mb-1">Step {msg.step}</div>
              )}
              {msg.actionType && (
                <div className="text-xs font-mono text-slate-500 mb-1">Action: {msg.actionType}</div>
              )}
              {msg.content && (
                <div className="text-sm prose prose-sm max-w-none prose-p:my-1">
                  {msg.role === "assistant" || msg.role === "candidates" ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>
              )}
              {msg.role === "candidates" && msg.candidates && (
                <CandidateCards
                  candidates={msg.candidates}
                  selectedId={msg.selectedNodeId ?? selectingNodeId}
                  onSelect={onSelectCandidate}
                  disabled={phase !== "selecting"}
                />
              )}
              {(msg.annotatedB64 || msg.imageB64) && (
                <img
                  src={`data:image/jpeg;base64,${msg.annotatedB64 || msg.imageB64}`}
                  alt="Step screenshot"
                  className="mt-3 rounded-lg max-h-64 object-contain border border-slate-200"
                />
              )}
            </div>
          </div>
        ))}
        {phase === "retrieving" && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-slate-100 rounded-2xl px-4 py-3 flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-slate-600">Running retrieval & VLM rerank…</span>
            </div>
          </div>
        )}
        {phase === "loading_memory" && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-slate-100 rounded-2xl px-4 py-3 flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-slate-600">Loading navigation memory…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-100 p-3 flex gap-2 items-end">
        {showStop ? (
          <button
            onClick={onStop}
            className="flex-1 px-4 py-2.5 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-500"
          >
            Stop Navigation
          </button>
        ) : showExecute ? (
          <>
            <textarea
              value={input}
              disabled
              rows={1}
              className="flex-1 px-3 py-2 rounded-lg border border-slate-100 bg-slate-50 text-sm text-slate-400 resize-none"
              placeholder="Select a destination above, then execute"
            />
            <button
              onClick={onExecute}
              className="px-5 py-2.5 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-500 shrink-0"
            >
              Execute
            </button>
          </>
        ) : (
          <>
            <textarea
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={handleKey}
              placeholder={
                phase === "selecting"
                  ? "Pick a candidate above…"
                  : "Describe your navigation goal…"
              }
              rows={2}
              disabled={inputDisabled}
              className="flex-1 px-3 py-2 rounded-lg border border-slate-200 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:bg-slate-50 disabled:text-slate-400"
            />
            <button
              onClick={onSend}
              disabled={inputDisabled || !input.trim()}
              className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50 self-end"
            >
              {sending ? "…" : "Send"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
