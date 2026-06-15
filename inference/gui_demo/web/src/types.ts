export interface AppConfig {
  debug_mode?: boolean;
  app?: { name?: string; description?: string };
  driver?: {
    device_id?: string;
    os_name?: string;
    appPackage?: string;
    appActivity?: string;
    skip_exploration_for_no_app_package?: boolean;
    reset_instruction?: string;
    use_launcher_intent?: boolean;
  };
  vlm?: {
    alibaba_api_key?: string;
    yibu_api_key?: string;
    use_yibu_api?: boolean;
  };
  agent?: {
    verbose_print?: boolean;
    url?: string;
    model_name?: string;
    settings?: { history_n?: number; resize_factor?: number };
    max_steps?: number;
  };
  logs?: { root?: string; resume_from_checkpoint?: boolean };
  use_memory_for_navigation?: boolean;
  query?: string;
  top_k_in_first_stage_retrieval?: number;
  top_k_retrieval_in_stage_2?: number;
  pick_candidate_index?: number;
}

export interface ConfigOption {
  id: string;
  label: string;
  filename: string;
}

export interface CheckResult {
  ok: boolean;
  message: string;
  model_id?: string;
}

export interface Candidate {
  node_id: string;
  score: number;
  depth: number;
  page_purpose: string;
  user_intents: string[];
  vlm_reasoning: string;
  screenshot_url: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "step" | "candidates";
  content: string;
  imageB64?: string;
  annotatedB64?: string;
  actionType?: string;
  step?: number;
  candidates?: Candidate[];
  selectedNodeId?: string;
}

export type WizardStep = "configure" | "prepare" | "navigate";

export type PreparePhase = "idle" | "checking" | "connecting" | "loading" | "done" | "error";

export type NavigatePhase = "input" | "retrieving" | "selecting" | "loading_memory" | "ready" | "executing";

export interface LoadItem {
  key: string;
  label: string;
  status: "pending" | "loading" | "done" | "error";
}

export const WIZARD_STEPS: { id: WizardStep; label: string; icon: string }[] = [
  { id: "configure", label: "Configure", icon: "⚙️" },
  { id: "prepare", label: "Prepare", icon: "🚀" },
  { id: "navigate", label: "Navigate", icon: "💬" },
];
