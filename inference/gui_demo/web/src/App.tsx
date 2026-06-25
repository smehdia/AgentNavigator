import { useCallback, useEffect, useRef, useState } from "react";
import {
  connectDevice,
  disconnectDevice,
  executeStart,
  executeStop,
  fetchDefaultConfig,
  fetchConfigList,
  fetchNavigationMemory,
  retrieve,
  runChecks,
  saveConfig,
  selectConfig,
  selectNode,
  subscribeLoadProgress,
  wsUrl,
} from "./api";
import ChatPanel from "./components/ChatPanel";
import ConfigForm from "./components/ConfigForm";
import PreparePanel from "./components/PreparePanel";
import Sidebar from "./components/Sidebar";
import type { AppConfig, ChatMessage, CheckResult, ConfigOption, LoadItem, NavigatePhase, PreparePhase, WizardStep } from "./types";
import { WIZARD_STEPS } from "./types";

function uid() {
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [step, setStep] = useState<WizardStep>("configure");
  const [config, setConfig] = useState<AppConfig>({});
  const [configOptions, setConfigOptions] = useState<ConfigOption[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [configListError, setConfigListError] = useState<string | null>(null);
  const [selectingConfig, setSelectingConfig] = useState(false);
  const [configSaved, setConfigSaved] = useState(false);
  const [saving, setSaving] = useState(false);

  const [agentCheck, setAgentCheck] = useState<CheckResult | null>(null);
  const [deviceCheck, setDeviceCheck] = useState<CheckResult | null>(null);
  const [preparePhase, setPreparePhase] = useState<PreparePhase>("idle");
  const [prepareError, setPrepareError] = useState<string | null>(null);

  const [deviceConnected, setDeviceConnected] = useState(false);
  const [deviceFrame, setDeviceFrame] = useState<string | null>(null);
  const deviceWsRef = useRef<WebSocket | null>(null);

  const [loadItems, setLoadItems] = useState<LoadItem[]>([]);
  const [resourcesLoaded, setResourcesLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [navPhase, setNavPhase] = useState<NavigatePhase>("input");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState("");
  const [candidatesMsgId, setCandidatesMsgId] = useState<string | null>(null);
  const [resetToStartPage, setResetToStartPage] = useState(true);
  const [modelThinking, setModelThinking] = useState(true);

  const execWsRef = useRef<WebSocket | null>(null);

  const unlocked = new Set<WizardStep>(["configure"]);
  if (configSaved) unlocked.add("prepare");
  if (resourcesLoaded) unlocked.add("navigate");

  const resetWizardAfterConfigChange = useCallback(() => {
    deviceWsRef.current?.close();
    execWsRef.current?.close();
    setConfigSaved(false);
    setStep("configure");
    setAgentCheck(null);
    setDeviceCheck(null);
    setPreparePhase("idle");
    setPrepareError(null);
    setDeviceConnected(false);
    setDeviceFrame(null);
    setLoadItems([]);
    setResourcesLoaded(false);
    setLoadError(null);
    setChatMessages([]);
    setChatInput("");
    setNavPhase("input");
    setSelectedNodeId(null);
    setCurrentQuery("");
    setCandidatesMsgId(null);
  }, []);

  useEffect(() => {
    setModelThinking(config.agent?.settings?.model_thinking ?? true);
  }, [config.agent?.settings?.model_thinking]);

  useEffect(() => {
    fetchConfigList()
      .then((configList) => {
        setConfigOptions(configList.configs);
        setConfigListError(null);
        if (configList.activeConfigId) {
          setSelectedConfigId((current) => current || configList.activeConfigId);
        }
      })
      .catch((e) => {
        console.error(e);
        setConfigListError(String(e));
      });

    fetchDefaultConfig()
      .then((defaultConfig) => {
        setConfig(defaultConfig.config);
        if (defaultConfig.configId) {
          setSelectedConfigId((current) => current || defaultConfig.configId);
        }
      })
      .catch((e) => {
        console.error(e);
        setConfigListError((current) => current ?? String(e));
      });
  }, []);

  const handleSelectConfig = async (configId: string) => {
    if (!configId || configId === selectedConfigId) return;

    const previousConfigId = selectedConfigId;
    const previousConfig = config;
    setSelectedConfigId(configId);
    setSelectingConfig(true);

    try {
      const result = await selectConfig(configId);
      setConfig(result.config);
      setSelectedConfigId(result.configId);
      resetWizardAfterConfigChange();
    } catch (e) {
      setSelectedConfigId(previousConfigId);
      setConfig(previousConfig);
      alert(String(e));
    } finally {
      setSelectingConfig(false);
    }
  };

  const startDeviceStream = useCallback(() => {
    deviceWsRef.current?.close();
    const ws = new WebSocket(wsUrl("/ws/device"));
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "frame" && data.image_b64) {
        setDeviceFrame(`data:image/jpeg;base64,${data.image_b64}`);
      }
    };
    ws.onclose = () => {
      if (deviceConnected) setTimeout(startDeviceStream, 2000);
    };
    deviceWsRef.current = ws;
  }, [deviceConnected]);

  useEffect(() => {
    if (deviceConnected) startDeviceStream();
    return () => deviceWsRef.current?.close();
  }, [deviceConnected, startDeviceStream]);

  const handleSaveConfig = async () => {
    setSaving(true);
    try {
      await saveConfig(config);
      setConfigSaved(true);
      setStep("prepare");
    } catch (e) {
      alert(String(e));
    } finally {
      setSaving(false);
    }
  };

  const loadResources = useCallback((): Promise<void> => {
    setLoadError(null);
    setLoadItems([]);

    return new Promise((resolve, reject) => {
      const unsubscribe = subscribeLoadProgress(
        (key, label, status) => {
          setLoadItems((prev) => {
            const existing = prev.find((i) => i.key === key);
            if (existing) {
              return prev.map((i) => (i.key === key ? { ...i, label, status: status as LoadItem["status"] } : i));
            }
            return [...prev, { key, label, status: status as LoadItem["status"] }];
          });
        },
        () => {
          unsubscribe();
          setResourcesLoaded(true);
          resolve();
        },
        (msg) => {
          unsubscribe();
          setLoadError(msg);
          reject(new Error(msg));
        }
      );
    });
  }, []);

  const handleStartPrepare = async () => {
    setPrepareError(null);
    setPreparePhase("checking");

    try {
      const result = await runChecks();
      setAgentCheck(result.agent);
      setDeviceCheck(result.device);
      if (!result.agent.ok || !result.device.ok) {
        setPreparePhase("error");
        setPrepareError("Pre-flight checks failed. Fix the agent server or device connection and try again.");
        return;
      }
    } catch (e) {
      setPreparePhase("error");
      setPrepareError(String(e));
      return;
    }

    setPreparePhase("connecting");
    try {
      await connectDevice();
      setDeviceConnected(true);
    } catch (e) {
      setPreparePhase("error");
      setPrepareError(String(e));
      return;
    }

    setPreparePhase("loading");
    try {
      await loadResources();
      setPreparePhase("done");
      setStep("navigate");
    } catch (e) {
      setPreparePhase("error");
      setPrepareError(String(e));
    }
  };

  const handleDisconnect = async () => {
    deviceWsRef.current?.close();
    await disconnectDevice();
    setDeviceConnected(false);
    setDeviceFrame(null);
    setResourcesLoaded(false);
    setPreparePhase("idle");
  };

  const handleSendPrompt = async () => {
    const query = chatInput.trim();
    if (!query || navPhase !== "input") return;

    setCurrentQuery(query);
    setSelectedNodeId(null);
    setCandidatesMsgId(null);
    setChatMessages((m) => [...m, { id: uid(), role: "user", content: query }]);
    setChatInput("");
    setNavPhase("retrieving");

    try {
      if (config.use_memory_for_navigation) {
        const result = await retrieve(query);
        if (result.candidates.length === 0) {
          setChatMessages((m) => [
            ...m,
            {
              id: uid(),
              role: "assistant",
              content: "No retrieval candidates found. Click **Execute** to navigate without memory.",
            },
          ]);
          setNavPhase("ready");
        } else {
          const msgId = uid();
          setCandidatesMsgId(msgId);
          setChatMessages((m) => [
            ...m,
            {
              id: msgId,
              role: "candidates",
              content: `Found **${result.candidates.length}** candidate pages. Tap the best match:`,
              candidates: result.candidates,
            },
          ]);
          setNavPhase("selecting");
        }
      } else {
        setChatMessages((m) => [
          ...m,
          {
            id: uid(),
            role: "assistant",
            content: "Memory navigation is off. Click **Execute** to start on-device navigation.",
          },
        ]);
        setNavPhase("ready");
      }
    } catch (e) {
      setChatMessages((m) => [...m, { id: uid(), role: "assistant", content: `Error: ${e}` }]);
      setNavPhase("input");
    }
  };

  const handleSelectCandidate = async (nodeId: string) => {
    if (navPhase !== "selecting") return;

    setSelectedNodeId(nodeId);
    if (candidatesMsgId) {
      setChatMessages((m) =>
        m.map((msg) => (msg.id === candidatesMsgId ? { ...msg, selectedNodeId: nodeId } : msg))
      );
    }

    setNavPhase("loading_memory");
    try {
      await selectNode(nodeId);
      const { memory } = await fetchNavigationMemory(nodeId, currentQuery);
      setChatMessages((m) => [
        ...m,
        { id: uid(), role: "assistant", content: `**Navigation memory for \`${nodeId}\`:**\n\n${memory}` },
      ]);
      setNavPhase("ready");
    } catch (e) {
      setChatMessages((m) => [...m, { id: uid(), role: "assistant", content: `Error loading memory: ${e}` }]);
      setNavPhase("selecting");
    }
  };

  const handleExecute = async () => {
    if (navPhase !== "ready") return;

    setNavPhase("executing");
    const resetMsg = resetToStartPage
      ? "Starting on-device navigation (resetting to start page first)…"
      : "Starting on-device navigation from current screen…";
    const thinkingMsg = modelThinking
      ? "Agent thinking mode is on."
      : "Agent thinking mode is off (tool_call only).";
    setChatMessages((m) => [
      ...m,
      { id: uid(), role: "system", content: resetMsg },
      { id: uid(), role: "system", content: thinkingMsg },
    ]);

    try {
      await executeStart(currentQuery, selectedNodeId ?? undefined, resetToStartPage, modelThinking);

      execWsRef.current?.close();
      const ws = new WebSocket(wsUrl("/ws/execution"));

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "step") {
          setChatMessages((m) => [
            ...m,
            {
              id: uid(),
              role: "step",
              step: data.step,
              content: data.thought || "",
              actionType: data.action_type,
              imageB64: data.screenshot_b64,
              annotatedB64: data.annotated_b64,
              timing: {
                driver_screenshot_s: data.driver_screenshot_s,
                model_prediction_s: data.model_prediction_s,
                other_processing_s: data.other_processing_s,
              },
            },
          ]);
          if (data.annotated_b64) {
            setDeviceFrame(`data:image/jpeg;base64,${data.annotated_b64}`);
          }
        } else if (data.type === "stopped") {
          setChatMessages((m) => [
            ...m,
            {
              id: uid(),
              role: "assistant",
              content: `Navigation stopped after ${data.total_steps} step(s).${data.trajectory_mp4 ? ` Trajectory saved to \`${data.trajectory_mp4}\`.` : ""} Send a new prompt to start again.`,
            },
          ]);
          setNavPhase("input");
          ws.close();
        } else if (data.type === "done") {
          setChatMessages((m) => [
            ...m,
            {
              id: uid(),
              role: "assistant",
              content: data.success
                ? `Navigation finished successfully in ${data.total_steps} steps.${data.trajectory_mp4 ? ` Trajectory saved to \`${data.trajectory_mp4}\`.` : ""} Send a new prompt to run again.`
                : `Max steps reached (${data.total_steps}).${data.trajectory_mp4 ? ` Trajectory saved to \`${data.trajectory_mp4}\`.` : ""} Send a new prompt to try again.`,
            },
          ]);
          setNavPhase("input");
        } else if (data.type === "error") {
          setChatMessages((m) => [...m, { id: uid(), role: "assistant", content: `Error: ${data.message}` }]);
          setNavPhase("input");
        }
      };

      ws.onerror = () => setNavPhase("input");
      ws.onclose = () => {
        setNavPhase((p) => (p === "executing" ? "input" : p));
      };

      execWsRef.current = ws;
    } catch (e) {
      setChatMessages((m) => [...m, { id: uid(), role: "assistant", content: `Error: ${e}` }]);
      setNavPhase("input");
    }
  };

  const handleStop = async () => {
    try {
      await executeStop();
      setChatMessages((m) => [
        ...m,
        { id: uid(), role: "system", content: "Stop requested — halting after current step…" },
      ]);
    } catch (e) {
      setChatMessages((m) => [...m, { id: uid(), role: "assistant", content: `Stop failed: ${e}` }]);
    }
  };

  const renderStep = () => {
    switch (step) {
      case "configure":
        return (
          <ConfigForm
            config={config}
            configOptions={configOptions}
            selectedConfigId={selectedConfigId}
            configListError={configListError}
            onSelectConfig={handleSelectConfig}
            selectingConfig={selectingConfig}
            onChange={setConfig}
            onSave={handleSaveConfig}
            saving={saving}
          />
        );
      case "prepare":
        return (
          <PreparePanel
            agent={agentCheck}
            device={deviceCheck}
            imageSrc={deviceFrame}
            deviceConnected={deviceConnected}
            loadItems={loadItems}
            loadError={loadError}
            resourcesLoaded={resourcesLoaded}
            phase={preparePhase}
            error={prepareError}
            onStart={handleStartPrepare}
            onDisconnect={handleDisconnect}
          />
        );
      case "navigate":
        return (
          <div className="h-[calc(100vh-8rem)] flex flex-col lg:grid lg:grid-cols-2 gap-4">
            <div className="min-h-0 flex-1">
              <ChatPanel
                messages={chatMessages}
                input={chatInput}
                onInputChange={setChatInput}
                onSend={handleSendPrompt}
                sending={navPhase === "retrieving"}
                phase={navPhase}
                onSelectCandidate={handleSelectCandidate}
                onExecute={handleExecute}
                onStop={handleStop}
                selectingNodeId={selectedNodeId}
                resetToStartPage={resetToStartPage}
                onResetToStartPageChange={setResetToStartPage}
                modelThinking={modelThinking}
                onModelThinkingChange={setModelThinking}
              />
            </div>
            <div className="hidden lg:flex items-center justify-center bg-slate-100 rounded-xl border border-slate-200 overflow-hidden min-h-[280px]">
              {deviceFrame ? (
                <img src={deviceFrame} alt="Live device" className="max-h-full object-contain" />
              ) : (
                <p className="text-slate-400 text-sm px-6 text-center">Device preview appears here during execution</p>
              )}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar current={step} unlocked={unlocked} onSelect={setStep} />
      <main className="flex-1 flex flex-col min-w-0">
        <header className="px-8 py-4 border-b border-slate-200 bg-white shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold capitalize">{step}</h2>
              <p className="text-xs text-slate-400">
                Step {WIZARD_STEPS.findIndex((s) => s.id === step) + 1} of {WIZARD_STEPS.length}
              </p>
            </div>
            <div className="flex gap-2">
              {agentCheck?.ok && (
                <span className="px-2 py-1 text-xs rounded-full bg-emerald-100 text-emerald-700">Agent ✓</span>
              )}
              {deviceCheck?.ok && (
                <span className="px-2 py-1 text-xs rounded-full bg-emerald-100 text-emerald-700">Device ✓</span>
              )}
              {resourcesLoaded && (
                <span className="px-2 py-1 text-xs rounded-full bg-indigo-100 text-indigo-700">Loaded ✓</span>
              )}
            </div>
          </div>
        </header>
        <div className="flex-1 p-8 overflow-y-auto">{renderStep()}</div>
        <footer className="px-8 py-3 border-t border-slate-100 bg-white shrink-0 flex justify-between">
          <button
            disabled={WIZARD_STEPS.findIndex((s) => s.id === step) === 0}
            onClick={() => {
              const idx = WIZARD_STEPS.findIndex((s) => s.id === step);
              if (idx > 0) setStep(WIZARD_STEPS[idx - 1].id);
            }}
            className="text-sm text-slate-500 hover:text-slate-800 disabled:opacity-30"
          >
            ← Back
          </button>
          <button
            disabled={
              WIZARD_STEPS.findIndex((s) => s.id === step) >= WIZARD_STEPS.length - 1 ||
              !unlocked.has(WIZARD_STEPS[WIZARD_STEPS.findIndex((s) => s.id === step) + 1]?.id)
            }
            onClick={() => {
              const idx = WIZARD_STEPS.findIndex((s) => s.id === step);
              if (idx < WIZARD_STEPS.length - 1) setStep(WIZARD_STEPS[idx + 1].id);
            }}
            className="text-sm text-accent hover:text-accent-light disabled:opacity-30 font-medium"
          >
            Next →
          </button>
        </footer>
      </main>
    </div>
  );
}
