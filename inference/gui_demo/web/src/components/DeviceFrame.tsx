interface Props {
  imageSrc: string | null;
  connected: boolean;
  connecting: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}

export default function DeviceFrame({ imageSrc, connected, connecting, onConnect, onDisconnect }: Props) {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Device Preview</h2>
          <p className="text-sm text-slate-500 mt-1">Live screenshot stream from connected device</p>
        </div>
        <div className="flex gap-2">
          {!connected ? (
            <button
              onClick={onConnect}
              disabled={connecting}
              className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light disabled:opacity-50"
            >
              {connecting ? "Connecting…" : "Connect to Device"}
            </button>
          ) : (
            <button
              onClick={onDisconnect}
              className="px-5 py-2.5 bg-slate-200 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-300"
            >
              Disconnect
            </button>
          )}
        </div>
      </div>

      <div className="flex justify-center">
        <div className="relative bg-slate-900 rounded-[2.5rem] p-3 shadow-2xl border-4 border-slate-800">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-24 h-6 bg-slate-900 rounded-b-2xl z-10" />
          <div className="w-[280px] h-[560px] bg-slate-800 rounded-[2rem] overflow-hidden flex items-center justify-center">
            {imageSrc ? (
              <img src={imageSrc} alt="Device screenshot" className="w-full h-full object-contain" />
            ) : (
              <div className="text-slate-500 text-sm text-center px-6">
                {connected ? "Waiting for frames…" : "Connect to see live preview"}
              </div>
            )}
          </div>
          {connected && (
            <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-emerald-600 font-medium">Live</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
