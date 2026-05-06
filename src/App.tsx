import React, { useEffect, useState } from "react";
import { Scale, Database, FlaskConical, Bot, Settings } from "lucide-react";
import RagPage from "../task1-rag/frontend/RagPage";
import EvalPage from "../task2-rag-eval/frontend/EvalPage";
import AgentPage from "../task3-agent/frontend/AgentPage";
import SettingsPage from "./pages/SettingsPage";

const TABS = [
    { id: "rag", label: "RAG Pipeline", icon: Database, desc: "Task 1" },
    { id: "eval", label: "Evaluation", icon: FlaskConical, desc: "Task 2" },
    { id: "agent", label: "Agent", icon: Bot, desc: "Task 3" },
    { id: "settings", label: "Settings", icon: Settings, desc: "" },
] as const;

type TabId = (typeof TABS)[number]["id"];
const ACTIVE_TAB_KEY = "lexai:activeTab";

/**
 * Top-level error boundary so a render error in one panel never wipes out
 * the whole app + bounces the user to a fresh tab. Silent recover beats
 * a blank screen any day.
 */
class TabErrorBoundary extends React.Component<
    { children: React.ReactNode; tabId: string },
    { error: Error | null }
> {
    state = { error: null as Error | null };
    static getDerivedStateFromError(error: Error) {
        return { error };
    }
    componentDidCatch(error: Error, info: React.ErrorInfo) {
        // eslint-disable-next-line no-console
        console.error(`[ErrorBoundary] tab=${this.props.tabId}`, error, info);
    }
    componentDidUpdate(prev: Readonly<{ tabId: string }>) {
        // Reset when the user navigates to a different tab so the next
        // panel doesn't inherit a stale error.
        if (prev.tabId !== this.props.tabId && this.state.error) {
            this.setState({ error: null });
        }
    }
    render() {
        if (this.state.error) {
            return (
                <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 max-w-2xl mx-auto mt-10">
                    <h2 className="font-semibold text-red-800 mb-2">Something went wrong on this tab.</h2>
                    <p className="text-sm text-red-700 mb-3">
                        Switch to another tab and back — your data is safe.
                    </p>
                    <pre className="text-xs text-red-600 bg-white border border-red-200 rounded p-2 overflow-auto">
                        {String(this.state.error?.message || this.state.error)}
                    </pre>
                </div>
            );
        }
        return this.props.children as React.ReactElement;
    }
}

export default function App() {
    // Tab persists across full reloads — avoids the "started a Task 2 run,
    // page hopped back to Task 1" complaint when anything causes a remount.
    const [activeTab, setActiveTabState] = useState<TabId>(() => {
        if (typeof window === "undefined") return "rag";
        const saved = window.localStorage.getItem(ACTIVE_TAB_KEY) as TabId | null;
        return saved && TABS.some((t) => t.id === saved) ? saved : "rag";
    });

    const setActiveTab = (id: TabId) => {
        setActiveTabState(id);
        try {
            window.localStorage.setItem(ACTIVE_TAB_KEY, id);
        } catch {
            /* storage may be disabled */
        }
    };

    // Catch any genuine page-level navigation attempt and log it so we know.
    useEffect(() => {
        const onBeforeUnload = (e: BeforeUnloadEvent) => {
            // eslint-disable-next-line no-console
            console.debug("[App] beforeunload fired — investigate stack");
            // Do NOT block the unload; just trace it.
        };
        window.addEventListener("beforeunload", onBeforeUnload);
        return () => window.removeEventListener("beforeunload", onBeforeUnload);
    }, []);

    return (
        <div className="min-h-screen bg-bg-light">
            <header className="bg-primary-dark text-white shadow-lg">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="bg-accent p-2 rounded-xl">
                            <Scale className="w-6 h-6" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold tracking-tight">LegalRAG</h1>
                            <p className="text-xs text-white/60">
                                GenAI Assessment — Legal Document Intelligence
                            </p>
                        </div>
                    </div>

                    <nav className="flex space-x-1 bg-white/10 rounded-xl p-1">
                        {TABS.map((tab) => {
                            const Icon = tab.icon;
                            const active = activeTab === tab.id;
                            return (
                                <button
                                    key={tab.id}
                                    type="button"
                                    id={`tab-${tab.id}`}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                                        active
                                            ? "bg-accent text-white shadow-md"
                                            : "text-white/70 hover:text-white hover:bg-white/10"
                                    }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="hidden sm:inline">{tab.label}</span>
                                    {tab.desc && (
                                        <span
                                            className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                                                active ? "bg-white/20" : "bg-white/10"
                                            }`}
                                        >
                                            {tab.desc}
                                        </span>
                                    )}
                                </button>
                            );
                        })}
                    </nav>
                </div>
            </header>

            {/* No `key={activeTab}` here — that was forcing a full remount of
                the active panel on every tab change, which threw away in-flight
                state (uploads, polls, partial chats). Each panel can manage
                its own lifecycle just fine. */}
            <main className="max-w-7xl mx-auto px-6 py-8">
                <TabErrorBoundary tabId={activeTab}>
                    <div style={{ display: activeTab === "rag" ? "block" : "none" }}>
                        <RagPage />
                    </div>
                    <div style={{ display: activeTab === "eval" ? "block" : "none" }}>
                        <EvalPage />
                    </div>
                    <div style={{ display: activeTab === "agent" ? "block" : "none" }}>
                        <AgentPage />
                    </div>
                    <div style={{ display: activeTab === "settings" ? "block" : "none" }}>
                        <SettingsPage />
                    </div>
                </TabErrorBoundary>
            </main>
        </div>
    );
}
