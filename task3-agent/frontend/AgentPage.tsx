import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
    Send,
    Loader2,
    Bot,
    Layers,
    FileText,
    HelpCircle,
    CheckCircle2,
    AlertTriangle,
    Trash2,
} from "lucide-react";
import Markdown from "../../src/components/Markdown";

interface AgentRun {
    id: string;
    query: string;
    status: string;
    current_step?: string;
    output_format?: string;
    search_strategy?: string;
    needs_clarification?: number;
    clarification_question?: string;
    started_at: string;
    completed_at?: string;
}
interface AgentStep {
    step_name: string;
    status: string;
    result_summary?: any;
    error?: string;
    started_at?: string;
    completed_at?: string;
}
interface AgentStatus extends AgentRun {
    user_clarification?: string;
    summary?: string;
    final_output?: string;
    error_log?: string;
    steps: AgentStep[];
}

export default function AgentPage() {
    const [query, setQuery] = useState("");
    const [enableHitl, setEnableHitl] = useState(true);
    const [runs, setRuns] = useState<AgentRun[]>([]);
    const [activeRunId, setActiveRunId] = useState<string | null>(null);
    const [status, setStatus] = useState<AgentStatus | null>(null);
    const [submitting, setSubmitting] = useState(false);
    const [clarificationReply, setClarificationReply] = useState("");
    const [resuming, setResuming] = useState(false);
    const pollRef = useRef<any>(null);
    const stepsEndRef = useRef<HTMLDivElement>(null);

    const refreshHistory = async () => {
        try {
            const res = await axios.get("/api/agent/history");
            setRuns(res.data.runs || []);
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        refreshHistory();
    }, []);

    // Polling: while a run isn't terminal we poll for steps + status.
    useEffect(() => {
        if (!activeRunId) return;
        const poll = async () => {
            try {
                const sRes = await axios.get(`/api/agent/status/${activeRunId}`);
                let out: AgentStatus = sRes.data;
                if (out.status === "completed") {
                    const r = await axios.get(`/api/agent/result/${activeRunId}`);
                    out = { ...out, ...r.data };
                }
                setStatus(out);
                if (out.status === "completed" || out.status === "failed") {
                    if (pollRef.current) clearInterval(pollRef.current);
                    refreshHistory();
                }
            } catch (e) {
                console.error(e);
            }
        };
        poll();
        pollRef.current = setInterval(poll, 1500);
        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, [activeRunId]);

    // Auto-scroll to bottom of steps when new steps appear
    useEffect(() => {
        if (stepsEndRef.current) {
            stepsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [status?.steps?.length, status?.final_output]);

    const submit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;
        setSubmitting(true);
        setStatus(null);
        try {
            const res = await axios.post("/api/agent/execute", {
                query,
                enable_hitl: enableHitl,
            });
            setActiveRunId(res.data.run_id);
            setQuery("");
            refreshHistory();
        } catch (e: any) {
            alert(e.response?.data?.error || "Failed to start agent");
        } finally {
            setSubmitting(false);
        }
    };

    const resume = async () => {
        if (!activeRunId) return;
        setResuming(true);
        try {
            await axios.post(`/api/agent/resume/${activeRunId}`, {
                user_clarification: clarificationReply,
            });
            setClarificationReply("");
        } catch (e: any) {
            alert(e.response?.data?.error || "Failed to resume agent");
        } finally {
            setResuming(false);
        }
    };

    const loadRun = async (id: string) => {
        setActiveRunId(id);
        // Optimistically fetch status + result in one shot so the panel
        // renders immediately rather than waiting for the next 1.5 s poll.
        try {
            const sRes = await axios.get(`/api/agent/status/${id}`);
            let out: AgentStatus = sRes.data;
            if (out.status === "completed") {
                const r = await axios.get(`/api/agent/result/${id}`);
                out = { ...out, ...r.data };
            }
            setStatus(out);
        } catch (e) {
            console.error(e);
            setStatus(null);
        }
    };

    const deleteRun = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            await axios.delete(`/api/agent/runs/${id}`);
            if (activeRunId === id) {
                setActiveRunId(null);
                setStatus(null);
            }
            refreshHistory();
        } catch (err) {
            console.error(err);
        }
    };

    const awaitingClarification = status?.status === "awaiting_clarification";

    return (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-130px)]">
            {/* Run history */}
            <aside className="lg:col-span-3 bg-white border border-primary/30 rounded-2xl shadow-sm flex flex-col overflow-hidden">
                <div className="p-3 border-b border-primary/20">
                    <h3 className="font-semibold flex items-center gap-2 text-sm">
                        <Layers className="w-4 h-4 text-accent" /> Run history
                    </h3>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
                    {runs.length === 0 && (
                        <p className="text-xs text-text-dark/50 p-3">No runs yet.</p>
                    )}
                    {runs.map((r) => (
                        <div
                            key={r.id}
                            onClick={() => loadRun(r.id)}
                            className={`w-full text-left p-2.5 rounded-lg border text-xs transition-colors cursor-pointer group relative ${activeRunId === r.id
                                ? "border-accent bg-accent/10"
                                : "border-primary/20 hover:bg-secondary/30"
                                }`}
                        >
                            <button
                                type="button"
                                onClick={(e) => deleteRun(r.id, e)}
                                className="absolute top-1.5 right-1.5 p-1 rounded hover:bg-red-100 text-red-400 hover:text-red-600"
                                title="Delete run"
                            >
                                <Trash2 className="w-3 h-3" />
                            </button>
                            <p className="font-medium line-clamp-2 pr-5">{r.query}</p>
                            <div className="mt-1 flex items-center justify-between text-[10px] text-text-dark/60 uppercase">
                                <StatusBadge status={r.status} />
                                <span>{new Date(r.started_at).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </aside>

            {/* Main */}
            <main className="lg:col-span-9 flex flex-col bg-white border border-primary/30 rounded-2xl shadow-sm overflow-hidden">
                <div className="flex-1 overflow-y-auto bg-bg-light p-6">
                    {!status && !submitting && (
                        <div className="h-full flex flex-col items-center justify-center text-text-dark/50 max-w-lg mx-auto text-center gap-3">
                            <Bot className="w-14 h-14 opacity-50 text-accent" />
                            <h2 className="text-lg font-semibold">Multi-step legal research agent</h2>
                            <p className="text-sm">
                                The agent will analyze your query, route through internal RAG and the live web,
                                ask for clarification if something's ambiguous, summarize, and produce a structured
                                checklist or report.
                            </p>
                        </div>
                    )}

                    {status && (
                        <div className="space-y-6 max-w-3xl mx-auto">
                            {/* Goal */}
                            <div className="bg-white border border-primary/20 rounded-xl p-4 shadow-sm">
                                <p className="text-[10px] uppercase font-bold text-accent mb-1">Goal</p>
                                <p className="text-sm font-medium">{status.query}</p>
                                <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
                                    {status.output_format && (
                                        <Tag>{status.output_format}</Tag>
                                    )}
                                    {status.search_strategy && (
                                        <Tag>{status.search_strategy}</Tag>
                                    )}
                                    {status.needs_clarification ? <Tag tone="warn">HITL</Tag> : null}
                                    <Tag tone="muted">status: {status.status}</Tag>
                                </div>
                            </div>

                            {/* HITL panel */}
                            {awaitingClarification && (
                                <div className="bg-amber-50 border-2 border-amber-200 rounded-xl p-4 shadow-sm">
                                    <div className="flex items-center gap-2 mb-2 text-amber-800">
                                        <HelpCircle className="w-5 h-5" />
                                        <h3 className="font-semibold">Agent needs your clarification</h3>
                                    </div>
                                    <p className="text-sm text-amber-900 mb-3">
                                        {status.clarification_question || "Please provide more details."}
                                    </p>
                                    <textarea
                                        value={clarificationReply}
                                        onChange={(e) => setClarificationReply(e.target.value)}
                                        placeholder="Your clarification…"
                                        className="w-full bg-white border border-amber-300 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-amber-300 outline-none min-h-[70px]"
                                    />
                                    <button
                                        type="button"
                                        onClick={resume}
                                        disabled={resuming || !clarificationReply.trim()}
                                        className="mt-2 bg-amber-600 hover:bg-amber-700 disabled:bg-amber-400 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
                                    >
                                        {resuming ? (
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                        ) : (
                                            <Send className="w-4 h-4" />
                                        )}
                                        Resume agent
                                    </button>
                                </div>
                            )}

                            {/* Steps */}
                            <div className="space-y-3">
                                <p className="text-[10px] uppercase font-bold text-text-dark/60 tracking-wider px-1">
                                    Execution steps
                                </p>
                                {status.steps?.map((step, idx) => (
                                    <StepCard key={idx} step={step} idx={idx + 1} last={idx === status.steps.length - 1} runStatus={status.status} />
                                ))}
                                <div ref={stepsEndRef} />
                            </div>

                            {/* Final output */}
                            {status.final_output ? (
                                <div className="bg-white border-2 border-accent/30 rounded-2xl p-6 shadow-sm">
                                    <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                                        <FileText className="w-5 h-5 text-accent" />
                                        Final {status.output_format || "Output"}
                                    </h3>
                                    <Markdown>{status.final_output}</Markdown>
                                </div>
                            ) : status.status === "failed" ? (
                                <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-5">
                                    <h3 className="font-semibold text-red-800 mb-2">
                                        This run did not finish
                                    </h3>
                                    <p className="text-sm text-red-700">
                                        {status.error_log
                                            ? status.error_log
                                            : "The agent stopped before producing a final output. Check the failed step above for details, then click Send to start a fresh run."}
                                    </p>
                                </div>
                            ) : status.status === "completed" ? (
                                <div className="bg-amber-50 border-2 border-amber-200 rounded-2xl p-5">
                                    <h3 className="font-semibold text-amber-800 mb-2">
                                        No final output saved for this run
                                    </h3>
                                    <p className="text-sm text-amber-700">
                                        The run was completed earlier but the final output isn't stored
                                        (this happens to runs from previous backend versions). Send the
                                        same query again to regenerate it.
                                    </p>
                                </div>
                            ) : null}

                            {/* Summary preview (intermediate brief) */}
                            {status.summary && status.status !== "completed" && (
                                <div className="bg-secondary/20 border border-primary/20 rounded-xl p-4">
                                    <p className="text-[10px] uppercase font-bold text-text-dark/60 mb-1">
                                        Intermediate brief
                                    </p>
                                    <Markdown>{status.summary}</Markdown>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Input */}
                <div className="p-4 border-t border-primary/20 bg-white">
                    <form onSubmit={submit} className="max-w-3xl mx-auto">
                        <div className="flex items-center gap-2 mb-2 text-xs">
                            <label className="flex items-center gap-1.5 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={enableHitl}
                                    onChange={(e) => setEnableHitl(e.target.checked)}
                                />
                                <span className="text-text-dark/70">
                                    Enable human-in-the-loop (clarify if ambiguous)
                                </span>
                            </label>
                        </div>
                        <div className="relative">
                            <textarea
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                disabled={submitting || awaitingClarification}
                                placeholder="e.g. 'Research GDPR rules for EU companies and produce a compliance checklist.'"
                                className="w-full bg-secondary/10 border border-primary/30 rounded-xl pl-4 pr-14 py-3 min-h-[60px] max-h-[200px] resize-y focus:ring-2 focus:ring-accent outline-none text-sm"
                            />
                            <button
                                type="submit"
                                disabled={submitting || !query.trim() || awaitingClarification}
                                className="absolute right-3 bottom-3 bg-accent hover:bg-accent/90 disabled:bg-accent/40 text-white p-2 rounded-lg"
                            >
                                {submitting ? (
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                ) : (
                                    <Send className="w-5 h-5" />
                                )}
                            </button>
                        </div>
                    </form>
                </div>
            </main>
        </div>
    );
}

function StepCard({
    step,
    idx,
    last,
    runStatus,
}: {
    step: AgentStep;
    idx: number;
    last: boolean;
    runStatus?: string;
}) {
    // Determine display status:
    // - If run is awaiting_clarification and step is 'running' → show as 'waiting' (HITL pause)
    // - If run is terminal (failed/completed) and step is 'running' → show as 'failed' (stale)
    // - Otherwise → use actual step status
    const effectiveStatus =
        step.status === "running" && runStatus === "awaiting_clarification"
            ? "waiting"
            : step.status === "running" && (runStatus === "failed" || runStatus === "completed")
                ? "failed"
                : step.status;
    const isRunning = effectiveStatus === "running";
    const isWaiting = effectiveStatus === "waiting";
    const isFailed = effectiveStatus === "failed";
    const isDone = effectiveStatus === "completed";

    const dotClass = isFailed
        ? "bg-red-500"
        : isDone
            ? "bg-green-500"
            : isWaiting
                ? "bg-amber-400 animate-pulse"
                : "bg-accent animate-pulse";
    const icon = isFailed ? "!" : isDone ? "✓" : idx;

    return (
        <div className="flex gap-3 items-start">
            <div className="flex flex-col items-center mt-1">
                <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-xs text-white font-semibold ${dotClass}`}
                >
                    {isRunning || isWaiting ? <Loader2 className="w-4 h-4 animate-spin" /> : icon}
                </div>
                {!last && <div className="w-0.5 flex-1 bg-primary/20 my-1 min-h-[20px]" />}
            </div>
            <div className="flex-1 bg-white border border-primary/20 rounded-xl p-3 shadow-sm">
                <div className="flex items-center justify-between mb-1.5">
                    <h4 className="font-semibold text-sm text-accent">{step.step_name}</h4>
                    <div className="flex items-center gap-2">
                        {step.started_at && step.completed_at && (
                            <span className="text-[10px] text-text-dark/50">
                                {((new Date(step.completed_at).getTime() - new Date(step.started_at).getTime()) / 1000).toFixed(1)}s
                            </span>
                        )}
                        <span
                            className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded ${isFailed
                                ? "bg-red-50 text-red-600"
                                : isDone
                                    ? "bg-green-50 text-green-700"
                                    : "bg-amber-50 text-amber-700"
                                }`}
                        >
                            {effectiveStatus}
                        </span>
                    </div>
                </div>
                {step.error && (
                    <p className="text-xs text-red-600 mb-1">⚠ {step.error}</p>
                )}
                {step.result_summary && (
                    <pre className="text-[11px] text-text-dark/70 bg-secondary/10 border border-primary/10 rounded p-2 whitespace-pre-wrap font-mono overflow-hidden">
                        {Object.entries(step.result_summary)
                            .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
                            .join("\n")}
                    </pre>
                )}
            </div>
        </div>
    );
}

function StatusBadge({ status }: { status: string }) {
    if (status === "completed")
        return (
            <span className="text-green-600 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> done
            </span>
        );
    if (status === "failed")
        return (
            <span className="text-red-500 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> failed
            </span>
        );
    if (status === "awaiting_clarification")
        return (
            <span className="text-amber-600 flex items-center gap-1">
                <HelpCircle className="w-3 h-3" /> needs input
            </span>
        );
    return (
        <span className="text-blue-600 flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" /> running
        </span>
    );
}

function Tag({
    children,
    tone = "default",
}: {
    children: React.ReactNode;
    tone?: "default" | "warn" | "muted";
}) {
    const cls =
        tone === "warn"
            ? "bg-amber-100 text-amber-800 border-amber-200"
            : tone === "muted"
                ? "bg-secondary/40 text-text-dark/70 border-primary/20"
                : "bg-accent/10 text-accent border-accent/20";
    return (
        <span className={`px-1.5 py-0.5 rounded border font-semibold uppercase ${cls}`}>
            {children}
        </span>
    );
}
