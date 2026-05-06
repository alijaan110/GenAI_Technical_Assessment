import React, { useState, useEffect } from "react";
import axios from "axios";
import {
    Play,
    CheckCircle2,
    AlertCircle,
    Loader2,
    Sparkles,
    Download,
    Plus,
    FileText,
    ChevronDown,
    ChevronRight,
} from "lucide-react";
import Markdown from "../../src/components/Markdown";

interface TestSet {
    test_set_id: string;
    count: number;
    created_at?: string;
}
interface TestQuestion {
    id: string;
    question: string;
    expected_answer: string;
    question_type: string;
    source_doc?: string | null;
    source_page?: number | null;
}
interface EvalRow {
    id: string;
    test_set_id: string;
    status: string;
    overall_faithfulness?: number;
    overall_relevancy?: number;
    overall_precision?: number;
    overall_recall?: number;
    overall_correctness?: number;
    hallucination_rate?: number;
    total_questions?: number;
    passed_questions?: number;
    failed_questions?: number;
    started_at: string;
    completed_at?: string;
}

export default function EvalPage() {
    const [sets, setSets] = useState<TestSet[]>([]);
    const [questions, setQuestions] = useState<TestQuestion[]>([]);
    const [activeSetId, setActiveSetId] = useState("");
    const [evalList, setEvalList] = useState<EvalRow[]>([]);

    const [newQ, setNewQ] = useState("");
    const [newA, setNewA] = useState("");

    const [genBusy, setGenBusy] = useState(false);
    const [genCount, setGenCount] = useState(2);
    const [genMaxChunks, setGenMaxChunks] = useState(8);
    const [toast, setToast] = useState<string | null>(null);

    const showToast = (msg: string) => {
        setToast(msg);
        window.setTimeout(() => setToast((t) => (t === msg ? null : t)), 4000);
    };

    const refreshSets = async () => {
        const sRes = await axios.get("/api/evaluation/test-questions");
        setSets(sRes.data.test_sets || []);
        if (sRes.data.test_sets?.length && !activeSetId) {
            setActiveSetId(sRes.data.test_sets[0].test_set_id);
        }
    };
    const refreshEvals = async () => {
        const eRes = await axios.get("/api/evaluation");
        setEvalList(eRes.data.evaluations || []);
    };
    const refreshQuestions = async (setId: string) => {
        if (!setId) {
            setQuestions([]);
            return;
        }
        const res = await axios.get(`/api/evaluation/test-questions/${setId}`);
        setQuestions(res.data.questions || []);
    };

    useEffect(() => {
        refreshSets().catch(console.error);
        refreshEvals().catch(console.error);
    }, []);

    useEffect(() => {
        refreshQuestions(activeSetId).catch(console.error);
    }, [activeSetId]);

    // Polls every 3s while any evaluation is running.
    useEffect(() => {
        const i = setInterval(() => {
            const anyRunning = evalList.some((e) => e.status === "running");
            if (anyRunning) refreshEvals().catch(console.error);
        }, 3000);
        return () => clearInterval(i);
    }, [evalList]);

    const addQuestion = async () => {
        if (!newQ.trim() || !newA.trim()) return;
        const res = await axios.post("/api/evaluation/test-questions", {
            test_set_id: activeSetId || null,
            questions: [{ question: newQ, expected_answer: newA, question_type: "factual" }],
        });
        setNewQ("");
        setNewA("");
        await refreshSets();
        const newId = activeSetId || res.data.test_set_id;
        setActiveSetId(newId);
        await refreshQuestions(newId);
    };

    const autoGenerate = async () => {
        setGenBusy(true);
        try {
            const res = await axios.post("/api/evaluation/test-questions/auto-generate", {
                n_per_chunk: genCount,
                max_chunks: genMaxChunks,
            });
            await refreshSets();
            setActiveSetId(res.data.test_set_id);
            await refreshQuestions(res.data.test_set_id);
            showToast(
                `✅ Generated ${res.data.question_count} test questions in set ${res.data.test_set_id.slice(0, 8)}…`
            );
        } catch (e: any) {
            showToast(e.response?.data?.error || "Auto-generation failed");
        } finally {
            setGenBusy(false);
        }
    };

    const runEvaluation = async () => {
        if (!activeSetId) return;
        try {
            await axios.post("/api/evaluation/run", { test_set_id: activeSetId });
            await refreshEvals();
            showToast("Evaluation started — results will appear when ready.");
        } catch (e: any) {
            showToast(e.response?.data?.error || "Couldn't start evaluation.");
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 relative">
            {toast && (
                <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 bg-text-dark text-white text-sm px-4 py-2 rounded-lg shadow-lg max-w-[600px]">
                    {toast}
                </div>
            )}
            {/* Test sets + auto-gen + add panel */}
            <aside className="lg:col-span-4 space-y-5">
                <section className="bg-white border border-primary/30 rounded-2xl shadow-sm p-5">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-accent" />
                        Auto-generate from documents
                    </h3>
                    <div className="grid grid-cols-2 gap-3 mb-3">
                        <label className="text-xs">
                            Questions / chunk
                            <input
                                type="number"
                                min={1}
                                max={5}
                                value={genCount}
                                onChange={(e) => setGenCount(parseInt(e.target.value || "1", 10))}
                                className="w-full mt-1 border border-primary/30 rounded px-2 py-1.5 text-sm bg-secondary/10"
                            />
                        </label>
                        <label className="text-xs">
                            Sample chunks
                            <input
                                type="number"
                                min={1}
                                max={50}
                                value={genMaxChunks}
                                onChange={(e) => setGenMaxChunks(parseInt(e.target.value || "1", 10))}
                                className="w-full mt-1 border border-primary/30 rounded px-2 py-1.5 text-sm bg-secondary/10"
                            />
                        </label>
                    </div>
                    <button
                        type="button"
                        onClick={autoGenerate}
                        disabled={genBusy}
                        className="w-full bg-accent hover:bg-accent/90 disabled:bg-accent/50 text-white py-2 rounded-lg flex items-center justify-center gap-2 text-sm font-medium"
                    >
                        {genBusy ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Sparkles className="w-4 h-4" />
                        )}
                        {genBusy ? "Generating…" : "Generate gold-standard set"}
                    </button>
                    <p className="text-[10px] text-text-dark/60 mt-2">
                        LLM derives diverse Q&A pairs (factual, conceptual, cross-reference, edge-case) from
                        ingested chunks.
                    </p>
                </section>

                <section className="bg-white border border-primary/30 rounded-2xl shadow-sm p-5 flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold">Test sets</h3>
                        <span className="text-xs text-text-dark/50">{sets.length} total</span>
                    </div>

                    <select
                        value={activeSetId}
                        onChange={(e) => setActiveSetId(e.target.value)}
                        className="w-full border border-primary/30 p-2 rounded-lg bg-secondary/10 text-sm mb-3"
                    >
                        <option value="">Select a set…</option>
                        {sets.map((s) => (
                            <option key={s.test_set_id} value={s.test_set_id}>
                                {s.test_set_id.slice(0, 8)} · {s.count} Qs
                            </option>
                        ))}
                    </select>

                    <div className="space-y-2 max-h-[40vh] overflow-y-auto pr-1">
                        {questions.map((q, i) => (
                            <div
                                key={q.id}
                                className="bg-secondary/20 border border-primary/20 rounded p-2.5 text-xs"
                            >
                                <p className="font-medium mb-1">
                                    Q{i + 1}: {q.question}
                                </p>
                                <p className="text-text-dark/70 line-clamp-2">A: {q.expected_answer}</p>
                                <div className="mt-1 flex gap-2 text-[10px] text-text-dark/50">
                                    <span className="bg-white px-1.5 rounded border border-primary/20">
                                        {q.question_type}
                                    </span>
                                    {q.source_doc && (
                                        <span>
                                            {q.source_doc} {q.source_page ? `· p.${q.source_page}` : ""}
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                        {!questions.length && (
                            <p className="text-xs text-text-dark/50">No questions in this set yet.</p>
                        )}
                    </div>

                    {activeSetId && questions.length > 0 && (
                        <button
                            type="button"
                            onClick={runEvaluation}
                            className="mt-3 w-full bg-green-600 hover:bg-green-700 text-white py-2.5 rounded-lg flex items-center justify-center gap-2 text-sm font-medium"
                        >
                            <Play className="w-4 h-4" /> Run evaluation
                        </button>
                    )}
                </section>

                <section className="bg-white border border-primary/30 rounded-2xl shadow-sm p-5">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Plus className="w-4 h-4 text-accent" /> Add question manually
                    </h3>
                    <div className="space-y-2">
                        <input
                            value={newQ}
                            onChange={(e) => setNewQ(e.target.value)}
                            placeholder="Question…"
                            className="w-full border border-primary/30 rounded p-2 text-sm bg-secondary/10"
                        />
                        <textarea
                            value={newA}
                            onChange={(e) => setNewA(e.target.value)}
                            placeholder="Expected answer (ground truth)…"
                            className="w-full border border-primary/30 rounded p-2 text-sm bg-secondary/10 min-h-[80px]"
                        />
                        <button
                            type="button"
                            onClick={addQuestion}
                            disabled={!newQ.trim() || !newA.trim()}
                            className="w-full bg-primary/20 hover:bg-primary/30 disabled:opacity-50 text-text-dark py-2 rounded text-sm font-medium"
                        >
                            Add to {activeSetId ? "selected set" : "new set"}
                        </button>
                    </div>
                </section>
            </aside>

            {/* Results */}
            <main className="lg:col-span-8 space-y-4">
                <h2 className="text-lg font-bold flex items-center gap-2">
                    <FileText className="w-5 h-5 text-accent" />
                    Evaluation Runs
                </h2>
                {evalList.length === 0 && (
                    <div className="bg-white border border-primary/30 rounded-2xl p-10 text-center text-text-dark/50">
                        No evaluations yet. Generate a test set and click Run Evaluation.
                    </div>
                )}
                <div className="space-y-3">
                    {evalList.map((ev) => (
                        <EvalCard key={ev.id} ev={ev} onRefresh={refreshEvals} />
                    ))}
                </div>
            </main>
        </div>
    );
}

function EvalCard({ ev, onRefresh }: { ev: EvalRow; onRefresh: () => void }) {
    const [open, setOpen] = useState(false);
    const [details, setDetails] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const toggle = async () => {
        const next = !open;
        setOpen(next);
        if (next && !details) {
            setLoading(true);
            try {
                const res = await axios.get(`/api/evaluation/results/${ev.id}`);
                setDetails(res.data);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        }
    };

    const downloadReport = async () => {
        try {
            const res = await axios.get(`/api/evaluation/results/${ev.id}/report`, {
                responseType: "blob",
            });
            const url = window.URL.createObjectURL(new Blob([res.data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", `eval_report_${ev.id.slice(0, 8)}.md`);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (e) {
            alert("No report available yet");
        }
    };

    const M = (val: number | undefined) =>
        val == null ? "–" : `${(val * 100).toFixed(1)}%`;

    return (
        <div className="bg-white border border-primary/30 rounded-2xl shadow-sm overflow-hidden">
            <div
                onClick={toggle}
                className="p-4 flex items-center justify-between cursor-pointer hover:bg-secondary/10 transition-colors"
            >
                <div className="flex items-center gap-3">
                    {open ? (
                        <ChevronDown className="w-4 h-4 text-text-dark/50" />
                    ) : (
                        <ChevronRight className="w-4 h-4 text-text-dark/50" />
                    )}
                    {ev.status === "running" ? (
                        <Loader2 className="w-5 h-5 text-accent animate-spin" />
                    ) : ev.status === "completed" ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600" />
                    ) : (
                        <AlertCircle className="w-5 h-5 text-red-500" />
                    )}
                    <div>
                        <p className="font-medium text-sm">Run {ev.id.slice(0, 8)}</p>
                        <p className="text-[11px] text-text-dark/60">
                            Set {ev.test_set_id.slice(0, 8)} · {new Date(ev.started_at).toLocaleString()}
                        </p>
                    </div>
                </div>

                <div className="hidden md:grid grid-cols-5 gap-3 text-xs">
                    <Metric label="Faithful" v={ev.overall_faithfulness} />
                    <Metric label="Relevant" v={ev.overall_relevancy} />
                    <Metric label="Precision" v={ev.overall_precision} />
                    <Metric label="Recall" v={ev.overall_recall} />
                    <Metric label="Correct" v={ev.overall_correctness} />
                </div>

                <div className="flex items-center gap-3">
                    {ev.status === "completed" && (
                        <button
                            type="button"
                            onClick={(e) => {
                                e.stopPropagation();
                                downloadReport();
                            }}
                            className="text-xs flex items-center gap-1 text-accent hover:underline"
                        >
                            <Download className="w-3.5 h-3.5" /> Report
                        </button>
                    )}
                    <span className="text-[10px] uppercase font-semibold tracking-wider text-text-dark/50">
                        {ev.status}
                    </span>
                </div>
            </div>

            {open && (
                <div className="border-t border-primary/20 p-4 bg-secondary/10">
                    {loading && <p className="text-sm text-text-dark/60">Loading details…</p>}
                    {details && (
                        <>
                            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-4">
                                <SummaryStat label="Total" v={details.total_questions} />
                                <SummaryStat label="Passed" v={details.passed_questions} />
                                <SummaryStat label="Failed" v={details.failed_questions} />
                                <SummaryStat
                                    label="Hallucination"
                                    v={
                                        details.hallucination_rate != null
                                            ? `${(details.hallucination_rate * 100).toFixed(1)}%`
                                            : "–"
                                    }
                                />
                                <SummaryStat
                                    label="Faithfulness"
                                    v={M(details.overall_faithfulness)}
                                />
                                <SummaryStat
                                    label="Correctness"
                                    v={M(details.overall_correctness)}
                                />
                            </div>

                            <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
                                {details.detailed_results?.map((r: any, i: number) => (
                                    <div
                                        key={i}
                                        className="bg-white border border-primary/20 rounded-lg p-3 text-sm"
                                    >
                                        <p className="font-semibold mb-1.5">Q{i + 1}: {r.question}</p>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-2">
                                            <div className="bg-green-50 border border-green-100 rounded p-2 max-h-60 overflow-y-auto">
                                                <p className="text-[10px] font-bold text-green-700 mb-1">
                                                    Expected
                                                </p>
                                                <Markdown>{r.expected_answer || ""}</Markdown>
                                            </div>
                                            <div className="bg-purple-50 border border-purple-100 rounded p-2 max-h-60 overflow-y-auto">
                                                <p className="text-[10px] font-bold text-purple-700 mb-1">
                                                    Generated
                                                </p>
                                                <Markdown>{r.answer || ""}</Markdown>
                                            </div>
                                        </div>
                                        <div className="flex flex-wrap gap-1.5 text-[10px]">
                                            <Pill label="Faith" v={r.scores.faithfulness} />
                                            <Pill label="Relev" v={r.scores.answer_relevancy} />
                                            <Pill label="Prec" v={r.scores.context_precision} />
                                            <Pill label="Recall" v={r.scores.context_recall} />
                                            <Pill label="Correct" v={r.scores.answer_correctness} />
                                            {r.has_hallucination && (
                                                <span className="bg-red-50 text-red-600 px-1.5 py-0.5 rounded font-semibold border border-red-100">
                                                    hallucination
                                                </span>
                                            )}
                                        </div>
                                        {r.issues?.length > 0 && (
                                            <ul className="mt-1.5 text-[10px] text-text-dark/60 list-disc list-inside">
                                                {r.issues.map((iss: string, k: number) => (
                                                    <li key={k}>{iss}</li>
                                                ))}
                                            </ul>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

function Metric({ label, v }: { label: string; v: number | undefined }) {
    const pct = v != null ? Math.round(v * 100) : null;
    const color =
        pct == null
            ? "text-text-dark/40"
            : pct >= 85
              ? "text-green-600"
              : pct >= 70
                ? "text-amber-600"
                : "text-red-500";
    return (
        <div className="text-center">
            <div className={`font-semibold ${color}`}>{pct == null ? "–" : `${pct}%`}</div>
            <div className="text-[10px] text-text-dark/50 uppercase">{label}</div>
        </div>
    );
}

function SummaryStat({ label, v }: { label: string; v: any }) {
    return (
        <div className="bg-white border border-primary/20 rounded p-2 text-center">
            <div className="text-sm font-semibold">{v ?? "–"}</div>
            <div className="text-[10px] text-text-dark/50 uppercase">{label}</div>
        </div>
    );
}

function Pill({ label, v }: { label: string; v: number | null | undefined }) {
    if (v == null) return null;
    const pct = Math.round(v * 100);
    const color =
        pct >= 85
            ? "bg-green-50 text-green-700 border-green-100"
            : pct >= 70
              ? "bg-amber-50 text-amber-700 border-amber-100"
              : "bg-red-50 text-red-600 border-red-100";
    return (
        <span className={`px-1.5 py-0.5 rounded border font-semibold ${color}`}>
            {label}: {pct}%
        </span>
    );
}
