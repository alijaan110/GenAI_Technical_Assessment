import React, { useState, useEffect } from "react";
import axios from "axios";
import {
    Save,
    Loader2,
    CheckCircle2,
    AlertCircle,
    Key,
    Cpu,
    Scissors,
    Database,
    Globe,
    HardDrive,
    RefreshCcw,
} from "lucide-react";

interface SettingsData {
    openai_api_key: string;
    anthropic_api_key: string;
    tavily_api_key: string;
    qdrant_url: string;
    qdrant_api_key: string;
    llm_provider: string;
    llm_model: string;
    chunk_size: string;
    chunk_overlap: string;
    openai_api_key_set: boolean;
    anthropic_api_key_set: boolean;
    tavily_api_key_set: boolean;
    qdrant_api_key_set: boolean;
}

interface Health {
    vector_backend: "qdrant" | "memory" | "unknown";
    qdrant_reachable: boolean;
    qdrant_active: boolean;
    time: string;
}

export default function SettingsPage() {
    const [settings, setSettings] = useState<SettingsData | null>(null);
    const [health, setHealth] = useState<Health | null>(null);
    const [form, setForm] = useState({
        openai_api_key: "",
        anthropic_api_key: "",
        tavily_api_key: "",
        qdrant_url: "",
        qdrant_api_key: "",
        llm_provider: "openai",
        llm_model: "gpt-4o-mini",
        chunk_size: "1200",
        chunk_overlap: "150",
    });
    const [saving, setSaving] = useState(false);
    const [refreshing, setRefreshing] = useState(false);
    const [status, setStatus] = useState<{ type: "success" | "error"; msg: string } | null>(null);

    const refreshSettings = () =>
        axios.get("/api/settings").then((res) => {
            setSettings(res.data);
            setForm((prev) => ({
                ...prev,
                llm_provider: res.data.llm_provider || "openai",
                llm_model: res.data.llm_model || "gpt-4o-mini",
                chunk_size: res.data.chunk_size || "1200",
                chunk_overlap: res.data.chunk_overlap || "150",
                qdrant_url: res.data.qdrant_url || "",
            }));
        });
    const refreshHealth = () =>
        axios
            .get("/api/system/health")
            .then((res) => setHealth(res.data))
            .catch(() => setHealth(null));

    useEffect(() => {
        refreshSettings().catch(console.error);
        refreshHealth();
    }, []);

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        setSaving(true);
        setStatus(null);
        try {
            const payload: any = {
                llm_provider: form.llm_provider,
                llm_model: form.llm_model,
                chunk_size: form.chunk_size,
                chunk_overlap: form.chunk_overlap,
                qdrant_url: form.qdrant_url,
            };
            if (form.openai_api_key) payload.openai_api_key = form.openai_api_key;
            if (form.anthropic_api_key) payload.anthropic_api_key = form.anthropic_api_key;
            if (form.tavily_api_key) payload.tavily_api_key = form.tavily_api_key;
            if (form.qdrant_api_key) payload.qdrant_api_key = form.qdrant_api_key;

            await axios.post("/api/settings", payload);
            setForm((prev) => ({
                ...prev,
                openai_api_key: "",
                anthropic_api_key: "",
                tavily_api_key: "",
                qdrant_api_key: "",
            }));
            await refreshSettings();
            await refreshHealth();
            setStatus({ type: "success", msg: "Settings saved." });
        } catch (e: any) {
            setStatus({ type: "error", msg: e.response?.data?.error || e.message });
        } finally {
            setSaving(false);
        }
    };

    const recheckHealth = async () => {
        setRefreshing(true);
        await refreshHealth();
        setRefreshing(false);
    };

    const MODELS: Record<string, string[]> = {
        openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        anthropic: [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
        ],
    };

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            <div>
                <h2 className="text-2xl font-bold text-text-dark">Settings</h2>
                <p className="text-sm text-text-dark/60 mt-1">
                    API keys, LLM provider, vector backend, and chunking parameters.
                </p>
            </div>

            {/* System health card */}
            <div className="bg-white border border-primary/30 rounded-2xl p-5 shadow-sm">
                <div className="flex items-start justify-between">
                    <div>
                        <h3 className="font-semibold text-base flex items-center gap-2">
                            <HardDrive className="w-4 h-4 text-accent" /> Vector backend
                        </h3>
                        {health ? (
                            <div className="mt-2 flex items-center gap-2 text-sm">
                                {health.vector_backend === "qdrant" ? (
                                    <>
                                        <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded font-semibold uppercase">
                                            Qdrant
                                        </span>
                                        <span className="text-text-dark/70">
                                            connected at {settings?.qdrant_url || "default"}
                                        </span>
                                    </>
                                ) : (
                                    <>
                                        <span className="bg-amber-100 text-amber-700 text-xs px-2 py-0.5 rounded font-semibold uppercase">
                                            In-memory fallback
                                        </span>
                                        <span className="text-text-dark/70">
                                            Qdrant not reachable — using local cosine store. Vectors still
                                            persist in SQLite.
                                        </span>
                                    </>
                                )}
                            </div>
                        ) : (
                            <p className="text-sm text-text-dark/50 mt-2">Probing…</p>
                        )}
                    </div>
                    <button
                        type="button"
                        onClick={recheckHealth}
                        className="text-xs flex items-center gap-1 text-accent hover:underline"
                    >
                        {refreshing ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                            <RefreshCcw className="w-3.5 h-3.5" />
                        )}
                        Re-probe
                    </button>
                </div>
            </div>

            {/* Status */}
            {status && (
                <div
                    className={`flex items-center gap-3 p-3.5 rounded-xl border ${
                        status.type === "success"
                            ? "bg-green-50 border-green-200 text-green-800"
                            : "bg-red-50 border-red-200 text-red-800"
                    }`}
                >
                    {status.type === "success" ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600 shrink-0" />
                    ) : (
                        <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
                    )}
                    <span className="text-sm font-medium">{status.msg}</span>
                </div>
            )}

            <form onSubmit={handleSave} className="space-y-5">
                {/* API keys */}
                <section className="bg-white border border-primary/30 rounded-2xl p-5 shadow-sm space-y-4">
                    <h3 className="font-semibold text-base flex items-center gap-2">
                        <Key className="w-4 h-4 text-accent" /> API keys
                    </h3>

                    <KeyField
                        label="OpenAI API Key"
                        placeholder="sk-…"
                        value={form.openai_api_key}
                        configured={settings?.openai_api_key_set}
                        onChange={(v) => setForm({ ...form, openai_api_key: v })}
                    />
                    <KeyField
                        label="Anthropic API Key"
                        placeholder="sk-ant-…"
                        value={form.anthropic_api_key}
                        configured={settings?.anthropic_api_key_set}
                        onChange={(v) => setForm({ ...form, anthropic_api_key: v })}
                    />
                    <KeyField
                        label="Tavily API Key (web search)"
                        placeholder="tvly-…"
                        value={form.tavily_api_key}
                        configured={settings?.tavily_api_key_set}
                        onChange={(v) => setForm({ ...form, tavily_api_key: v })}
                        helper="Used by Task 3 agent for live web research. Without it the agent gracefully runs RAG-only."
                    />
                </section>

                {/* Vector backend */}
                <section className="bg-white border border-primary/30 rounded-2xl p-5 shadow-sm space-y-4">
                    <h3 className="font-semibold text-base flex items-center gap-2">
                        <Database className="w-4 h-4 text-accent" /> Vector backend (Qdrant)
                    </h3>
                    <div>
                        <label className="block text-sm font-medium mb-1.5">Qdrant URL</label>
                        <input
                            value={form.qdrant_url}
                            onChange={(e) => setForm({ ...form, qdrant_url: e.target.value })}
                            placeholder="http://localhost:6333"
                            className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
                        />
                        <p className="text-[11px] text-text-dark/60 mt-1">
                            Leave blank to default to <code className="text-xs">http://localhost:6333</code>.
                            Run Qdrant locally with{" "}
                            <code className="text-xs">docker run -p 6333:6333 qdrant/qdrant</code>. If
                            unreachable, we fall back to the in-memory store.
                        </p>
                    </div>
                    <KeyField
                        label="Qdrant API Key (cloud only)"
                        placeholder="qdrant-…"
                        value={form.qdrant_api_key}
                        configured={settings?.qdrant_api_key_set}
                        onChange={(v) => setForm({ ...form, qdrant_api_key: v })}
                    />
                </section>

                {/* LLM */}
                <section className="bg-white border border-primary/30 rounded-2xl p-5 shadow-sm space-y-4">
                    <h3 className="font-semibold text-base flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-accent" /> LLM configuration
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-sm font-medium mb-1.5">Provider</label>
                            <select
                                value={form.llm_provider}
                                onChange={(e) => {
                                    const provider = e.target.value;
                                    setForm({
                                        ...form,
                                        llm_provider: provider,
                                        llm_model: MODELS[provider]?.[0] || "gpt-4o-mini",
                                    });
                                }}
                                className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
                            >
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1.5">Model</label>
                            <select
                                value={form.llm_model}
                                onChange={(e) => setForm({ ...form, llm_model: e.target.value })}
                                className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
                            >
                                {(MODELS[form.llm_provider] || MODELS.openai).map((m) => (
                                    <option key={m} value={m}>
                                        {m}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                </section>

                {/* Chunking */}
                <section className="bg-white border border-primary/30 rounded-2xl p-5 shadow-sm space-y-4">
                    <h3 className="font-semibold text-base flex items-center gap-2">
                        <Scissors className="w-4 h-4 text-accent" /> Document chunking
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-sm font-medium mb-1.5">Chunk size (chars)</label>
                            <input
                                type="number"
                                min={200}
                                max={4000}
                                value={form.chunk_size}
                                onChange={(e) => setForm({ ...form, chunk_size: e.target.value })}
                                className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1.5">Chunk overlap (chars)</label>
                            <input
                                type="number"
                                min={0}
                                max={500}
                                value={form.chunk_overlap}
                                onChange={(e) => setForm({ ...form, chunk_overlap: e.target.value })}
                                className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
                            />
                        </div>
                    </div>
                    <p className="text-[11px] text-text-dark/60">
                        Sections are first split on legal boundaries (Article / Section / §), then
                        oversized sections are recursively split with this chunk size and overlap.
                    </p>
                </section>

                <button
                    type="submit"
                    disabled={saving}
                    className="w-full bg-accent hover:bg-accent/90 disabled:bg-accent/50 text-white font-semibold py-3 rounded-xl flex items-center justify-center gap-2 shadow-md transition-all"
                >
                    {saving ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                        <Save className="w-5 h-5" />
                    )}
                    {saving ? "Saving…" : "Save settings"}
                </button>
            </form>

            <p className="text-[11px] text-text-dark/50 flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5" /> All keys are persisted to a local SQLite settings table.
                Environment variables override stored values.
            </p>
        </div>
    );
}

function KeyField({
    label,
    placeholder,
    value,
    configured,
    onChange,
    helper,
}: {
    label: string;
    placeholder: string;
    value: string;
    configured?: boolean;
    onChange: (v: string) => void;
    helper?: string;
}) {
    return (
        <div>
            <label className="block text-sm font-medium mb-1.5">{label}</label>
            <input
                type="password"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={configured ? "•••••••• (already configured)" : placeholder}
                className="w-full border border-primary/30 rounded-lg px-3 py-2 bg-secondary/10 outline-none focus:ring-2 focus:ring-accent text-sm"
            />
            {configured && (
                <p className="text-xs text-green-600 mt-1 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Configured
                </p>
            )}
            {helper && !configured && (
                <p className="text-[11px] text-text-dark/60 mt-1">{helper}</p>
            )}
        </div>
    );
}
