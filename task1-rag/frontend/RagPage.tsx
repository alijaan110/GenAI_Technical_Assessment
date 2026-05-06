import React, { useState, useCallback, useEffect, useRef } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import {
    UploadCloud,
    File as FileIcon,
    Trash,
    Send,
    Loader2,
    Plus,
    MessagesSquare,
    AlertTriangle,
    BookOpen,
    Pencil,
    Check,
    X,
} from "lucide-react";
import Markdown from "../../src/components/Markdown";

interface SessionRow {
    id: string;
    title: string;
    message_count: number;
    updated_at: string;
}
interface CitationSource {
    chunk_id: string;
    document_name: string;
    page: number | null;
    section: string;
    sub_section: string;
    excerpt: string;
    relevance_score: number;
    dense_score: number | null;
    sparse_score: number | null;
}
interface ChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    sources: CitationSource[] | null;
    is_grounded: number | null;
    retrieval_score: number | null;
    created_at: string;
}

const ACTIVE_SESSION_KEY = "lexai:activeSession";

export default function RagPage() {
    const [documents, setDocuments] = useState<any[]>([]);
    const [sessions, setSessions] = useState<SessionRow[]>([]);
    // Hydrate activeSession from localStorage so a hard refresh keeps the
    // user in the same conversation rather than dropping them on a blank chat.
    const [activeSession, setActiveSessionState] = useState<string | null>(() => {
        if (typeof window === "undefined") return null;
        return window.localStorage.getItem(ACTIVE_SESSION_KEY);
    });
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [query, setQuery] = useState("");
    const [busy, setBusy] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [toast, setToast] = useState<string | null>(null);
    const [renamingSession, setRenamingSession] = useState<string | null>(null);
    const [renameValue, setRenameValue] = useState("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Always mirror activeSession to localStorage so the next mount picks it up.
    const setActiveSession = (id: string | null) => {
        setActiveSessionState(id);
        try {
            if (id) window.localStorage.setItem(ACTIVE_SESSION_KEY, id);
            else window.localStorage.removeItem(ACTIVE_SESSION_KEY);
        } catch {
            /* localStorage may be disabled — non-fatal */
        }
    };

    const showToast = (msg: string) => {
        setToast(msg);
        window.setTimeout(() => setToast((t) => (t === msg ? null : t)), 4000);
    };

    // ── Loaders ───────────────────────────────────────────────────────
    const refreshDocs = async () => {
        try {
            const res = await axios.get("/api/rag/documents");
            setDocuments(res.data.documents || []);
        } catch (e) {
            console.error(e);
        }
    };
    const refreshSessions = async (): Promise<SessionRow[]> => {
        try {
            const res = await axios.get("/api/sessions");
            const list: SessionRow[] = res.data.sessions || [];
            setSessions(list);
            return list;
        } catch (e) {
            console.error(e);
            return [];
        }
    };
    const loadSession = async (id: string) => {
        try {
            const res = await axios.get(`/api/sessions/${id}`);
            setActiveSession(id);
            setMessages(res.data.messages || []);
        } catch (e) {
            console.error(e);
            showToast("Couldn't load that conversation.");
        }
    };

    useEffect(() => {
        // On mount: load docs + sessions, then either restore the persisted
        // session, fall back to the most recent one, or create a fresh one
        // so the user lands in a usable chat immediately.
        (async () => {
            await refreshDocs();
            const list = await refreshSessions();
            const persisted = activeSession;
            if (persisted && list.some((s) => s.id === persisted)) {
                await loadSession(persisted);
            } else if (list.length > 0) {
                await loadSession(list[0].id);
            } else {
                try {
                    const r = await axios.post("/api/sessions", { title: "New chat" });
                    setActiveSession(r.data.session.id);
                    setMessages([]);
                    await refreshSessions();
                } catch (e) {
                    console.error(e);
                }
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, busy]);

    // ── Sessions ──────────────────────────────────────────────────────
    const startNewSession = async () => {
        try {
            const res = await axios.post("/api/sessions", { title: "New chat" });
            setActiveSession(res.data.session.id);
            setMessages([]);
            await refreshSessions();
        } catch (e) {
            console.error(e);
            showToast("Couldn't start a new chat — is the backend running?");
        }
    };

    const deleteSession = async (id: string) => {
        try {
            await axios.delete(`/api/sessions/${id}`);
            if (activeSession === id) {
                // Pick the next-most-recent session so the user is never
                // dropped onto an empty screen mid-conversation.
                const remaining = sessions.filter((s) => s.id !== id);
                if (remaining.length > 0) {
                    await loadSession(remaining[0].id);
                } else {
                    setActiveSession(null);
                    setMessages([]);
                }
            }
            await refreshSessions();
        } catch (e) {
            console.error(e);
            showToast("Couldn't delete that conversation.");
        }
    };

    const beginRename = (s: SessionRow, e: React.MouseEvent) => {
        e.stopPropagation();
        setRenamingSession(s.id);
        setRenameValue(s.title);
    };

    const commitRename = async (id: string) => {
        try {
            await axios.patch(`/api/sessions/${id}`, { title: renameValue.trim() || "Untitled" });
            setRenamingSession(null);
            await refreshSessions();
        } catch (e) {
            console.error(e);
        }
    };

    // ── Documents ─────────────────────────────────────────────────────
    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        setUploading(true);
        try {
            const res = await axios.post("/api/rag/upload", formData);
            await refreshDocs();
            showToast(
                `Indexed ${res.data.total_chunks} chunks from ${file.name} (${res.data.backend} backend).`
            );
        } catch (e: any) {
            const msg = e.response?.data?.error || e.message || "Upload failed";
            // Use a non-blocking toast instead of alert() — alert() blurs the
            // tab and was a likely contributor to the "page refreshes" feel.
            showToast(`Upload failed: ${msg}`);
        } finally {
            setUploading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { "application/pdf": [".pdf"] },
        multiple: false,
    });

    const deleteDoc = async (id: string) => {
        try {
            await axios.delete(`/api/rag/documents/${id}`);
            await refreshDocs();
        } catch (e) {
            console.error(e);
        }
    };

    // ── Chat (token-by-token streaming via SSE) ───────────────────────
    const send = async (e?: React.FormEvent) => {
        // Defensively prevent any default form submission to guarantee no
        // navigation / page refresh from the chat box.
        if (e && typeof e.preventDefault === "function") e.preventDefault();
        if (e && typeof (e as any).stopPropagation === "function") (e as any).stopPropagation();
        const q = query.trim();
        if (!q) return;

        // Reuse the existing active session — only create a new one if the
        // user is genuinely starting from scratch. Each prompt MUST stay
        // inside the same conversation so the sidebar reads like ChatGPT.
        let sid = activeSession;
        if (!sid) {
            try {
                const r = await axios.post("/api/sessions", { title: q.slice(0, 60) });
                sid = r.data.session.id;
                setActiveSession(sid);
            } catch (err: any) {
                showToast("Couldn't create a session — is the backend running?");
                return;
            }
        }

        const userMsg: ChatMessage = {
            id: `tmp-user-${Date.now()}`,
            role: "user",
            content: q,
            sources: null,
            is_grounded: null,
            retrieval_score: null,
            created_at: new Date().toISOString(),
        };
        const aiId = `tmp-ai-${Date.now()}`;
        const aiMsg: ChatMessage = {
            id: aiId,
            role: "assistant",
            content: "",
            sources: null,
            is_grounded: null,
            retrieval_score: null,
            created_at: new Date().toISOString(),
        };
        setMessages((m) => [...m, userMsg, aiMsg]);
        setQuery("");
        setBusy(true);

        try {
            const resp = await fetch("/api/rag/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
                body: JSON.stringify({
                    question: q,
                    top_k: 8,
                    use_hybrid: true,
                    session_id: sid,
                }),
            });
            if (!resp.ok || !resp.body) {
                const txt = await resp.text();
                throw new Error(txt || `HTTP ${resp.status}`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            // Apply each parsed SSE event to the in-flight assistant message.
            const apply = (event: string, payload: any) => {
                if (event === "sources") {
                    setMessages((m) =>
                        m.map((msg) => (msg.id === aiId ? { ...msg, sources: payload } : msg))
                    );
                } else if (event === "token") {
                    const piece = typeof payload === "string" ? payload : String(payload);
                    setMessages((m) =>
                        m.map((msg) =>
                            msg.id === aiId ? { ...msg, content: msg.content + piece } : msg
                        )
                    );
                } else if (event === "done") {
                    setMessages((m) =>
                        m.map((msg) =>
                            msg.id === aiId
                                ? {
                                      ...msg,
                                      is_grounded: payload?.is_grounded ? 1 : 0,
                                      retrieval_score: payload?.retrieval_score ?? null,
                                  }
                                : msg
                        )
                    );
                } else if (event === "error") {
                    setMessages((m) =>
                        m.map((msg) =>
                            msg.id === aiId
                                ? {
                                      ...msg,
                                      content:
                                          (msg.content || "") +
                                          `\n\n**Error:** ${payload?.message || "stream failed"}`,
                                  }
                                : msg
                        )
                    );
                }
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                // SSE events are separated by a blank line.
                let idx;
                while ((idx = buffer.indexOf("\n\n")) >= 0) {
                    const block = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    let event = "message";
                    const dataLines: string[] = [];
                    for (const line of block.split("\n")) {
                        if (line.startsWith("event:")) event = line.slice(6).trim();
                        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
                    }
                    if (!dataLines.length) continue;
                    const raw = dataLines.join("\n");
                    let payload: any = raw;
                    try {
                        payload = JSON.parse(raw);
                    } catch {
                        /* keep as string for non-JSON events */
                    }
                    apply(event, payload);
                }
            }

            await refreshSessions();
        } catch (err: any) {
            setMessages((m) =>
                m.map((msg) =>
                    msg.id === aiId
                        ? {
                              ...msg,
                              content:
                                  (msg.content || "") +
                                  `\n\n**Error:** ${err?.message || "Request failed"}`,
                              is_grounded: 0,
                          }
                        : msg
                )
            );
        } finally {
            setBusy(false);
        }
    };

    // ── Render ────────────────────────────────────────────────────────
    return (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-130px)] relative">
            {toast && (
                <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 bg-text-dark text-white text-sm px-4 py-2 rounded-lg shadow-lg animate-fade-in max-w-[600px]">
                    {toast}
                </div>
            )}
            {/* Sessions sidebar */}
            <aside className="lg:col-span-2 bg-white border border-primary/30 rounded-2xl shadow-sm flex flex-col overflow-hidden">
                <div className="p-3 border-b border-primary/20">
                    <button
                        type="button"
                        onClick={startNewSession}
                        className="w-full flex items-center justify-center gap-2 bg-accent text-white py-2 rounded-lg hover:bg-accent/90 transition-colors text-sm font-medium"
                    >
                        <Plus className="w-4 h-4" /> New chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {sessions.length === 0 && (
                        <p className="text-xs text-text-dark/50 p-3">No conversations yet.</p>
                    )}
                    {sessions.map((s) => (
                        <div
                            key={s.id}
                            onClick={() => renamingSession !== s.id && loadSession(s.id)}
                            className={`group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                                activeSession === s.id
                                    ? "bg-accent/10 text-accent"
                                    : "hover:bg-secondary/40"
                            }`}
                        >
                            <div className="flex items-center gap-2 min-w-0 flex-1">
                                <MessagesSquare className="w-4 h-4 shrink-0" />
                                {renamingSession === s.id ? (
                                    <input
                                        autoFocus
                                        value={renameValue}
                                        onChange={(e) => setRenameValue(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter") commitRename(s.id);
                                            if (e.key === "Escape") setRenamingSession(null);
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                        className="flex-1 min-w-0 bg-white border border-accent/40 rounded px-2 py-0.5 text-xs"
                                    />
                                ) : (
                                    <span className="truncate">{s.title}</span>
                                )}
                            </div>
                            <div className="hidden group-hover:flex items-center gap-1">
                                {renamingSession === s.id ? (
                                    <>
                                        <button
                                            type="button"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                commitRename(s.id);
                                            }}
                                            className="text-green-600 hover:text-green-700"
                                        >
                                            <Check className="w-3.5 h-3.5" />
                                        </button>
                                        <button
                                            type="button"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setRenamingSession(null);
                                            }}
                                            className="text-text-dark/60 hover:text-text-dark"
                                        >
                                            <X className="w-3.5 h-3.5" />
                                        </button>
                                    </>
                                ) : (
                                    <>
                                        <button
                                            type="button"
                                            onClick={(e) => beginRename(s, e)}
                                            className="text-text-dark/50 hover:text-accent"
                                        >
                                            <Pencil className="w-3.5 h-3.5" />
                                        </button>
                                        <button
                                            type="button"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                // Synchronous confirm() blurs the tab and is itself a
                                                // common cause of "the page reloaded" complaints.
                                                // Inline-confirm by checking for shift to skip prompt.
                                                if (e.shiftKey || window.confirm("Delete this conversation?")) {
                                                    deleteSession(s.id);
                                                }
                                            }}
                                            className="text-text-dark/50 hover:text-red-500"
                                        >
                                            <Trash className="w-3.5 h-3.5" />
                                        </button>
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </aside>

            {/* Chat panel */}
            <section className="lg:col-span-7 bg-white border border-primary/30 rounded-2xl shadow-sm flex flex-col overflow-hidden">
                <div className="px-5 py-3 border-b border-primary/20 flex items-center justify-between">
                    <h2 className="font-semibold flex items-center gap-2">
                        <BookOpen className="w-4 h-4 text-accent" />
                        {activeSession
                            ? sessions.find((s) => s.id === activeSession)?.title || "Chat"
                            : "Legal RAG Chat"}
                    </h2>
                    {documents.length === 0 && (
                        <span className="text-xs text-orange-600 flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5" /> Upload a PDF to start
                        </span>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.length === 0 && !busy && (
                        <div className="h-full flex flex-col items-center justify-center text-text-dark/50 gap-3">
                            <BookOpen className="w-12 h-12 opacity-50" />
                            <p className="text-sm max-w-md text-center">
                                Ask anything about your uploaded legal documents. Every answer is grounded in the
                                source PDFs with inline citations.
                            </p>
                        </div>
                    )}
                    {messages.map((m) => (
                        <MessageBubble key={m.id} m={m} />
                    ))}
                    {busy && (
                        <div className="flex items-center gap-2 text-sm text-accent">
                            <Loader2 className="w-4 h-4 animate-spin" /> LexAI is searching documents…
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={send} className="p-4 border-t border-primary/20 bg-secondary/10">
                    <div className="relative max-w-3xl mx-auto">
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    send();
                                }
                            }}
                            placeholder="Ask a legal question (e.g. 'What does Article 17 GDPR require?')"
                            className="w-full bg-white border border-primary/30 rounded-xl pl-4 pr-14 py-3 min-h-[60px] max-h-[200px] resize-y focus:ring-2 focus:ring-accent focus:border-accent outline-none text-sm"
                        />
                        <button
                            type="submit"
                            disabled={busy || !query.trim()}
                            className="absolute right-3 bottom-3 bg-accent hover:bg-accent/90 disabled:bg-accent/40 text-white p-2 rounded-lg transition-colors"
                        >
                            {busy ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                        </button>
                    </div>
                </form>
            </section>

            {/* Documents panel */}
            <aside className="lg:col-span-3 flex flex-col gap-4 overflow-hidden">
                <div className="bg-white border border-primary/30 rounded-2xl p-4 shadow-sm">
                    <h3 className="font-semibold text-sm mb-3">Upload PDF</h3>
                    <div
                        {...getRootProps()}
                        className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-colors ${
                            isDragActive
                                ? "border-accent bg-accent/5"
                                : "border-primary/30 hover:bg-secondary/30"
                        }`}
                    >
                        <input {...getInputProps()} />
                        <UploadCloud className="w-8 h-8 text-accent mx-auto mb-2" />
                        <p className="text-xs font-medium">
                            {uploading ? "Uploading & ingesting…" : "Drop a PDF or click to choose"}
                        </p>
                        <p className="text-[10px] text-text-dark/60 mt-1">
                            Parsed → hierarchical chunked → dense + BM25 indexed
                        </p>
                    </div>
                </div>

                <div className="bg-white border border-primary/30 rounded-2xl p-4 shadow-sm flex-1 overflow-hidden flex flex-col">
                    <h3 className="font-semibold text-sm mb-3">Document Library</h3>
                    <div className="space-y-2 overflow-y-auto flex-1">
                        {documents.length === 0 && (
                            <p className="text-xs text-text-dark/50">No documents uploaded yet.</p>
                        )}
                        {documents.map((d) => (
                            <div
                                key={d.id}
                                className="flex flex-col bg-secondary/20 border border-primary/20 rounded-lg p-2.5"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="flex items-center gap-2 min-w-0">
                                        <FileIcon className="w-4 h-4 text-accent shrink-0" />
                                        <p className="text-xs font-medium truncate">{d.filename}</p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => deleteDoc(d.id)}
                                        className="text-text-dark/50 hover:text-red-500"
                                    >
                                        <Trash className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                                <div className="mt-1 flex items-center justify-between text-[10px] text-text-dark/60">
                                    <span>
                                        {d.total_pages || "?"} pages · {d.total_chunks || 0} chunks ·{" "}
                                        {d.doc_type || "doc"}
                                    </span>
                                    <span
                                        className={`font-medium ${
                                            d.status === "completed"
                                                ? "text-green-600"
                                                : d.status === "failed"
                                                  ? "text-red-500"
                                                  : "text-orange-500"
                                        }`}
                                    >
                                        {d.status}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>
        </div>
    );
}

function MessageBubble({ m }: { m: ChatMessage }) {
    const isUser = m.role === "user";
    if (isUser) {
        return (
            <div className="flex justify-end">
                <div className="max-w-[80%] bg-accent text-white rounded-2xl rounded-br-sm px-4 py-2.5 shadow-sm">
                    <p className="text-sm whitespace-pre-wrap">{m.content}</p>
                </div>
            </div>
        );
    }
    const grounded = m.is_grounded === 1;
    return (
        <div className="space-y-3">
            <div className="flex justify-start">
                <div className="max-w-[85%] bg-secondary/30 border border-primary/20 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                    <div className="flex items-center gap-2 mb-2 text-[10px] uppercase tracking-wider text-text-dark/60">
                        <span>LexAI</span>
                        {grounded ? (
                            <span className="text-green-600 font-semibold">grounded</span>
                        ) : (
                            <span className="text-orange-600 font-semibold">unsourced</span>
                        )}
                        {m.retrieval_score != null && (
                            <span>· retrieval {m.retrieval_score.toFixed(3)}</span>
                        )}
                    </div>
                    <Markdown>{m.content}</Markdown>
                </div>
            </div>
        </div>
    );
}
