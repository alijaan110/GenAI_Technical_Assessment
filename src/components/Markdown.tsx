import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Shared Markdown renderer with Claude/ChatGPT-style typography.
 * Used everywhere we display LLM output: chat answers, eval rows,
 * agent final report. Tailwind classes are scoped to this component
 * so we can keep the rest of the app token-driven.
 */
export default function Markdown({ children }: { children: string }) {
    return (
        <div className="lex-md max-w-none text-sm leading-relaxed text-text-dark">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    h1: (p) => <h1 className="text-lg font-bold mt-3 mb-2" {...p} />,
                    h2: (p) => <h2 className="text-base font-bold mt-3 mb-2" {...p} />,
                    h3: (p) => (
                        <h3 className="text-sm font-bold uppercase tracking-wider text-accent mt-4 mb-1.5" {...p} />
                    ),
                    h4: (p) => <h4 className="text-sm font-semibold mt-3 mb-1" {...p} />,
                    p: (p) => <p className="mb-2 last:mb-0" {...p} />,
                    ul: (p) => <ul className="list-disc list-outside ml-5 mb-2 space-y-0.5" {...p} />,
                    ol: (p) => <ol className="list-decimal list-outside ml-5 mb-2 space-y-0.5" {...p} />,
                    li: (p) => <li className="leading-relaxed" {...p} />,
                    strong: (p) => <strong className="font-semibold text-text-dark" {...p} />,
                    em: (p) => <em className="italic" {...p} />,
                    blockquote: (p) => (
                        <blockquote
                            className="border-l-4 border-accent/40 pl-3 py-0.5 my-2 italic text-text-dark/80 bg-secondary/20 rounded-r"
                            {...p}
                        />
                    ),
                    code: (props: any) => {
                        const { inline, children, ...rest } = props;
                        if (inline) {
                            return (
                                <code
                                    className="bg-secondary/40 text-accent px-1 py-0.5 rounded text-[0.85em] font-mono"
                                    {...rest}
                                >
                                    {children}
                                </code>
                            );
                        }
                        return (
                            <code
                                className="block bg-primary-dark/95 text-text-light p-3 rounded-lg my-2 overflow-x-auto text-[0.85em] font-mono whitespace-pre-wrap"
                                {...rest}
                            >
                                {children}
                            </code>
                        );
                    },
                    pre: ({ children }) => <>{children}</>,
                    a: (p) => (
                        <a
                            className="text-accent underline hover:text-accent/80"
                            target="_blank"
                            rel="noreferrer"
                            {...p}
                        />
                    ),
                    table: (p) => (
                        <div className="overflow-x-auto my-2">
                            <table
                                className="min-w-full border-collapse text-xs border border-primary/20"
                                {...p}
                            />
                        </div>
                    ),
                    thead: (p) => <thead className="bg-secondary/40" {...p} />,
                    th: (p) => (
                        <th className="border border-primary/20 px-2 py-1 text-left font-semibold" {...p} />
                    ),
                    td: (p) => <td className="border border-primary/20 px-2 py-1 align-top" {...p} />,
                    hr: () => <hr className="my-3 border-primary/20" />,
                    input: (p: any) => {
                        // Render task-list items ("- [ ] thing") as real checkboxes.
                        if (p.type === "checkbox") {
                            return (
                                <input
                                    type="checkbox"
                                    disabled
                                    checked={!!p.checked}
                                    className="mr-1.5 align-middle accent-accent"
                                />
                            );
                        }
                        return <input {...p} />;
                    },
                }}
            >
                {children}
            </ReactMarkdown>
        </div>
    );
}
