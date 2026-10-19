import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { GitBranch } from "../icons";
import Composer from "./Composer";
import { Spinner } from "./ui";
import type { ChatMessage, Repo } from "../types";

const EXAMPLES = [
  "What does this repo do overall?",
  "Where is configuration loaded?",
  "Summarize the main entry point.",
  "How is error handling done?",
];

interface ChatPanelProps {
  messages: ChatMessage[];
  busy: boolean;
  hasRepos: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
  repos: Repo[];
  scope: string | null;
  onScopeChange: (id: string | null) => void;
  onOpenSources: () => void;
}

function fmtMs(ms?: number): string | null {
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function MetricsFooter({
  message,
  onOpenSources,
}: {
  message: ChatMessage;
  onOpenSources: () => void;
}) {
  const { metrics, sources } = message;
  if (!metrics || metrics.empty) return null;
  const parts = [
    ["retrieval", fmtMs(metrics.retrieval_ms)],
    ["first token", fmtMs(metrics.first_token_ms)],
    ["total", fmtMs(metrics.total_ms)],
  ].filter(([, v]) => v != null) as [string, string][];

  return (
    <>
      {metrics.citations && metrics.citations.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {metrics.citations.map((c) => (
            <span
              key={c}
              className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[11px] text-zinc-600"
            >
              {c}
            </span>
          ))}
        </div>
      )}
      <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-zinc-100 pt-3 text-[11px] text-zinc-500">
        {(sources?.length ?? 0) > 0 && (
          <button
            type="button"
            onClick={onOpenSources}
            className="rounded border border-zinc-200 bg-white px-2 py-0.5 font-medium text-zinc-700 transition hover:border-zinc-300"
          >
            {sources!.length} sources
          </button>
        )}
        {parts.map(([label, value]) => (
          <span key={label}>
            {label} <span className="text-zinc-700">{value}</span>
          </span>
        ))}
        {metrics.model && (
          <span className="font-mono text-zinc-400">{metrics.model}</span>
        )}
      </div>
    </>
  );
}

function AssistantMessage({
  message,
  onOpenSources,
}: {
  message: ChatMessage;
  onOpenSources: () => void;
}) {
  const { content, streaming, metrics, error } = message;
  const waiting = streaming && !content;
  const showCaret = streaming && content.length > 0;

  return (
    <div>
      {waiting ? (
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Spinner className="h-3.5 w-3.5" />
          {message.sources
            ? `Reading ${message.sources.length} chunks…`
            : "Retrieving context…"}
        </div>
      ) : (
        <>
          <div className="markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {content}
            </ReactMarkdown>
            {showCaret && <span className="caret" aria-hidden />}
          </div>
          {metrics?.empty && metrics.message && (
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
              {metrics.message}
            </div>
          )}
          {error && (
            <div className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          {!streaming && <MetricsFooter message={message} onOpenSources={onOpenSources} />}
        </>
      )}
    </div>
  );
}

export default function ChatPanel({
  messages,
  busy,
  hasRepos,
  onSend,
  onStop,
  repos,
  scope,
  onScopeChange,
  onOpenSources,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

 
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  });

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-zinc-50">
      <div
        ref={scrollRef}
        onScroll={() => {
          const el = scrollRef.current;
          if (el) {
            stickRef.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 120;
          }
        }}
        className="flex-1 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-zinc-900">
              <GitBranch className="h-5 w-5 text-white" />
            </div>
            <h1 className="mt-4 text-lg font-semibold tracking-tight">
              Ask your repositories
            </h1>
            <p className="mt-1.5 max-w-md text-sm leading-relaxed text-zinc-500">
              Hybrid BM25 + vector retrieval, cross-encoder reranking, and
              answers grounded in the indexed source files with{" "}
              <span className="font-mono text-[13px]">file:line</span>{" "}
              citations.
            </p>
            {hasRepos ? (
              <div className="mt-6 flex max-w-lg flex-wrap justify-center gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => onSend(q)}
                    className="rounded-full border border-zinc-200 bg-white px-3.5 py-1.5 text-xs text-zinc-600 transition hover:border-zinc-300 hover:bg-zinc-100 hover:text-zinc-900"
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : (
              <p className="mt-6 rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-500">
                Index a GitHub repository from the sidebar to get started.
              </p>
            )}
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-6 px-6 py-6">
            {messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-br-md bg-zinc-900 px-4 py-2.5 text-sm leading-relaxed text-zinc-50">
                    {m.content}
                  </div>
                </div>
              ) : (
                <AssistantMessage
                  key={m.id}
                  message={m}
                  onOpenSources={onOpenSources}
                />
              )
            )}
          </div>
        )}
      </div>

      <Composer
        busy={busy}
        onSend={onSend}
        onStop={onStop}
        repos={repos}
        scope={scope}
        onScopeChange={onScopeChange}
      />
    </main>
  );
}
