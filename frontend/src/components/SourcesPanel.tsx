import { useState } from "react";
import { ChevronDown, Close } from "../icons";
import { SectionLabel } from "./ui";
import type { Source } from "../types";

function ScoreChips({ source }: { source: Source }) {
  const fmt = (v: number) =>
    Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
  return (
    <span className="flex flex-wrap gap-1">
      {source.rerank_score != null && (
        <span className="rounded-sm bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] text-white">
          rerank {fmt(source.rerank_score)}
        </span>
      )}
      {source.semantic_score != null && (
        <span className="rounded-sm border border-zinc-200 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500">
          vec {fmt(source.semantic_score)}
        </span>
      )}
      {source.keyword_score != null && (
        <span className="rounded-sm border border-zinc-200 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500">
          bm25 {fmt(source.keyword_score)}
        </span>
      )}
    </span>
  );
}

function chunkBody(text: string): string {
 
  const firstNl = text.indexOf("\n");
  if (firstNl !== -1 && text.slice(0, firstNl).startsWith("File:")) {
    return text.slice(firstNl + 1).trim();
  }
  return text;
}

function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-lg border border-zinc-200 bg-white">
      <div className="flex items-start gap-2 p-3">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="min-w-0 flex-1 text-left"
          title={open ? "Hide chunk" : "Show chunk"}
        >
          <div className="flex items-center gap-1.5">
            <span className="truncate font-mono text-xs font-medium text-zinc-800">
              {source.file}
            </span>
            <ChevronDown
              className={`h-3 w-3 shrink-0 text-zinc-400 transition-transform ${
                open ? "rotate-180" : ""
              }`}
            />
          </div>
          <div className="mt-0.5 font-mono text-[10.5px] text-zinc-400">
            L{source.start_line}–{source.end_line}
          </div>
          <div className="mt-1.5">
            <ScoreChips source={source} />
          </div>
        </button>
      </div>
      {open && (
        <pre className="mx-3 mb-3 max-h-72 overflow-auto rounded-md bg-zinc-50 p-2.5 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-words text-zinc-700">
          {chunkBody(source.text)}
        </pre>
      )}
    </li>
  );
}

export default function SourcesPanel({
  sources,
  onClose,
}: {
  sources: Source[];
  onClose: () => void;
}) {
  return (
    <aside className="hidden w-[360px] shrink-0 flex-col border-l border-zinc-200 bg-white lg:flex">
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <div className="flex items-center gap-2">
          <SectionLabel>Sources</SectionLabel>
          {sources.length > 0 && (
            <span className="rounded-sm bg-zinc-100 px-1.5 py-0.5 font-mono text-[10.5px] text-zinc-500">
              {sources.length}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          title="Hide sources panel"
          className="rounded p-1 text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-600"
        >
          <Close className="h-4 w-4" />
        </button>
      </div>
      {sources.length === 0 ? (
        <p className="px-4 py-4 text-xs leading-relaxed text-zinc-400">
          Retrieved chunks appear here after the first answer, with rerank and
          retrieval scores per source.
        </p>
      ) : (
        <ul className="flex-1 space-y-2.5 overflow-y-auto p-3">
          {sources.map((s) => (
            <SourceCard key={s.id} source={s} />
          ))}
        </ul>
      )}
    </aside>
  );
}
