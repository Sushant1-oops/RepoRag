import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { ArrowUp, Stop } from "../icons";
import type { Repo } from "../types";

interface ComposerProps {
  busy: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
  repos: Repo[];
  scope: string | null;
  onScopeChange: (id: string | null) => void;
}

export default function Composer({
  busy,
  onSend,
  onStop,
  repos,
  scope,
  onScopeChange,
}: ComposerProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);


  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 192)}px`;
  }, [text]);

  const send = () => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setText("");
    onSend(trimmed);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="border-t border-zinc-200 bg-white px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-zinc-300 bg-white p-2 shadow-sm transition focus-within:border-zinc-400 focus-within:ring-2 focus-within:ring-blue-600/10">
          <select
            value={scope ?? ""}
            onChange={(e) => onScopeChange(e.target.value || null)}
            title="Repository scope"
            className="h-8 max-w-[180px] shrink-0 truncate rounded-md border border-zinc-200 bg-zinc-50 px-2 text-xs text-zinc-600 outline-none transition hover:border-zinc-300 focus:ring-2 focus:ring-blue-600/10"
          >
            <option value="">All repositories</option>
            {repos.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id.replace("__", "/")}
              </option>
            ))}
          </select>
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              repos.length === 0
                ? "Index a repository first…"
                : "Ask about the indexed code…"
            }
            className="max-h-48 min-w-0 flex-1 resize-none bg-transparent px-1 py-1.5 text-sm outline-none placeholder:text-zinc-400"
          />
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              title="Stop generating"
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-zinc-900 text-white transition hover:bg-zinc-700"
            >
              <Stop className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={send}
              disabled={!text.trim()}
              title="Send"
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-zinc-900 text-white transition hover:bg-zinc-700 disabled:opacity-30"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-[11px] text-zinc-400">
          Answers are grounded in indexed code and cite{" "}
          <span className="font-mono">file:line</span>. Verify before relying on
          them.
        </p>
      </div>
    </div>
  );
}
