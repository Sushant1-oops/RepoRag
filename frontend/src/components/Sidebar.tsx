import { useState } from "react";
import type { FormEvent } from "react";
import { Database, GitBranch, Plus, Trash } from "../icons";
import { SectionLabel, Spinner, Switch } from "./ui";
import type { ChatSummary, Health, Repo, Settings } from "../types";

function relTime(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(s) || s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

interface SidebarProps {
  repos: Repo[];
  health: Health | null;
  scope: string | null;
  onScopeChange: (id: string | null) => void;
  onAddRepo: (url: string) => Promise<void>;
  onRemoveRepo: (id: string) => Promise<void>;
  chats: ChatSummary[];
  activeChatId: number | null;
  onLoadChat: (id: number) => Promise<void>;
  onNewChat: () => void;
  onDeleteChat: (id: number) => Promise<void>;
  indexing: boolean;
  progressLines: string[];
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
}

export default function Sidebar({
  repos,
  health,
  scope,
  onScopeChange,
  onAddRepo,
  onRemoveRepo,
  chats,
  activeChatId,
  onLoadChat,
  onNewChat,
  onDeleteChat,
  indexing,
  progressLines,
  settings,
  onSettingsChange,
}: SidebarProps) {
  const [url, setUrl] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || indexing) return;
    setUrl("");
    void onAddRepo(trimmed);
  };

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-200 bg-white">
      {/* Brand */}
      <div className="flex items-center gap-2.5 border-b border-zinc-200 px-4 py-3.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900">
          <GitBranch className="h-4 w-4 text-white" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold tracking-tight">Repo RAG</div>
          <div className="text-[11px] text-zinc-500">
            Grounded code Q&amp;A
          </div>
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto px-4 py-4">
       
        <section className="space-y-2">
          <SectionLabel>Add repository</SectionLabel>
          <form onSubmit={submit} className="flex gap-2">
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://github.com/owner/repo"
              disabled={indexing}
              spellCheck={false}
              className="h-8 min-w-0 flex-1 rounded-md border border-zinc-300 bg-white px-2.5 text-xs outline-none transition placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-2 focus:ring-blue-600/10 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={indexing || !url.trim()}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-zinc-900 text-white transition hover:bg-zinc-700 disabled:opacity-40"
              title="Clone and index"
            >
              {indexing ? (
                <Spinner className="h-3.5 w-3.5" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
            </button>
          </form>
          {(indexing || progressLines.length > 0) && (
            <div className="max-h-36 space-y-0.5 overflow-y-auto rounded-md border border-zinc-200 bg-zinc-50 p-2 font-mono text-[10.5px] leading-relaxed text-zinc-500">
              {progressLines.map((line, i) => (
                <div
                  key={i}
                  className={line.startsWith("error:") ? "text-red-600" : ""}
                >
                  {line}
                </div>
              ))}
            </div>
          )}
        </section>

        
        <section className="space-y-1.5">
          <SectionLabel>Repositories</SectionLabel>
          <button
            type="button"
            onClick={() => onScopeChange(null)}
            className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition ${
              scope === null ? "bg-zinc-100" : "hover:bg-zinc-50"
            }`}
          >
            <Database
              className={`h-3.5 w-3.5 shrink-0 ${
                scope === null ? "text-blue-600" : "text-zinc-400"
              }`}
            />
            <span className="flex-1 truncate text-xs font-medium text-zinc-700">
              All repositories
            </span>
            <span className="font-mono text-[10.5px] text-zinc-400">
              {repos.length}
            </span>
          </button>
          {repos.map((repo) => {
            const selected = scope === repo.id;
            return (
              <div
                key={repo.id}
                className={`group flex items-center gap-1 rounded-md pl-2 pr-1 transition ${
                  selected ? "bg-zinc-100" : "hover:bg-zinc-50"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onScopeChange(selected ? null : repo.id)}
                  className="flex min-w-0 flex-1 items-center gap-2 py-1.5 text-left"
                >
                  <GitBranch
                    className={`h-3.5 w-3.5 shrink-0 ${
                      selected ? "text-blue-600" : "text-zinc-400"
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span
                      className={`block truncate text-xs font-medium ${
                        selected ? "text-zinc-900" : "text-zinc-700"
                      }`}
                    >
                      {repo.id.replace("__", "/")}
                    </span>
                    <span className="block font-mono text-[10.5px] text-zinc-400">
                      {repo.files} files · {repo.chunks} chunks
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => void onRemoveRepo(repo.id)}
                  title="Remove from index"
                  className="rounded p-1.5 text-zinc-300 opacity-0 transition group-hover:opacity-100 hover:bg-zinc-200 hover:text-red-600 focus:opacity-100"
                >
                  <Trash className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })}
          {repos.length === 0 && (
            <p className="px-2 py-1 text-[11px] text-zinc-400">
              Nothing indexed yet. Add a GitHub URL above.
            </p>
          )}
        </section>

        {/* Chats */}
        <section className="space-y-1.5">
          <div className="flex items-center justify-between">
            <SectionLabel>Chats</SectionLabel>
            <button
              type="button"
              onClick={onNewChat}
              title="Start a new chat"
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900"
            >
              <Plus className="h-3 w-3" /> New
            </button>
          </div>
          {chats.length === 0 && (
            <p className="px-2 py-1 text-[11px] text-zinc-400">
              Conversations are saved here automatically.
            </p>
          )}
          {chats.map((chat) => {
            const active = chat.id === activeChatId;
            return (
              <div
                key={chat.id}
                className={`group flex items-center gap-1 rounded-md pl-2 pr-1 transition ${
                  active ? "bg-zinc-100" : "hover:bg-zinc-50"
                }`}
              >
                <button
                  type="button"
                  onClick={() => void onLoadChat(chat.id)}
                  className="flex min-w-0 flex-1 flex-col items-start py-1.5 text-left"
                >
                  <span
                    className={`max-w-full truncate text-xs font-medium ${
                      active ? "text-zinc-900" : "text-zinc-700"
                    }`}
                  >
                    {chat.title}
                  </span>
                  <span className="text-[10.5px] text-zinc-400">
                    {relTime(chat.updated_at)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => void onDeleteChat(chat.id)}
                  title="Delete chat"
                  className="rounded p-1.5 text-zinc-300 opacity-0 transition group-hover:opacity-100 hover:bg-zinc-200 hover:text-red-600 focus:opacity-100"
                >
                  <Trash className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })}
        </section>

        
        <section className="space-y-3">
          <SectionLabel>Retrieval</SectionLabel>
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-600">Top-k chunks</span>
              <span className="font-mono text-xs text-zinc-900">
                {settings.topK}
              </span>
            </div>
            <input
              type="range"
              min={2}
              max={16}
              step={1}
              value={settings.topK}
              onChange={(e) =>
                onSettingsChange({ ...settings, topK: Number(e.target.value) })
              }
              className="w-full accent-blue-600"
            />
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="min-w-0">
              <span className="block text-xs text-zinc-600">
                Hybrid retrieval
              </span>
              <span className="block text-[10.5px] text-zinc-400">
                BM25 + vector fusion
              </span>
            </span>
            <Switch
              checked={settings.useHybrid}
              onChange={(v) => onSettingsChange({ ...settings, useHybrid: v })}
            />
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="min-w-0">
              <span className="block text-xs text-zinc-600">Reranker</span>
              <span className="block text-[10.5px] text-zinc-400">
                Cross-encoder re-scoring
              </span>
            </span>
            <Switch
              checked={settings.useReranker}
              onChange={(v) =>
                onSettingsChange({ ...settings, useReranker: v })
              }
            />
          </div>
        </section>
      </div>


      <div className="border-t border-zinc-200 px-4 py-3">
        <div className="flex items-center gap-2 text-[11px] text-zinc-500">
          <span
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              health ? "bg-emerald-500" : "bg-red-500"
            }`}
          />
          <span className="min-w-0 flex-1 truncate font-mono">
            {health?.model ?? "backend offline"}
          </span>
          {health && (
            <span className="shrink-0 text-zinc-400">
              {health.chunks} chunks
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
