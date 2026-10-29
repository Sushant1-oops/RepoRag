import { useCallback, useEffect, useRef, useState } from "react";
import AuthGate from "./components/AuthGate";
import ChatPanel from "./components/ChatPanel";
import Sidebar from "./components/Sidebar";
import SourcesPanel from "./components/SourcesPanel";
import {
  deleteChat,
  deleteRepo,
  fetchChat,
  fetchChats,
  fetchHealth,
  fetchMe,
  fetchRepos,
  indexRepo,
  logout,
  streamChat,
} from "./api";
import type {
  ChatMessage,
  ChatSummary,
  Health,
  Repo,
  Settings,
  Source,
  User,
} from "./types";

let seq = 0;
const genId = () => `m${Date.now().toString(36)}-${seq++}`;

// display name for a repo id; pre-auth fallback decodes the safe id
const repoName = (id: string) => id.replace(/__/g, "/");

export default function App() {
  // session: resolved once on mount; nothing renders until it is known
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);

  const [repos, setRepos] = useState<Repo[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [scope, setScope] = useState<string | null>(null); // repo id, null = all
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [progressLines, setProgressLines] = useState<string[]>([]);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [settings, setSettings] = useState<Settings>({
    topK: 8,
    useHybrid: true,
    useReranker: true,
  });
  const abortRef = useRef<AbortController | null>(null);

  // Saved conversations; the active id is mirrored in a ref so streaming
  // callbacks never read a stale closure value.
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const activeChatRef = useRef<number | null>(null);
  const setActiveChat = useCallback((id: number | null) => {
    activeChatRef.current = id;
    setActiveChatId(id);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [repoList, h] = await Promise.all([fetchRepos(), fetchHealth()]);
      setRepos(repoList);
      setHealth(h);
      
      setScope((s) => (s && repoList.some((r) => r.id === s) ? s : null));
    } catch {
      setHealth(null);
    }
  }, []);

  const refreshChats = useCallback(async () => {
    try {
      setChats(await fetchChats());
    } catch {
      // backend unreachable: keep the current list
    }
  }, []);

  useEffect(() => {
    fetchMe()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setAuthReady(true));
  }, []);

  useEffect(() => {
    void refresh();
    void refreshChats();
  }, [refresh, refreshChats, user]);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || abortRef.current) return;

      const botId = genId();
      setMessages((all) => [
        ...all,
        { id: genId(), role: "user", content: trimmed },
        { id: botId, role: "assistant", content: "", streaming: true },
      ]);
      setBusy(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const patch = (fn: (msg: ChatMessage) => ChatMessage) =>
        setMessages((all) => all.map((m) => (m.id === botId ? fn(m) : m)));

      try {
        await streamChat(
          {
            question: trimmed,
            chat_id: activeChatRef.current,
            repo: scope,
            top_k: settings.topK,
            use_hybrid: settings.useHybrid,
            use_reranker: settings.useReranker,
          },
          {
            onChat: ({ chat_id }) => {
              setActiveChat(chat_id);
              void refreshChats();
            },
            onSources: (sources: Source[]) => patch((msg) => ({ ...msg, sources })),
            onToken: (token: string) =>
              patch((msg) => ({ ...msg, content: msg.content + token })),
            onDone: (metrics) => {
              patch((msg) => ({ ...msg, streaming: false, metrics }));
              void refreshChats(); // move the chat to the top of the list
            },
            onError: (error) => patch((msg) => ({ ...msg, streaming: false, error })),
          },
          controller.signal
        );
      } catch (e) {
        const aborted = e instanceof DOMException && e.name === "AbortError";
        patch((msg) => ({
          ...msg,
          streaming: false,
          error: aborted
            ? undefined
            : e instanceof Error
              ? e.message
              : String(e),
        }));
      } finally {
        abortRef.current = null;
        setBusy(false);
      }
    },
    [scope, settings]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const addRepo = useCallback(
    async (url: string) => {
      setIndexing(true);
      setProgressLines([]);
      try {
        await indexRepo(url, {
          onProgress: (msg) => setProgressLines((lines) => [...lines, msg]),
          onDone: (stats) => {
            setProgressLines((lines) => [
              ...lines,
              `done — ${stats.files} files, ${stats.chunks} chunks`,
            ]);
            void refresh();
          },
          onError: (msg) =>
            setProgressLines((lines) => [...lines, `error: ${msg}`]),
        });
      } catch (e) {
        setProgressLines((lines) => [
          ...lines,
          `error: ${e instanceof Error ? e.message : String(e)}`,
        ]);
      } finally {
        setIndexing(false);
      }
    },
    [refresh]
  );

  const removeRepo = useCallback(
    async (id: string) => {
      if (!window.confirm(`Remove ${repoName(id)} from the index?`)) {
        return;
      }
      try {
        await deleteRepo(id);
      } finally {
        void refresh();
      }
    },
    [refresh]
  );

  const loadChat = useCallback(
    async (id: number) => {
      try {
        const record = await fetchChat(id);
        setActiveChat(id);
        setMessages(
          record.messages.map((m) => ({
            id: `db${m.id}`,
            role: m.role,
            content: m.content,
            sources: m.sources ?? [],
            metrics: m.metrics ?? undefined,
            error: m.error ?? undefined,
          })),
        );
      } catch {
        void refreshChats(); // deleted elsewhere: resync the list
      }
    },
    [refreshChats, setActiveChat]
  );

  const newChat = useCallback(() => {
    setActiveChat(null);
    setMessages([]);
  }, [setActiveChat]);

  const removeChat = useCallback(
    async (id: number) => {
      try {
        await deleteChat(id);
      } finally {
        if (activeChatRef.current === id) newChat();
        void refreshChats();
      }
    },
    [newChat, refreshChats]
  );

  const signOut = useCallback(async () => {
    try {
      await logout();
    } finally {
      // drop all state from the previous account before the gate re-renders
      setActiveChat(null);
      setMessages([]);
      setRepos([]);
      setChats([]);
      setScope(null);
      setUser(null);
    }
  }, [setActiveChat]);

  const activeSources =
    [...messages]
      .reverse()
      .find((m) => m.role === "assistant" && (m.sources?.length ?? 0) > 0)
      ?.sources ?? [];

  if (!authReady) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-zinc-400">
        Loading…
      </div>
    );
  }
  if (!user) {
    return <AuthGate onAuthed={setUser} />;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        user={user}
        onLogout={signOut}
        repos={repos}
        health={health}
        scope={scope}
        onScopeChange={setScope}
        onAddRepo={addRepo}
        onRemoveRepo={removeRepo}
        indexing={indexing}
        progressLines={progressLines}
        settings={settings}
        onSettingsChange={setSettings}
        chats={chats}
        activeChatId={activeChatId}
        onLoadChat={loadChat}
        onNewChat={newChat}
        onDeleteChat={removeChat}
      />
      <ChatPanel
        messages={messages}
        busy={busy}
        hasRepos={repos.length > 0}
        onSend={ask}
        onStop={stop}
        repos={repos}
        scope={scope}
        onScopeChange={setScope}
        onOpenSources={() => setSourcesOpen(true)}
      />
      {sourcesOpen && (
        <SourcesPanel
          sources={activeSources}
          onClose={() => setSourcesOpen(false)}
        />
      )}
    </div>
  );
}
