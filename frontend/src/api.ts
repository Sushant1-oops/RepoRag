

import type {
  ChatMetrics,
  ChatRecord,
  ChatSummary,
  Health,
  IndexStats,
  Repo,
  Source,
} from "./types";


async function consumeSse(
  res: Response,
  onEvent: (event: string, data: unknown) => void
): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      try {
        onEvent(event, JSON.parse(dataLines.join("\n")));
      } catch {
       
      }
    }
  }
}

async function postSse(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorMessage(res));
  await consumeSse(res, onEvent);
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail ?? `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}



export async function fetchHealth(): Promise<Health> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as Health;
}

export async function fetchRepos(): Promise<Repo[]> {
  const res = await fetch("/api/repos");
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as Repo[];
}

export async function deleteRepo(id: string): Promise<void> {
  const res = await fetch(`/api/repos?repo=${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await errorMessage(res));
}

// ---------------------------------------------------------------- chats

export async function fetchChats(): Promise<ChatSummary[]> {
  const res = await fetch("/api/chats");
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as ChatSummary[];
}

export async function fetchChat(id: number): Promise<ChatRecord> {
  const res = await fetch(`/api/chats/${id}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as ChatRecord;
}

export async function deleteChat(id: number): Promise<void> {
  const res = await fetch(`/api/chats/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorMessage(res));
}



export interface ChatHandlers {
  onChat: (info: { chat_id: number }) => void;
  onSources: (sources: Source[]) => void;
  onToken: (token: string) => void;
  onDone: (metrics: ChatMetrics) => void;
  onError: (message: string) => void;
}

export interface ChatRequest {
  question: string;
  chat_id: number | null; // null -> the server creates a new chat
  repo: string | null;
  top_k: number;
  use_hybrid: boolean;
  use_reranker: boolean;
}


export async function streamChat(
  req: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal
): Promise<void> {
  await postSse(
    "/api/chat",
    req,
    (event, data) => {
      const d = data as Record<string, unknown>;
      if (event === "chat")
        handlers.onChat({ chat_id: d.chat_id as number });
      else if (event === "sources") handlers.onSources(d as unknown as Source[]);
      else if (event === "token") handlers.onToken(d as unknown as string);
      else if (event === "done") handlers.onDone(d as unknown as ChatMetrics);
      else if (event === "error")
        handlers.onError((d.message as string) ?? "Unknown error");
    },
    signal
  );
}

export interface IndexHandlers {
  onProgress: (message: string) => void;
  onDone: (stats: IndexStats) => void;
  onError: (message: string) => void;
}


export async function indexRepo(
  url: string,
  handlers: IndexHandlers
): Promise<void> {
  await postSse("/api/repos", { url }, (event, data) => {
    const d = data as Record<string, unknown>;
    if (event === "progress")
      handlers.onProgress((d.message as string) ?? "");
    else if (event === "done") handlers.onDone(d as unknown as IndexStats);
    else if (event === "error")
      handlers.onError((d.message as string) ?? "Unknown error");
  });
}
