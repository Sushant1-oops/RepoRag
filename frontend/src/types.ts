/** Shared API + UI types. Mirrors the payloads sent by server.py. */

export interface Health {
  ok: boolean;
  chunks: number;
  model: string;
}

export interface Repo {
  id: string; 
  url: string;
  files: number;
  chunks: number;
}

export interface IndexStats {
  repo: string;
  files: number;
  chunks: number;
}


export interface Source {
  id: string;
  text: string;
  repo: string;
  file: string;
  start_line: number;
  end_line: number;
  semantic_score: number | null;
  keyword_score: number | null;
  semantic_rank: number | null;
  keyword_rank: number | null;
  fused_score: number;
  rerank_score: number | null;
}


export interface ChatMetrics {
  citations?: string[];
  retrieval_ms?: number;
  first_token_ms?: number;
  total_ms?: number;
  model?: string;
  empty?: boolean;
  message?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  sources?: Source[];
  metrics?: ChatMetrics;
  error?: string;
}

export interface Settings {
  topK: number;
  useHybrid: boolean;
  useReranker: boolean;
}

export interface ChatSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

/** Message row as stored in SQLite (sources/metrics JSON already decoded). */
export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  metrics: ChatMetrics | null;
  error: string | null;
  created_at: string;
}

export interface ChatRecord extends ChatSummary {
  messages: StoredMessage[];
}
