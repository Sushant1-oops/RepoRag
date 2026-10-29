import { useState } from "react";
import type { FormEvent } from "react";
import { login, register } from "../api";
import type { User } from "../types";
import { Spinner } from "./ui";

// Sign-in wall in front of the app: every visitor picks an account first,
// so repos and chats stay isolated per user (see repo_rag/auth.py).
export default function AuthGate({
  onAuthed,
}: {
  onAuthed: (user: User) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const user =
        mode === "login"
          ? await login(username, password)
          : await register(username, password);
      onAuthed(user);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-zinc-50 px-4">
      <form
        onSubmit={submit}
        className="w-full max-w-sm space-y-4 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm"
      >
        <div>
          <h1 className="text-lg font-semibold text-zinc-900">
            {mode === "login" ? "Sign in to Repo RAG" : "Create your account"}
          </h1>
          <p className="mt-1 text-xs text-zinc-500">
            Each account has its own repositories and chat history.
          </p>
        </div>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-zinc-600">
            Username
          </span>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
            className="w-full rounded-md border border-zinc-200 px-3 py-2 text-sm outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-blue-600/10"
          />
        </label>

        <label className="block">
          <span className="mb-1 block text-xs font-medium text-zinc-600">
            Password
          </span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={8}
            className="w-full rounded-md border border-zinc-200 px-3 py-2 text-sm outline-none transition focus:border-zinc-400 focus:ring-2 focus:ring-blue-600/10"
          />
        </label>

        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-600">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-md bg-zinc-900 py-2 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-50"
        >
          {busy && <Spinner className="h-4 w-4" />}
          {mode === "login" ? "Sign in" : "Create account"}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError(null);
          }}
          className="w-full text-center text-xs text-zinc-500 transition hover:text-zinc-800"
        >
          {mode === "login"
            ? "No account yet? Register"
            : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}
