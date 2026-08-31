import { useState } from "react";

import { ApiError, api } from "../api";
import { LogoMark } from "./Logo";
import type { User } from "../types";

/** The door: one column, two states, the same look as the rest of the page.
 *
 *  The first person here makes their own account and becomes the
 *  administrator; everyone after that needs an invite, and the server is the
 *  one that says so — the field is simply left empty for the first account.
 */
export function AuthScreen({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [invite, setInvite] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      if (mode === "register") {
        await api.register(username.trim(), password, invite.trim());
        // Registering makes the account but not the session: walk in.
        const user = await api.login(username.trim(), password);
        onSignedIn(user);
      } else {
        const user = await api.login(username.trim(), password);
        onSignedIn(user);
      }
    } catch (failure) {
      setProblem(
        failure instanceof ApiError ? failure.errors.join(" ") : "The request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <div className="auth">
        <div className="auth-mark" aria-hidden="true">
          <LogoMark size={44} />
        </div>
        <h1>{mode === "login" ? "Welcome back" : "Make your account"}</h1>

        <label className="field">
          <span>Username</span>
          <input
            value={username}
            autoComplete="username"
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {mode === "register" ? (
          <label className="field">
            <span>
              Invite <span className="hint">from the administrator</span>
            </span>
            <input
              value={invite}
              onChange={(event) => setInvite(event.target.value)}
            />
          </label>
        ) : null}

        {problem ? (
          <p className="wrong" role="status">
            <b>{problem}</b>
          </p>
        ) : null}

        <div className="go">
          <button
            className="make"
            disabled={busy || username.trim() === "" || password === ""}
            onClick={() => void submit()}
          >
            {busy ? "One moment…" : mode === "login" ? "Sign in" : "Register"}
          </button>
        </div>

        <p className="switch">
          {mode === "login" ? (
            <>
              First time here?{" "}
              <button onClick={() => setMode("register")}>Make an account</button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button onClick={() => setMode("login")}>Sign in</button>
            </>
          )}
        </p>
        <p className="note">
          The administrator account is defined on the server; every other
          account is made with a single-use invite.
        </p>
      </div>
    </main>
  );
}
