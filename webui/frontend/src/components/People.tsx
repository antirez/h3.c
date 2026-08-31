import { useCallback, useEffect, useState } from "react";

import { ApiError, api } from "../api";
import type { Invite, User } from "../types";
import { DeleteControl } from "./DeleteControl";

/** Accounts and invites, for the administrator only (R30). */
export function People({ me }: { me: User }) {
  const [users, setUsers] = useState<User[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [newCode, setNewCode] = useState<string | null>(null);
  const [passwords, setPasswords] = useState<Record<number, string>>({});
  const [problem, setProblem] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [list, codes] = await Promise.all([api.users(), api.invites()]);
    setUsers(list);
    setInvites(codes);
  }, []);

  useEffect(() => {
    /* The state updates land in promise callbacks, not in the effect body,
     * so there is no cascading render to avoid here. */
    /* eslint-disable react-hooks/set-state-in-effect */
    void refresh();
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [refresh]);

  function fail(failure: unknown, fallback: string) {
    setProblem(failure instanceof ApiError ? failure.errors.join(" ") : fallback);
  }

  async function makeInvite() {
    setProblem(null);
    try {
      const { code } = await api.createInvite();
      setNewCode(code);
      await refresh();
    } catch (failure) {
      fail(failure, "The invite could not be made.");
    }
  }

  async function remove(user: User) {
    setProblem(null);
    try {
      await api.deleteUser(user.id ?? 0);
      await refresh();
    } catch (failure) {
      fail(failure, "That account could not be deleted.");
    }
  }

  async function reset(user: User) {
    setProblem(null);
    const secret = passwords[user.id ?? 0] ?? "";
    try {
      await api.resetPassword(user.id ?? 0, secret);
      setPasswords((current) => ({ ...current, [user.id ?? 0]: "" }));
      setProblem(`${user.username} was signed out and must use the new password.`);
    } catch (failure) {
      fail(failure, "The password could not be changed.");
    }
  }

  return (
    <div className="people">
      <div className="row invites">
        <button className="lib" onClick={() => void makeInvite()}>
          New invite
        </button>
        {newCode ? (
          <span className="code" title="Copy this and send it to the person">
            {newCode}
          </span>
        ) : null}
        <span className="invlist">
          {invites.filter((invite) => !invite.used).length} unused ·{" "}
          {invites.filter((invite) => invite.used).length} used
        </span>
      </div>

      {users.map((user) => (
        <div className="row person" key={user.username}>
          <span className="who">
            {user.username}
            {user.role === "admin" ? <em> · administrator</em> : null}
          </span>
          <input
            type="password"
            placeholder="new password"
            value={passwords[user.id ?? 0] ?? ""}
            onChange={(event) =>
              setPasswords((current) => ({
                ...current,
                [user.id ?? 0]: event.target.value,
              }))
            }
          />
          <button
            className="lib"
            disabled={(passwords[user.id ?? 0] ?? "") === ""}
            onClick={() => void reset(user)}
          >
            reset password
          </button>
          {user.id !== me.id && user.id !== undefined ? (
            <DeleteControl label="delete" onDelete={() => remove(user)} />
          ) : null}
        </div>
      ))}

      {problem ? (
        <p className="note" role="status">
          {problem}
        </p>
      ) : null}
      <p className="note">
        Resetting a password signs that person out. An account can only be
        deleted once it owns no videos and no uploads.
      </p>
    </div>
  );
}
