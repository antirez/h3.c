import { useState } from "react";

/** Deleting a video is final — no bin, no undo — so it asks once first.
 *
 *  The question replaces the button in place rather than opening a dialog:
 *  the answer stays where the eye already is, on the take being deleted.
 */
export function DeleteControl({ label, onDelete }: {
  label: string;
  onDelete: () => void | Promise<void>;
}) {
  const [asking, setAsking] = useState(false);

  if (!asking) {
    return (
      <button className="del" onClick={() => setAsking(true)}>
        {label}
      </button>
    );
  }
  return (
    <span className="asking">
      <span>Delete for good?</span>
      <button className="del yes" onClick={() => void onDelete()}>
        Yes
      </button>
      <button className="del" onClick={() => setAsking(false)}>
        Keep
      </button>
    </span>
  );
}
