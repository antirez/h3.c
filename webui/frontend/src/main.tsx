import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
// Fonts are bundled, not fetched: a machine with a GPU may well be offline.
import "@fontsource/instrument-sans/400.css";
import "@fontsource/instrument-serif/400.css";
import "@fontsource/instrument-serif/400-italic.css";
import "@fontsource/instrument-sans/500.css";
import "@fontsource/instrument-sans/600.css";
import "@fontsource/instrument-sans/700.css";
import "@fontsource/martian-mono/300.css";
import "@fontsource/martian-mono/500.css";
import "./styles.css";

const container = document.getElementById("root");
if (!container) throw new Error("missing #root");
createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
