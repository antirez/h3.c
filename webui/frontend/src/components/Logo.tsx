/** The h3c studio mark (R31, concept A — "the frame").
 *
 *  The developing frame drawn as a sign: an open frame with the
 *  perforation on its left edge, and the corner at the bottom right still
 *  developing in the accent. One SVG, used from the favicon to the header;
 *  `currentColor` makes it follow the theme with no new colour.
 */
export function LogoMark({ size = 22 }: { size?: number }) {
  const stroke = size <= 20 ? 3 : 2.6;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      aria-hidden="true"
      className="mark"
    >
      <rect
        x="7"
        y="5"
        width="20"
        height="22"
        rx="3.5"
        fill="none"
        stroke="currentColor"
        strokeWidth={stroke}
      />
      <rect x="2.5" y="9" width="3.5" height="4.5" rx="1.4" fill="currentColor" />
      <rect x="2.5" y="18.5" width="3.5" height="4.5" rx="1.4" fill="currentColor" />
      <path
        d="M13 27v-3.5a2 2 0 0 1 2-2h12"
        fill="none"
        stroke="var(--accent)"
        strokeWidth={stroke}
        strokeLinecap="round"
      />
    </svg>
  );
}
