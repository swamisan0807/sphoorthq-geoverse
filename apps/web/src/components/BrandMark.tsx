/** Stylized stand-in for the SphoorthiQ logo - a quantum-orbit atom mark
 * (three crossing orbit rings + nucleus, blue-to-green gradient), echoing
 * the "QUANTUM" pillar icon in the real logo and unambiguous even at small
 * sizes (the previous crossing-strand version read as an "X" in the
 * sidebar). Used everywhere until the real asset is dropped at
 * apps/web/public/logo.png, at which point LoginPage swaps to that
 * automatically (see its onError fallback). */
export default function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="sq-grad" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#4f8cf0" />
          <stop offset="1" stopColor="#34d399" />
        </linearGradient>
      </defs>
      <ellipse cx="24" cy="24" rx="19" ry="8" stroke="url(#sq-grad)" strokeWidth="2.2" />
      <ellipse cx="24" cy="24" rx="19" ry="8" stroke="url(#sq-grad)" strokeWidth="2.2" transform="rotate(60 24 24)" />
      <ellipse cx="24" cy="24" rx="19" ry="8" stroke="url(#sq-grad)" strokeWidth="2.2" transform="rotate(120 24 24)" />
      <circle cx="24" cy="24" r="4.5" fill="url(#sq-grad)" />
    </svg>
  );
}
