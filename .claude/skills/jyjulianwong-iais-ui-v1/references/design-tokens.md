# Design tokens — source of truth

Extracted directly from the Investment-AI-Suite landing page (`tailwind.config.js`, `src/index.css`).
These exact values are what make a restyled site read as "the same family" rather than "dark theme,
generally." Don't approximate them from memory — copy the hex values.

## Color palette

| Token          | Hex       | Role                                                                          |
| -------------- | --------- | ------------------------------------------------------------------------------ |
| `ink-950`      | `#060913` | Page background (body, deepest layer of the aurora gradient)                  |
| `ink-900`      | `#0a0f1f` | Aurora gradient midtone                                                       |
| `ink-800`      | `#10172b` | Aurora gradient highlight, near the top of the page                           |
| `gold` (DEFAULT) | `#e7c47e` | **Primary accent.** Interactive/active states, emphasis text, glows, focus ring |
| `gold-soft`    | `#f3ddae` | Lighter end of gold gradients, hover-text tone                                |
| `gold-deep`    | `#a8803c` | Darker end of gold gradients (bar charts, depth)                              |
| `mint` (DEFAULT) | `#5ce1c6` | **Secondary accent.** Alternate data series, secondary emphasis               |
| `mint-deep`    | `#1f9e88` | Darker end of mint gradients                                                  |
| —              | `#2b6cb0` | Optional tertiary accent, used sparingly (e.g. a third background orb)        |

Body text and structural colors are plain Tailwind slate / white, used at specific opacities rather
than as new tokens:

| Use                                   | Value                                   |
| -------------------------------------- | ---------------------------------------- |
| Primary body text                      | `#e2e8f0` (slate-200)                   |
| Secondary body text / descriptions     | slate-300                               |
| Tertiary text / meta / captions        | slate-400                               |
| Card border                            | `rgba(255,255,255,0.09)` (white/[0.09]) |
| Card background tint (over `.glass`)   | `rgba(255,255,255,0.075)` → `rgba(255,255,255,0.022)` gradient |
| Hairline dividers                      | `rgba(255,255,255,0.07–0.15)`           |
| Pill/button idle background            | `rgba(255,255,255,0.04)`                |
| Pill/button idle border                | `rgba(255,255,255,0.10)`                |

**Role discipline:** gold = primary brand accent (don't overuse — one emphasized word per heading, one
active/hover state per interactive element, focus rings, key glow effects). Mint = secondary accent,
mainly for a second data series or a secondary success-flavored signal. Everything else stays in the
ink/slate/white range. A page that's mostly gold reads as gaudy, not branded.

## Typography

| Role                          | Font                                         | Tailwind config name |
| ------------------------------ | --------------------------------------------- | --------------------- |
| Display / headings             | **Playfair Display** (serif, italic for emphasis spans) | `font-serif`  |
| Body / UI text                 | **Inter**                                     | `font-sans` (default) |
| Micro-labels, numerals, index tags, data figures, timestamps | **JetBrains Mono** | `font-mono` |

Google Fonts weights actually loaded: Inter 300/400/500/600/700; JetBrains Mono 400/500;
Playfair Display 400/500/600/700 roman + 400/500/600 italic. See `assets/google-fonts.snippet.html`.

Playfair Display defaults to old-style figures, which look wrong next to financial numbers — always
pair it with:

```css
.font-serif {
  font-variant-numeric: lining-nums;
}
```

Observed type scale (Tailwind size classes, for reference — not a rigid rule, but keep the *relationship*
between hero/section/body sizes):

- Hero H1: `text-5xl sm:text-7xl`, `font-semibold`, `leading-[1.05]`, `tracking-tight`
- Section H2: `text-3xl sm:text-4xl`, `font-semibold`, `tracking-tight`
- Card title: `text-xl`, `font-semibold`
- Body / description: `text-base` or `text-lg`, `leading-relaxed`
- Micro-label (mono): `text-xs`, `tracking-[0.25em]` (section index) or `tracking-widest` (tags)

## Spacing & radius conventions

- Cards: `rounded-2xl`, padding `p-6`
- Larger panels (pipeline containers, chart containers): `rounded-3xl`, padding `p-6 sm:p-8`
- Pills / badges / buttons: `rounded-full`
- Section vertical rhythm: `py-20` per major section, `mb-12` under a section heading
- Page max-width: `max-w-6xl` for content sections, `max-w-3xl`/`max-w-2xl` for hero/prose blocks

## Motion

- Primary easing: `cubic-bezier(0.16, 1, 0.3, 1)` — a pronounced ease-out, used for hover lifts and
  scroll-reveal. This specific curve is part of the house feel; don't substitute a generic `ease-out`.
- Hover/color transitions: `200ms`
- Spotlight transform/border transition: `300ms`
- Scroll-reveal entrance: `800ms`–`1000ms`, `both` fill mode
- Stagger step for a list of reveal items: `+130ms` per item (sections), `+65ms` per item (tighter
  pipeline steps)
- Always respect `prefers-reduced-motion: reduce` — every animation in this system has a static
  fallback (see `css-recipes.md`).

## Where to find the live originals

If you have access to the Investment-AI-Suite repo itself, the primary sources are
`tailwind.config.js` and `src/index.css` — this file is a snapshot of those, kept in sync manually.
