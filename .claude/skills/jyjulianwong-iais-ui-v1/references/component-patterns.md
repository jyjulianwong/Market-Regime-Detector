# Component patterns

How the tokens and CSS recipes compose into the landing page's actual UI. These are patterns to
translate into the target repo's existing component structure — don't force a target site to adopt a
pattern it has no real use for (e.g. don't invent a multi-step pipeline diagram for a single-page
tool that has no pipeline).

## Page shell

Fixed `.aurora` background (see `css-recipes.md`) sitting behind everything via `z-index: -1`, 2-3
`.orb` blobs inside it (gold, mint-deep, and optionally a third brand-adjacent hue), each a different
size/position/animation-delay so they drift independently. Main content scrolls over the top; nothing
else needs `z-index` management.

## Hero

- Serif H1, mixing `.text-ivory-gradient` (most of the words) with one `.text-gold-gradient` span for
  the emphasized word.
- A subtitle paragraph in slate, centered, `max-width` constrained so it doesn't stretch full-bleed.
- A pill CTA/link: `.glass`, `rounded-full`, small pulsing status dot (a `bg-gold` dot plus a second
  `animate-ping` dot layered under it) + icon + label. Hover: border and text shift toward gold.
- Optional decorative SVG behind the hero text, bottom-anchored, masked to fade out near the top, low
  opacity, animated draw-in via `stroke-dashoffset`. On the landing page this is a stylized price
  line because the product is market-flavored — **only reuse the literal "price chart" motif if the
  target product is itself about market/price data.** Otherwise use an equivalent abstract line/area
  shape (e.g. a generic flowing curve, a subtle node/graph silhouette) in the same gold→mint gradient
  treatment, or omit the decorative SVG entirely if it doesn't fit.
- Everything in the hero enters with the `fly-in` reveal, staggered ~120ms apart (headline, subtitle,
  CTA).

## Section heading

The consistent opener for every major section:

1. A mono micro-label row: a two-digit index (`01`, `02`, ...) in `gold`, followed by a short
   horizontal gradient rule (`gold` fading to transparent).
2. A serif H2 directly below, white, bold-ish weight, tight tracking.
3. An optional slate intro paragraph below that, width-constrained.

The whole block reveals as one unit on scroll (not staggered internally).

## Card (e.g. a repo/project tile, a resource link)

- Container: `.glass .spotlight`, `rounded-2xl`, padding, `flex flex-col`, full height within its
  grid cell so a row of cards lines up even with uneven content length.
- Title: serif, semibold, white, transitions to gold on group-hover.
- Description: slate, a few lines, clamp long text rather than growing the card unpredictably.
- Footer meta row, separated from the body by a hairline (`border-top: 1px solid rgba(255,255,255,0.07)`):
  small slate meta info on the left (e.g. a colored dot + label, a count with an icon), a pill-style
  link button on the right (`rounded-full`, bordered, hover → gold border/background tint/text).
- Loading state: a skeleton card of the same shape, each text line replaced by a `.shimmer` block.

## Pipeline / step diagram

For any multi-stage process the target repo wants to visualize (a data pipeline, a request flow, an
architecture diagram):

- Each step is a numbered marker (small circle, gold border + gold-tinted fill + glow, mono digit)
  next to a node — either a `.glass .spotlight` card-like node (for an internal component the user can
  link to) or a plainer dashed-border node (for an external dependency like a cloud resource, with no
  link).
- Between consecutive steps: a vertical `.flow-line` with a `.flow-arrow` at its foot, and a small
  pill label describing the relationship (e.g. "uploads reports to", "imported by") placed beside the
  line.
- If there's a column of *independent* items with no pipeline relationship between them (e.g. a list
  of standalone tools), skip the connector entirely and just stack the nodes with reveal stagger — no
  numbered markers, no flow-lines, since there's no sequence to communicate.
- Reveal stagger for a tight step sequence: ~65ms between steps. For a looser list of independent
  items: ~130ms between items.

## Numeric / bar comparison

For any "compare these values" visualization:

- Horizontal bars, thin (`~0.875rem` tall), rounded on the open end, filled with a gradient
  (gold-deep→gold for a "primary"/estimate series, mint-deep→mint for a "secondary"/actual series),
  with a soft colored glow shadow matching the fill.
- A value missing from a series renders as a short dashed-border stub instead of a solid bar, so
  "not yet recorded" is visually distinct from "recorded as zero."
- A light vertical grid (very low-opacity white lines) behind the bars, with axis tick labels in mono
  type beneath.
- On scroll into view, bars grow from `scaleX(0)` to `scaleX(1)` with the house easing curve,
  staggered per row; the numeric value label beside each bar fades in slightly after the bar finishes
  growing, not simultaneously.
- A small legend row above the chart, one swatch + label per series.

## Footer

A thin horizontal hairline (`linear-gradient` from transparent → `rgba(255,255,255,0.15)` → transparent,
1px tall), then a centered, small, serif signature/credit line in slate below it. Keep it understated —
this is the one place on the page that's deliberately quiet.
