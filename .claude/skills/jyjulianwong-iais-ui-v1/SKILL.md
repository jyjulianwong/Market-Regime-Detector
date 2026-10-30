---
name: jyjulianwong-iais-ui-v1
description: Restyle a website or app's UI to match Julian Wong's "Investment AI Suite" family look — the dark "fintech noir" aesthetic used on the flagship Investment-AI-Suite landing page (deep ink/navy background, gold + mint accents, glassmorphism cards, Playfair Display serif headings over Inter body text and JetBrains Mono labels, cursor-spotlight hover on cards, scroll-reveal entrance animations). Use this whenever Julian asks to make a repo's UI "match my other site", "look like the landing page", "match the Investment AI Suite branding/style", rebrand or restyle a tool's frontend, give a project "the same look" as his other investment repos, or build a new page/dashboard for one of his investment tools and wants it consistent with the rest of the suite — even if he doesn't name this skill directly. Works on any frontend stack (Tailwind projects get ready-made config/CSS snippets; plain CSS, CSS-in-JS, and component-library stacks like Chakra/MUI get translation guidance), not just Vite+React+Tailwind.
---

# Investment AI Suite UI v1 ("fintech noir")

A portable copy of the design system built for the Investment-AI-Suite landing page. Copy this whole
skill folder into another repo's `.claude/skills/` to bring the same look there. The goal is always
**restyle, not redesign**: keep the target site's content, layout, information architecture, and
functionality exactly as they are — change the paint, the type, the motion, and the surface treatment.

Read the reference files as you need them rather than all at once:

- `references/design-tokens.md` — the authoritative palette, type scale, spacing/radius and motion
  values. Read this first, every time — it's the source of truth and small details (exact hex values,
  easing curves) matter for the family resemblance to actually read as "the same site."
- `references/css-recipes.md` — framework-agnostic CSS for every reusable effect (glass cards, cursor
  spotlight, gradient text, animated connector lines, shimmer skeletons, scroll-reveal, focus rings,
  reduced-motion fallback). Read this when you're about to implement one of these effects, especially
  on a non-Tailwind stack.
- `references/component-patterns.md` — how the tokens and CSS recipes compose into the actual UI
  patterns on the landing page (hero, section heading, card, pipeline/connector diagram, data bars,
  skeleton loading, footer). Read this when deciding how to translate the target repo's existing
  components into this look.
- `assets/tailwind-theme.snippet.js` — the literal `theme.extend` block (colors + fontFamily) to merge
  into a Tailwind project's `tailwind.config.js`.
- `assets/design-system.css` — the full set of CSS-recipe classes as one paste-able file, for a global
  stylesheet in any stack (Tailwind `@layer components`, plain CSS, or a CSS file imported by a
  component-library project).
- `assets/google-fonts.snippet.html` — the `<link>` tags for the three fonts.

## Workflow

1. **Look before you touch.** Read the target repo's current styling entry points (global CSS,
   `tailwind.config.js`, theme provider, component library setup) and skim 2-3 representative
   components. You need to know what's already there before you decide what to port.

2. **Detect the stack** and pick a path:
   - **Tailwind (any React/Vue/Svelte flavor):** merge `assets/tailwind-theme.snippet.js`'s colors
     and fontFamily into the existing `tailwind.config.js` (keep any of the project's own tokens that
     don't collide — don't delete unrelated theme config). Append the relevant classes from
     `assets/design-system.css` into the project's global stylesheet, inside `@layer components` /
     `@layer utilities` as appropriate. Add the font `<link>` tags from
     `assets/google-fonts.snippet.html` to the HTML head.
   - **Component library (Chakra, MUI, Ant, etc.):** don't fight the library with ad-hoc overrides.
     Push the palette and fonts into the library's theme/token layer (e.g. Chakra `extendTheme` /
     `createSystem`, MUI `createTheme`) so every component inherits them for free, then layer the
     CSS-recipe effects (glass, spotlight, reveal) on top only where the landing page actually uses
     them — hero, cards, section headers — via the library's `sx`/`css` prop or a small number of
     global CSS classes from `design-system.css`.
   - **Plain CSS / CSS Modules / styled-components / component library (Chakra, MUI, etc.) / no
     build step at all:** rather than retyping the recipes, copy `assets/design-system.css` straight
     into the target repo (e.g. as `src/styles/ias-design-system.css`) and import/link it once at the
     app root — it's already framework-agnostic and namespaced (`ias-*` classes, `--ias-*` variables)
     so it won't collide with the project's own styles or a component library's CSS. Then apply the
     `ias-*` classes where `component-patterns.md` calls for them. For the cursor-spotlight effect
     you'll also need the small pointermove snippet from `references/css-recipes.md` (it's ~5 lines
     of vanilla JS/TS, no hook required if the stack isn't React).
   - **No visual frontend at all** (a CLI, a notebook, a backend-only service, a Slack bot): this
     skill doesn't apply — say so rather than inventing a UI that wasn't asked for.

3. **Apply the palette and type first**, before any of the decorative effects. Background goes to
   `ink-950`, body text to the slate scale, headings to Playfair Display, body to Inter, any
   micro-labels (tags, index numbers, timestamps, data figures) to JetBrains Mono. This alone gets a
   site most of the way to "family resemblance" even before glass/spotlight/reveal are added.

4. **Layer in the signature effects** from `component-patterns.md`, picked to fit what the page
   already has — don't force a pattern where it doesn't belong:
   - Cards, link tiles, nav items → `.glass` + `.spotlight`.
   - Any big display heading → gradient text (`.text-ivory-gradient`, reserve
     `.text-gold-gradient` for the one or two words that should carry emphasis — see the "restraint"
     note below).
   - Section openers → the mono-index + gold-rule + serif-H2 pattern.
   - Multi-step pipelines / architecture diagrams → the numbered-node + animated dashed connector
     pattern.
   - Numeric comparisons → the gradient bar-chart pattern (gold vs. mint series, dashed outline for
     missing data).
   - Anything that loads async → shimmer skeletons shaped like the eventual content, not a generic
     spinner.
   - Content entering on scroll → the `fly-in` reveal, staggered per item (see the stagger convention
     in `component-patterns.md` — it's what makes a list feel choreographed rather than random).
   - A full-bleed page background → the aurora gradient + drifting blurred orbs. Reuse gold / mint /
     ink; a project can swap in one extra brand-appropriate hue for a third orb (the landing page uses
     a blue, `#2b6cb0`, for exactly this reason) but shouldn't introduce an unrelated palette.

5. **Keep restraint.** Gold is the suite's signature accent, not a wash — on the landing page it marks
   interactive/active states, the one emphasized word in a heading, and the brand-colored glow effects;
   body text, most borders, and most backgrounds stay in the ink/slate range. Mint is the secondary
   accent (alternate data series, success/secondary CTA) — don't let gold and mint compete for the
   same role on one page. If a page is mostly gold, it will read as gaudy rather than branded.

6. **Don't drop existing substance.** Content, copy, data, routes, and interactive behavior should be
   unchanged unless the user explicitly asked for a content/IA change — this skill is about visual
   treatment. If an existing decorative element doesn't map cleanly onto a suite pattern (e.g. the
   landing page's hero price-chart squiggle only makes sense for a markets-flavored product), replace
   it with an equivalent *abstract* decorative element in the same style rather than importing something
   domain-irrelevant.

7. **Accessibility is part of the look, not optional polish.** Always include the gold focus-visible
   ring and the `prefers-reduced-motion` fallback block from `css-recipes.md` — the landing page ships
   both, and a restyle that drops them isn't actually matching it.

8. **Verify contrast and motion** on a real render (dev server + browser, or at minimum a careful read
   of the final CSS) before calling the restyle done — dark-on-dark glass panels and animated reveals
   are exactly the kind of thing that look fine in markup and wrong on screen.

## Versioning note

This is `v1` of the suite's shared look. If the landing page's design changes again later and you want
to propagate that forward, create a sibling `jyjulianwong-iais-ui-v2` skill (copy this one, update the
tokens/recipes/patterns from the new source) rather than silently drifting this one — that way repos
that copied `v1` keep a stable reference and can upgrade deliberately.
