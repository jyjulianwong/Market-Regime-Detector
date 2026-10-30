# CSS recipes — framework-agnostic

Every effect the landing page uses, as plain CSS. These are the same rules Tailwind's
`@layer components` block compiles down to, written here without any Tailwind dependency so they
drop into plain CSS, CSS Modules, a styled-components `createGlobalStyle`, or a component library's
global stylesheet unchanged. `assets/design-system.css` is this same content as one ready-to-paste
file.

Assume a dark page background (`#060913`) and light text (`#e2e8f0`) are already set on `body`.

## Ambient aurora background + drifting orbs

```css
.aurora {
  position: fixed;
  inset: 0;
  z-index: -1;
  overflow: hidden;
  background: radial-gradient(ellipse 120% 80% at 50% -10%, #111c38 0%, #060913 62%);
}

.aurora::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
  background-size: 64px 64px;
  -webkit-mask-image: radial-gradient(ellipse 70% 55% at 50% 15%, #000 0%, transparent 75%);
  mask-image: radial-gradient(ellipse 70% 55% at 50% 15%, #000 0%, transparent 75%);
}

.orb {
  position: absolute;
  border-radius: 9999px;
  filter: blur(100px);
  opacity: 0.32;
  will-change: transform;
  animation: drift 30s ease-in-out infinite alternate;
}

@keyframes drift {
  from { transform: translate3d(0, 0, 0) scale(1); }
  to   { transform: translate3d(60px, 40px, 0) scale(1.15); }
}
```

Usage: one `<div class="aurora">` fixed behind everything, containing 2-3 absolutely-positioned
`.orb` elements sized `~30-38rem`, colored with the brand hues (gold, mint-deep, and optionally one
extra hue), each given a distinct position and a negative `animation-delay` so they drift out of sync.

## Glass card

```css
.glass {
  position: relative;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.022));
  border: 1px solid rgba(255, 255, 255, 0.09);
  backdrop-filter: blur(20px) saturate(150%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.07),
    0 24px 48px -28px rgba(0, 0, 0, 0.7);
}
```

## Cursor-spotlight hover

CSS:

```css
.spotlight {
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease;
}

.spotlight::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s ease;
  background: radial-gradient(360px circle at var(--mx, 50%) var(--my, 50%), rgba(231, 196, 126, 0.15), transparent 60%);
}

.spotlight:hover,
.spotlight:focus-visible {
  transform: translateY(-3px);
  border-color: rgba(231, 196, 126, 0.38);
}

.spotlight:hover::before,
.spotlight:focus-visible::before {
  opacity: 1;
}
```

The element needs `position: relative` and `overflow: hidden` (or rely on `.glass`, which already
sets `position: relative`) plus `border-radius` set so `::before` can inherit it. Pair it with a
pointer-position tracker — React version:

```ts
function useSpotlight<T extends HTMLElement>() {
  return useCallback((e: PointerEvent<T>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--mx', `${e.clientX - rect.left}px`)
    e.currentTarget.style.setProperty('--my', `${e.clientY - rect.top}px`)
  }, [])
}
// <article className="glass spotlight" onPointerMove={onPointerMove}>
```

Vanilla JS version (any stack, no framework needed):

```js
document.querySelectorAll('.spotlight').forEach((el) => {
  el.addEventListener('pointermove', (e) => {
    const rect = el.getBoundingClientRect()
    el.style.setProperty('--mx', `${e.clientX - rect.left}px`)
    el.style.setProperty('--my', `${e.clientY - rect.top}px`)
  })
})
```

## Gradient text

```css
.text-gold-gradient {
  background: linear-gradient(120deg, #f3ddae 0%, #e7c47e 45%, #b98a3e 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.text-ivory-gradient {
  background: linear-gradient(180deg, #ffffff 30%, #a9b4c9 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
```

Use `text-ivory-gradient` for most of a heading, `text-gold-gradient` only on the one word that
should carry emphasis (see the restraint note in `design-tokens.md`).

## Animated connector line (for pipeline/step diagrams)

```css
.flow-line {
  position: relative;
  width: 2px;
  background-image: repeating-linear-gradient(to bottom, #e7c47e 0 5px, transparent 5px 11px);
  background-size: 2px 11px;
  animation: flow 1.1s linear infinite;
  opacity: 0.75;
}

.flow-arrow {
  position: absolute;
  bottom: -2px;
  left: 50%;
  width: 0;
  height: 0;
  transform: translateX(-50%);
  border-left: 4.5px solid transparent;
  border-right: 4.5px solid transparent;
  border-top: 6px solid #e7c47e;
}

@keyframes flow {
  to { background-position: 0 11px; }
}
```

Markup: a vertical `.flow-line` (give it a `min-height`) with a single `<span class="flow-arrow">`
child at its foot, between two step nodes. See `component-patterns.md` for the full node+connector
layout.

## Shimmer skeleton

```css
.shimmer {
  background: linear-gradient(100deg, rgba(255, 255, 255, 0.04) 20%, rgba(255, 255, 255, 0.11) 40%, rgba(255, 255, 255, 0.04) 60%);
  background-size: 200% 100%;
  animation: shimmer 1.6s linear infinite;
}

@keyframes shimmer {
  to { background-position: -200% 0; }
}
```

Build skeletons as plain divs shaped like the real content (a `.shimmer` block the width/height of a
title, another for each line of body text, etc.) rather than a generic spinner.

## Scroll-reveal entrance

```css
@keyframes fly-in {
  from { opacity: 0; transform: translateY(28px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.fly-in-hidden {
  opacity: 0;
  transform: translateY(28px) scale(0.98);
}

.animate-fly-in {
  animation: fly-in 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}
```

Start an element with class `fly-in-hidden`; once an `IntersectionObserver` reports it's in view, swap
the class to `animate-fly-in` (and optionally set `animation-delay` inline for staggering). Vanilla JS:

```js
document.querySelectorAll('[data-reveal]').forEach((el) => {
  const io = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) {
      el.classList.remove('fly-in-hidden')
      el.classList.add('animate-fly-in')
      io.disconnect()
    }
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' })
  el.classList.add('fly-in-hidden')
  io.observe(el)
})
```

React version is a tiny hook (`useInView`) returning `[ref, inView]` from the same
`IntersectionObserver` logic, with a wrapper component that toggles the class — see the landing
page's `useInView.ts` / `RevealOnScroll.tsx` if you have access to that repo as a direct reference.

## Focus ring & selection

```css
a:focus-visible,
button:focus-visible {
  outline: 2px solid #e7c47e;
  outline-offset: 3px;
  border-radius: 0.5rem;
}

::selection {
  background-color: #e7c47e33;
  color: #f3ddae;
}
```

## Reduced motion — always include this

```css
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }

  .orb, .flow-line, .shimmer, .animate-fly-in {
    animation: none;
  }

  .fly-in-hidden {
    opacity: 1;
    transform: none;
  }

  .spotlight, .spotlight:hover {
    transform: none;
  }
}
```

Extend this block to cover any other animated class you add (e.g. a `bar-grow` transition for charts
should also get `transition: none` here).
