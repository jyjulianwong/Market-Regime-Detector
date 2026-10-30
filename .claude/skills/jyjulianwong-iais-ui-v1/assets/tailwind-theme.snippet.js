/**
 * Merge this `theme.extend` content into the target repo's tailwind.config.js.
 * Keep any of the project's own existing extend keys that don't collide with these.
 */
module.exports = {
  colors: {
    ink: {
      950: '#060913',
      900: '#0a0f1f',
      800: '#10172b',
    },
    gold: {
      DEFAULT: '#e7c47e',
      soft: '#f3ddae',
      deep: '#a8803c',
    },
    mint: {
      DEFAULT: '#5ce1c6',
      deep: '#1f9e88',
    },
  },
  fontFamily: {
    serif: ['"Playfair Display"', 'Georgia', 'serif'],
    sans: ['Inter', 'system-ui', 'sans-serif'],
    mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
  },
}
