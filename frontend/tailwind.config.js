/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        dash: {
          bg:       '#0a0a0a',
          surface:  '#111111',
          card:     '#171717',
          border:   '#272727',
          border2:  '#333333',
          accent:   '#c9a84c',
          purple:   '#9d7bb5',
          green:    '#5a9e7c',
          yellow:   '#c47a3a',
          red:      '#b05c5c',
          muted:    '#8a8a8a',
          dim:      '#484848',
          text:     '#e8e8e8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
