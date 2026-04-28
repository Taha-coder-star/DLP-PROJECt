/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        accent: {
          DEFAULT: '#6366f1',
          light: '#818cf8',
          muted: '#4f46e5',
        },
        surface: {
          DEFAULT: '#0a0e1a',
          raised: '#111827',
          hover: '#1a2035',
        },
      },
    },
  },
  plugins: [],
};
