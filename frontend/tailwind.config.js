/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: { sans: ["system-ui", "-apple-system", '"Segoe UI"', "sans-serif"] },
      colors: {
        surface: "var(--surface-1)",
        page: "var(--page)",
        ink: { DEFAULT: "var(--text-primary)", 2: "var(--text-secondary)", muted: "var(--text-muted)" },
        line: "var(--gridline)",
        brand: { DEFAULT: "var(--series-1)", strong: "var(--brand-strong)" },
      },
    },
  },
  plugins: [],
};
