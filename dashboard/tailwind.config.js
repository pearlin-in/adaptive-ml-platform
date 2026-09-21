/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#12151A",
        canvas: "#F3F4F1",
        surface: "#FFFFFF",
        line: "#D7D9D4",
        muted: "#6B7268",
        stable: "#2F6F5E",
        "stable-soft": "#E4EEEB",
        canary: "#A67C2E",
        "canary-soft": "#F3E9D6",
        alert: "#B23A2E",
        "alert-soft": "#F5E1DE",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
    },
  },
  plugins: [],
}