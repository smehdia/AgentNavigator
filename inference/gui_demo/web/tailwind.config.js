/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sidebar: { DEFAULT: "#0f172a", hover: "#1e293b" },
        accent: { DEFAULT: "#6366f1", light: "#818cf8" },
      },
    },
  },
  plugins: [],
};
