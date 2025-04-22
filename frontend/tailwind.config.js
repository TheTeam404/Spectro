/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Tells Tailwind where to look for classes
  ],
  theme: {
    extend: {
        // Add custom sci-fi theme colors, fonts, animations here
        fontFamily: {
            mono: ['"Courier New"', 'monospace'], // Example mono font stack
        },
        colors: {
            'sci-cyan': 'rgb(0, 220, 220)',
            'sci-magenta': 'rgb(255, 0, 255)',
            'sci-green': 'rgb(0, 255, 0)',
            'sci-yellow': 'rgb(255, 255, 0)',
        },
        // Example animation for busy indicator
        keyframes: {
             pulse_subtle: {
                 '0%, 100%': { opacity: 0.7 },
                 '50%': { opacity: 1 },
             }
        },
        animation: {
             pulse_subtle: 'pulse_subtle 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        }
    },
  },
  plugins: [
      // require('@tailwindcss/forms'), // Optional: plugin for better form styling
  ],
}