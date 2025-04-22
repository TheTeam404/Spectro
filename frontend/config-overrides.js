// frontend/config-overrides.js
const webpack = require('webpack');

module.exports = function override(config, env) {
    console.log("Applying config overrides..."); // Add console log for feedback

    // --- Fallbacks for Node Core Modules ---
    config.resolve.fallback = {
        ...config.resolve.fallback, // Keep any existing fallbacks
        "buffer": require.resolve("buffer/"),
        "stream": require.resolve("stream-browserify"),
        "assert": require.resolve("assert/"),
        "process": require.resolve("process/browser") // Ensure process fallback is explicitly listed here too
        // You might need others depending on deep dependencies:
        // "crypto": require.resolve("crypto-browserify"),
        // "http": require.resolve("stream-http"),
        // "https": require.resolve("https-browserify"),
        // "os": require.resolve("os-browserify/browser"),
        // "url": require.resolve("url/")
    };

    // --- Provide Necessary Plugins ---
    config.plugins = [
        ...(config.plugins || []), // Keep existing plugins
        // Provide Buffer and process globals
        new webpack.ProvidePlugin({
            process: 'process/browser.js', // Use the specific file path here!
            Buffer: ['buffer', 'Buffer'],
        }),
    ];

    // --- Ignore Source Map Warnings ---
    config.ignoreWarnings = [
         ...(config.ignoreWarnings || []),
         /Failed to parse source map/ // Ignore source map parsing warnings
    ];

    console.log("Config overrides applied.");
    return config;
}