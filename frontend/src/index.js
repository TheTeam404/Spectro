// frontend/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Import global styles (including Tailwind)
import App from './App'; // Import main app component

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  // StrictMode removed temporarily if Plotly causes double-render issues in dev
  // <React.StrictMode>
    <App />
  // </React.StrictMode>
);