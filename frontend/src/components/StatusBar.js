// frontend/src/components/StatusBar.js
import React from 'react';

const StatusBar = ({ status, error, isBusy }) => {
    let textClass = 'text-gray-400';
    let bgClass = 'bg-gray-800/80 backdrop-blur-sm'; // Add slight blur effect
    let displayedStatus = status || 'Idle';
    let icon = null;

    if (error) {
        textClass = 'text-red-200';
        bgClass = 'bg-red-800/80 backdrop-blur-sm';
        displayedStatus = error.length > 100 ? error.substring(0, 97) + '...' : error; // Truncate long errors
        icon = <span className="mr-2 text-red-400">● Error</span>; // Error Symbol
    } else if (isBusy) {
        textClass = 'text-yellow-300';
        // bgClass = 'bg-yellow-800/80'; // Optional busy background
        displayedStatus = status || 'Processing...';
        icon = <span className="mr-2 animate-pulse">◌ Working</span>; // Busy Symbol (pulsing)
    } else if (status && status !== 'Idle' && status !== 'Ready.') {
         textClass = 'text-green-300'; // Success status
         icon = <span className="mr-2 text-green-400">● OK</span>; // OK symbol
    }


    return (
        <footer className={`px-4 py-1 text-xs border-t border-gray-700/50 flex justify-between items-center transition-colors duration-200 ${bgClass} ${textClass} shadow-inner`}
        >
            <div className="flex items-center">
                {icon}
                <span>{displayedStatus}</span>
            </div>
           {/* Add engine state or other info here if needed */}
        </footer>
    );
};
export default StatusBar;