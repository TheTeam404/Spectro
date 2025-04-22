// frontend/src/components/CollapsibleSection.js
import React, { useState } from 'react';

const CollapsibleSection = ({ title, children, defaultOpen = false }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="border border-gray-700/50 rounded bg-gray-800/30 mb-2 shadow-sm">
            <button
                className="w-full text-left px-3 py-1.5 bg-gray-700/60 hover:bg-gray-600/60 rounded-t text-cyan-300 font-medium text-sm focus:outline-none flex justify-between items-center transition duration-150 ease-in-out"
                onClick={() => setIsOpen(!isOpen)}
                aria-expanded={isOpen}
            >
                {title}
                {/* Animated chevron */}
                <svg className={`w-4 h-4 transform transition-transform duration-200 ${isOpen ? 'rotate-0' : '-rotate-90'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            {/* Conditional rendering with maybe a slight transition? */}
            {isOpen && (
                <div className="p-3 border-t border-gray-700/50">
                    {children}
                </div>
            )}
        </div>
    );
};
export default CollapsibleSection;