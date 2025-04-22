// frontend/src/components/ControlPanel.js
import React, { useState, useRef, useCallback } from 'react';
import CollapsibleSection from './CollapsibleSection';

const ControlPanel = ({
    onLoadData, onSmooth, onFindPeaks, onFitPeaks, onAutoAnalyze, onFetchNist,
    engineStatus, isBusy, hasData, hasPeaks,
}) => {
    const fileInputRef = useRef(null);
    const [smoothParams, setSmoothParams] = useState({ method: 'savitzky_golay', window_length: 5, polyorder: 2 });
    const [peakParams, setPeakParams] = useState({ method: 'simple', prominence: '', height: '' });
    const [fitParams, setFitParams] = useState({ baseline_method: 'snip', selection_criterion: 'aic' });
    const [autoParams, setAutoParams] = useState({ elements: 'Fe, Si, Al, Ca, Mg, O, H, Na, K, Ti' });
    const [nistParams, setNistParams] = useState({ elements: 'Fe, Si, Al, Ca, Mg' });

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('spectrumFile', file);
            onLoadData(formData);
        }
        if (fileInputRef.current) { fileInputRef.current.value = ''; }
    };

    const handleParamChange = useCallback((setter, field, value, type) => {
        setter(prev => ({
            ...prev,
            [field]: type === 'number' ? (value === '' ? null : parseFloat(value)) : value
        }));
    }, []);

    const handleNistParamChange = useCallback((setter, field, value, type) => {
        setter(prev => ({ ...prev, [field]: value }));
    }, []);

    const triggerLoad = () => fileInputRef.current?.click();

    const runAuto = () => {
        const elementsList = autoParams.elements.split(',').map(el => el.trim()).filter(el => el);
        if (!elementsList.length) { alert("Please enter elements for Auto Analysis."); return; }
        onAutoAnalyze({ elements: elementsList });
    };

     const handleFetchNistClick = () => {
         const elementsList = nistParams.elements.split(',').map(el => el.trim()).filter(el => el);
         if (!elementsList.length) { alert("Please enter element symbols for NIST fetch."); return; }
         if (onFetchNist) { onFetchNist({ elements: elementsList }); }
    };


    // --- Get current params dynamically ---
    // FIX: Use console.warn instead of log.warn
    const getCurrentSmoothParams = () => {
        const base = { method: smoothParams.method };
        if (smoothParams.method === 'savitzky_golay') {
            const wl = parseInt(smoothParams.window_length, 10);
            const po = parseInt(smoothParams.polyorder, 10);
            if (wl && wl >= 3 && wl % 2 !== 0 && po >= 0 && po < wl) {
                base.window_length = wl;
                base.polyorder = po;
            } else {
                // Use console.warn for browser logging
                console.warn("Invalid Savitzky-Golay parameters entered.");
                alert("Invalid Savitzky-Golay parameters: Window must be odd, >=3. Order must be >=0 and < Window.");
                return null; // Prevent run if invalid
            }
        } else if (smoothParams.method === 'moving_average') {
             const ws = parseInt(smoothParams.window_size, 10);
             if (ws && ws >= 1) { base.window_size = ws; }
             else {
                 console.warn("Invalid Moving Average parameters.");
                 alert("Invalid Moving Average parameters: Window size must be >= 1.");
                 return null;
            }
        } else if (smoothParams.method === 'gaussian') {
             const sig = parseFloat(smoothParams.sigma);
             if (sig && sig > 0) { base.sigma = sig; }
             else {
                 console.warn("Invalid Gaussian parameters.");
                 alert("Invalid Gaussian parameters: Sigma must be > 0.");
                 return null;
             }
        }
        return base;
    }

     const getCurrentPeakParams = () => {
         const base = { method: peakParams.method };
         if (peakParams.prominence !== null && peakParams.prominence !== '') {
            const prom = parseFloat(peakParams.prominence);
            if (!isNaN(prom) && prom >= 0) base.prominence = prom; else { alert('Invalid Prominence value.'); return null;}
         }
         if (peakParams.height !== null && peakParams.height !== '') {
             const h = parseFloat(peakParams.height);
             if (!isNaN(h) && h >= 0) base.height = h; else { alert('Invalid Min Height value.'); return null;}
         }
         if (peakParams.method === 'advanced_nist') {
            const window_nm = parseFloat(peakParams.search_window_nm);
             if(!isNaN(window_nm) && window_nm > 0) base.search_window_nm = window_nm; else { alert('Invalid Search Window value.'); return null; }
            // Elements are not sent here for manual, rely on pre-fetched
         }
         return base;
     }

      const getCurrentFitParams = () => {
          return {
              baseline_method: fitParams.baseline_method,
              selection_criterion: fitParams.selection_criterion,
          };
      }

      // --- Click Handlers ---
      const handleSmoothClick = () => {
          const params = getCurrentSmoothParams();
          if(params) onSmooth(params);
          // Alert moved inside getCurrentSmoothParams
      }
       const handleFindPeaksClick = () => {
          const params = getCurrentPeakParams();
          if(params) onFindPeaks(params);
           // Alert moved inside getCurrentPeakParams (or could be added)
      }
       const handleFitPeaksClick = () => {
          const params = getCurrentFitParams();
          if(params) onFitPeaks(params);
      }

    // --- Render JSX ---
    return (
        <div className="panel h-full overflow-y-auto flex flex-col space-y-3 text-sm">
            {/* ... ( Title and Load button as before ) ... */}
            <h2 className="text-lg font-semibold text-cyan-400 border-b border-gray-600 pb-1 mb-1">Controls</h2>
            <div>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".csv,.txt,.asc,.spc" />
                <button onClick={triggerLoad} disabled={isBusy} className="btn btn-primary w-full"> {isBusy ? 'Working...' : 'Import Spectrum'} </button>
            </div>

            {/* --- Auto Analysis --- */}
            <CollapsibleSection title="▶ Auto Analysis" defaultOpen={true}>
                {/* ... (content as before) ... */}
                 <div className="space-y-3"> <div> <label htmlFor="auto-elements">Target Elements (CSV):</label> <input type="text" id="auto-elements" name="elements" value={autoParams.elements} onChange={(e) => handleParamChange(setAutoParams, 'elements', e.target.value, 'text')} className="w-full" placeholder="e.g. Fe, Si, Ca"/> </div> <button onClick={runAuto} disabled={isBusy || !hasData} className="btn btn-primary w-full"> {isBusy ? 'Analyzing...' : 'Run Auto Pipeline'} </button> </div>
            </CollapsibleSection>

            {/* --- Manual Steps --- */}
            <CollapsibleSection title="▶ Manual Steps">
                {/* Smoothing */}
                <div className="mb-3 p-2 border border-gray-700 rounded space-y-2">
                     {/* ... (Smoothing inputs and button as before, using handleSmoothClick) ... */}
                      <h4 className="font-medium text-gray-300">Smoothing</h4> <div className="grid grid-cols-3 gap-2 items-end"> <div className="col-span-3"> <label htmlFor="smooth-method">Method:</label> <select id="smooth-method" name="method" value={smoothParams.method} onChange={(e) => handleParamChange(setSmoothParams, 'method', e.target.value, 'text')} className="w-full"> <option value="savitzky_golay">Savitzky-Golay</option> <option value="moving_average">Moving Average</option> <option value="gaussian">Gaussian</option> </select> </div> {smoothParams.method === 'savitzky_golay' && ( <> <div><label htmlFor="smooth-wl">Window:</label><input type="number" id="smooth-wl" name="window_length" min="3" step="2" value={smoothParams.window_length || ''} onChange={(e) => handleParamChange(setSmoothParams, 'window_length', e.target.value, 'number')} className="w-full" placeholder="e.g. 5"/></div> <div><label htmlFor="smooth-po">Order:</label><input type="number" id="smooth-po" name="polyorder" min="0" step="1" max={(smoothParams.window_length || 3) - 1} value={smoothParams.polyorder ?? ''} onChange={(e) => handleParamChange(setSmoothParams, 'polyorder', e.target.value, 'number')} className="w-full" placeholder="e.g. 2"/></div> </> )} {smoothParams.method === 'moving_average' && ( <div><label htmlFor="smooth-ws">Window:</label><input type="number" id="smooth-ws" name="window_size" min="1" step="1" value={smoothParams.window_size || ''} onChange={(e) => handleParamChange(setSmoothParams, 'window_size', e.target.value, 'number')} className="w-full" placeholder="e.g. 3"/></div> )} {smoothParams.method === 'gaussian' && ( <div><label htmlFor="smooth-sigma">Sigma:</label><input type="number" id="smooth-sigma" name="sigma" min="0.1" step="0.1" value={smoothParams.sigma || ''} onChange={(e) => handleParamChange(setSmoothParams, 'sigma', e.target.value, 'number')} className="w-full" placeholder="e.g. 1.0"/></div> )} </div> <button onClick={handleSmoothClick} disabled={isBusy || !hasData} className="btn btn-secondary w-full mt-1"> Apply Smoothing </button>
                 </div>
                {/* Peak Detection */}
                 <div className="mb-3 p-2 border border-gray-700 rounded space-y-2">
                     {/* ... (Peak Detection inputs and button as before, using handleFindPeaksClick) ... */}
                      <h4 className="font-medium text-gray-300">Peak Detection</h4> <div className="grid grid-cols-2 gap-2 items-end"> <div> <label htmlFor="peak-method">Method:</label> <select id="peak-method" name="method" value={peakParams.method} onChange={(e) => handleParamChange(setPeakParams, 'method', e.target.value, 'text')} className="w-full"> <option value="simple">Simple</option> <option value="advanced_nist">Advanced (NIST)</option> </select> </div> {peakParams.method === 'advanced_nist' && ( <div> <label htmlFor="peak-window">Window (nm):</label> <input type="number" id="peak-window" name="search_window_nm" min="0.01" step="0.01" value={peakParams.search_window_nm || ''} onChange={(e) => handleParamChange(setPeakParams, 'search_window_nm', e.target.value, 'number')} className="w-full" placeholder="e.g. 0.2"/> </div> )} <div><label htmlFor="peak-prom">Prominence:</label><input type="number" id="peak-prom" name="prominence" min="0" step="0.1" value={peakParams.prominence ?? ''} onChange={(e) => handleParamChange(setPeakParams, 'prominence', e.target.value, 'number')} className="w-full" placeholder="Auto if blank"/></div> <div><label htmlFor="peak-height">Min Height:</label><input type="number" id="peak-height" name="height" min="0" step="1" value={peakParams.height ?? ''} onChange={(e) => handleParamChange(setPeakParams, 'height', e.target.value, 'number')} className="w-full" placeholder="Auto if blank"/></div> </div> <button onClick={handleFindPeaksClick} disabled={isBusy || !hasData} className="btn btn-secondary w-full mt-1"> Find Peaks </button> <p className="text-xs text-gray-500 italic mt-1">Note: 'Advanced (NIST)' requires fetching NIST data first.</p>
                 </div>
                 {/* Peak Fitting */}
                  <div className="p-2 border border-gray-700 rounded space-y-2">
                      {/* ... (Peak Fitting inputs and button as before, using handleFitPeaksClick) ... */}
                       <h4 className="font-medium text-gray-300">Peak Fitting</h4> <div className="grid grid-cols-2 gap-2 mb-2"> <div><label htmlFor="fit-baseline">Baseline:</label><select id="fit-baseline" name="baseline_method" value={fitParams.baseline_method} onChange={(e) => handleParamChange(setFitParams, 'baseline_method', e.target.value, 'text')} className="w-full"> <option value="snip">SNIP</option> <option value="polynomial">Polynomial</option> <option value="linear">Linear</option> <option value="none">None</option> </select></div> <div><label htmlFor="fit-criterion">Best Fit By:</label><select id="fit-criterion" name="selection_criterion" value={fitParams.selection_criterion} onChange={(e) => handleParamChange(setFitParams, 'selection_criterion', e.target.value, 'text')} className="w-full"> <option value="aic">AIC</option> <option value="bic">BIC</option> </select></div> </div> <button onClick={handleFitPeaksClick} disabled={isBusy || !hasPeaks} className="btn btn-secondary w-full mt-1"> Fit Detected Peaks </button>
                  </div>
            </CollapsibleSection>
        </div>
    );
};

// ... (keep CollapsibleSection component) ...
const LocalCollapsibleSection = ({ title, children, defaultOpen = false }) => { /* ... as before ... */
     const [isOpen, setIsOpen] = useState(defaultOpen);
     return ( <div className="border border-gray-700/50 rounded bg-gray-800/30 mb-2 shadow-sm"> <button className="w-full text-left px-3 py-1.5 bg-gray-700/60 hover:bg-gray-600/60 rounded-t text-cyan-300 font-medium text-sm focus:outline-none flex justify-between items-center transition duration-150 ease-in-out" onClick={() => setIsOpen(!isOpen)} aria-expanded={isOpen}> {title} <svg className={`w-4 h-4 transform transition-transform duration-200 ${isOpen ? 'rotate-0' : '-rotate-90'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg> </button> {isOpen && ( <div className="p-3 border-t border-gray-700/50"> {children} </div> )} </div> );
};

export default React.memo(ControlPanel);