// frontend/src/App.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ControlPanel from './components/ControlPanel';
import PlotDisplay from './components/PlotDisplay';
import ResultsPanel from './components/ResultsPanel';
import StatusBar from './components/StatusBar';
import * as api from './api'; // Import API functions

function App() {
    // --- State ---
    const [spectrumData, setSpectrumData] = useState(null); // Raw { wl, int }
    const [processedData, setProcessedData] = useState(null); // Displayed { wl, int }
    const [baselineData, setBaselineData] = useState(null); // Baseline { wl, int }
    const [peaksData, setPeaksData] = useState(null); // Peaks array [ { index, wl, int, ...} ]
    const [fitResults, setFitResults] = useState(null); // Fits dict { peakIdx: {..., fit_curve_x, fit_curve_y} }
    const [quantResults, setQuantResults] = useState(null);
    const [cfLibsResults, setCfLibsResults] = useState(null);
    const [mlResults, setMlResults] = useState(null);

    const [selectedPeakIndex, setSelectedPeakIndex] = useState(null);
    const [isBusy, setIsBusy] = useState(false);
    const [engineStatus, setEngineStatus] = useState('Ready. Load data to begin.');
    const [errorMessage, setErrorMessage] = useState(null);
    const [roiHighlight, setRoiHighlight] = useState(null);

    // Ref to store latest state for callbacks if needed, avoiding stale closures
    const stateRef = useRef({});
    stateRef.current = { spectrumData, fitResults, selectedPeakIndex };

    // --- API Call Wrapper ---
    const handleApiCall = useCallback(async (apiFunc, params, statusMessage) => {
        setIsBusy(true);
        setErrorMessage(null);
        setEngineStatus(statusMessage || 'Processing...');
        console.log(`API Request: ${apiFunc.name}`, params); // Log request
        try {
            const result = await apiFunc(params);
            console.log(`API Response: ${apiFunc.name}`, result); // Log response
            // Use message from backend response if available
            setEngineStatus(result?.message || `${statusMessage || 'Action'} complete.`);
            return result; // Return full result object
        } catch (error) {
            console.error(`API Error (${apiFunc.name}):`, error);
            setErrorMessage(error.message || 'An unknown error occurred.');
            setEngineStatus('Error occurred. Check status bar.');
            return null;
        } finally {
            setIsBusy(false);
        }
    }, []);

    // --- Specific Action Handlers ---
    const handleLoadData = useCallback(async (formData) => {
        const result = await handleApiCall(api.loadData, formData, 'Loading data');
        if (result?.status === 'success' && result?.data?.spectrum) {
            setSpectrumData(result.data.spectrum);
            setProcessedData(result.data.spectrum); // Reset processed data
            setPeaksData(null); setFitResults(null); setQuantResults(null);
            setCfLibsResults(null); setMlResults(null); setSelectedPeakIndex(null);
            setBaselineData(null); setRoiHighlight(null);
        }
    }, [handleApiCall]);

    const handleSmooth = useCallback(async (params) => {
        const result = await handleApiCall(api.smoothData, params, 'Smoothing');
        if (result?.status === 'success' && result?.data?.smoothed_spectrum) {
            setProcessedData(result.data.smoothed_spectrum);
            setPeaksData(null); setFitResults(null); setSelectedPeakIndex(null); // Clear downstream
            setRoiHighlight(null);
        }
    }, [handleApiCall]);

    const handleFindPeaks = useCallback(async (params) => {
        const result = await handleApiCall(api.findPeaks, params, 'Finding peaks');
        if (result?.status === 'success' && result?.data?.peaks !== undefined) {
            console.log("Setting peaksData state with:", result.data.peaks); // Check for peaks array explicitly
            setPeaksData(result.data.peaks);
            setFitResults(null); setSelectedPeakIndex(null); // Clear downstream
            setRoiHighlight(null);
        }
        else {
            console.log("Not setting peaksData state.");}
    }, [handleApiCall]);

     const handleFitPeaks = useCallback(async (params) => {
        const result = await handleApiCall(api.fitPeaks, params, 'Fitting peaks');
        if (result?.status === 'success' && result?.data?.fit_results) {
            setFitResults(result.data.fit_results);
            setSelectedPeakIndex(null); // Clear selection after batch fit
            setRoiHighlight(null);
        }
    }, [handleApiCall]);

     const handleAutoAnalyze = useCallback(async (params) => {
        const result = await handleApiCall(api.runAutoAnalysis, params, 'Auto Analyzing');
        if (result?.status === 'success' && result?.data) {
            // Update relevant states based on backend response
            if (result.data.spectrum) setProcessedData(result.data.spectrum); // Show final processed spectrum
            else if(spectrumData) setProcessedData(spectrumData); // Fallback if backend doesn't send spectrum
            setPeaksData(result.data.peaks || null);
            setFitResults(result.data.fits || null); // Assume backend calculates fit curves if returning fits
            setQuantResults(result.data.analysis_type === 'Quant' ? result.data.output : null);
            setCfLibsResults(result.data.analysis_type === 'CF-LIBS' ? result.data.output : null);
            setMlResults(result.data.analysis_type === 'ML' ? result.data.output : null); // Adjust if ML returns differently
            setSelectedPeakIndex(null); setRoiHighlight(null);
        }
    }, [handleApiCall, spectrumData]); // Add spectrumData dependency


    // --- UI Event Handlers ---
    const handlePeakSelect = useCallback((peakIndex) => {
        console.log("Selected Peak Index:", peakIndex);
        setSelectedPeakIndex(peakIndex);
        // Highlight ROI on plot using latest state from ref
        const currentFitResults = stateRef.current.fitResults;
        const currentSpectrumData = stateRef.current.spectrumData;
        const fitInfo = currentFitResults?.[peakIndex];
        if (fitInfo?.roi_slice && currentSpectrumData?.wavelength) {
            const { start, stop } = fitInfo.roi_slice;
             // Ensure indices are within bounds
             if (start >= 0 && stop > start && stop <= currentSpectrumData.wavelength.length) {
                setRoiHighlight({
                    startWl: currentSpectrumData.wavelength[start],
                    endWl: currentSpectrumData.wavelength[stop-1]
                });
             } else {
                console.warn(`Invalid ROI slice [${start}, ${stop}) for spectrum length ${currentSpectrumData.wavelength.length}`);
                setRoiHighlight(null);
             }
        } else {
            setRoiHighlight(null);
        }
    }, []); // No changing dependencies

    const selectedPeakFitDetails = fitResults ? (fitResults[selectedPeakIndex] || null) : null;
    const hasData = !!spectrumData; // Boolean check if data is loaded
    const hasPeaks = !!peaksData && peaksData.length > 0; // Boolean check

    // --- Layout (Flexbox) ---
    return (
        <div className="flex flex-col h-screen overflow-hidden bg-gray-900">
            {/* Main Content Area */}
            <div className="flex flex-1 overflow-hidden">
                {/* Control Panel */}
                <aside className="w-72 xl:w-80 h-full overflow-y-auto flex-shrink-0 border-r border-gray-700 p-1">
                    <ControlPanel
                        onLoadData={handleLoadData}
                        onSmooth={handleSmooth}
                        onFindPeaks={handleFindPeaks}
                        onFitPeaks={handleFitPeaks}
                        onAutoAnalyze={handleAutoAnalyze}
                        engineStatus={engineStatus}
                        isBusy={isBusy}
                        hasData={hasData}
                        hasPeaks={hasPeaks}
                    />
                </aside>

                {/* Main Viewport (Plot + Results Side Panel) */}
                <main className="flex-1 flex overflow-hidden">
                    {/* Plot Area */}
                     <section className="flex-1 min-w-0 min-h-0 p-2"> {/* min-w-0 allows plot to shrink */}
                         <PlotDisplay
                             spectrumData={processedData || spectrumData} // Show processed if available
                             peaksData={peaksData}
                             fitResults={fitResults}
                             selectedPeakIndex={selectedPeakIndex}
                             onPeakClick={handlePeakSelect}
                             baselineData={baselineData} // Pass calculated baseline
                             roiHighlight={roiHighlight}
                         />
                     </section>

                     {/* Results Panel */}
                     <aside className="w-64 xl:w-72 h-full overflow-y-auto flex-shrink-0 border-l border-gray-700 p-1">
                          <ResultsPanel
                              selectedPeakFit={selectedPeakFitDetails}
                              quantResults={quantResults}
                              cfLibsResults={cfLibsResults}
                              mlResults={mlResults}
                          />
                     </aside>
                 </main>
            </div>

             {/* Status Bar */}
             <StatusBar status={engineStatus} error={errorMessage} isBusy={isBusy} />
        </div>
    );
}

export default App;