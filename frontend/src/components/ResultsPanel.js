// frontend/src/components/ResultsPanel.js
import React from 'react';

const ResultsPanel = ({
    selectedPeakFit, // Detailed fit info for one peak
    quantResults,    // { element: concentration, ... }
    cfLibsResults,   // { status, message, plasma_temperature_K, ... }
    mlResults,       // { status, message, output: { predictions?, transformed_data? } }
}) => {

    const formatNumber = (num, digits = 3, exponentialLimit = 6) => {
        if (num === null || num === undefined || num === '' || isNaN(num)) return 'N/A';
        const absNum = Math.abs(num);
        if (absNum === 0) return '0';
        if (absNum < 10**(-exponentialLimit) || absNum >= 10**exponentialLimit) {
            return num.toExponential(digits - 1);
        }
        // Adjust fixed digits based on magnitude, aiming for 'digits' significant figures
        let fixedDigits = digits - Math.floor(Math.log10(absNum)) - 1;
        fixedDigits = Math.max(0, fixedDigits); // Ensure non-negative
        // Limit max decimal places for very small numbers not caught by exp limit
        fixedDigits = Math.min(fixedDigits, digits + 1);
        return num.toFixed(fixedDigits);
    };

    const renderFitDetails = () => {
        if (!selectedPeakFit) return <p className="text-gray-500 italic text-center mt-4">Select a fitted peak to view details.</p>;
        if (selectedPeakFit.status !== 'Success') {
            return <p className="text-red-400 px-2">Fit Failed: {selectedPeakFit.message || 'Unknown reason.'}</p>;
        }

        return (
            <div className="text-xs space-y-1 px-2">
                <p><span className="font-semibold text-cyan-400 w-20 inline-block">Best Fit:</span> {selectedPeakFit.best_fit_type}</p>
                <p><span className="font-semibold text-cyan-400 w-20 inline-block">R²:</span> {formatNumber(selectedPeakFit.best_metrics?.r_squared, 4)}</p>
                <p><span className="font-semibold text-cyan-400 w-20 inline-block">AIC:</span> {formatNumber(selectedPeakFit.best_metrics?.aic, 2)}</p>
                <p><span className="font-semibold text-cyan-400 w-20 inline-block">BIC:</span> {formatNumber(selectedPeakFit.best_metrics?.bic, 2)}</p>
                <p><span className="font-semibold text-cyan-400 w-20 inline-block">Flags:</span> {selectedPeakFit.quality_flags?.join(', ') || 'None'}</p>
                <p className="font-semibold text-cyan-400 mt-2">Parameters:</p>
                <ul className="list-none pl-1 space-y-0.5">
                    {selectedPeakFit.best_params && Object.entries(selectedPeakFit.best_params).map(([key, value]) => (
                        <li key={key} className="ml-2"><span className="text-gray-400 w-20 inline-block">{key}:</span> {formatNumber(value)}</li>
                    ))}
                </ul>
                 {/* Optional: Display parameter errors */}
                 {selectedPeakFit.param_errors && (
                    <>
                        <p className="font-semibold text-cyan-400 mt-1">Param Errors (+/-):</p>
                        <ul className="list-none pl-1 space-y-0.5">
                            {Object.entries(selectedPeakFit.param_errors).map(([key, value]) => (
                                <li key={key+'-err'} className="ml-2"><span className="text-gray-400 w-20 inline-block">{key}:</span> {formatNumber(value, 2)}</li>
                             ))}
                         </ul>
                     </>
                 )}
            </div>
        );
    };

     const renderAnalysisResults = (title, resultsData, format = 'percent') => {
        if (!resultsData || Object.keys(resultsData).length === 0) return null;
         return (
            <div className="mt-3 px-2">
                 <h4 className="font-medium text-gray-300 border-t border-gray-700 pt-2 mb-1">{title}</h4>
                 <table className="w-full text-left text-xs">
                     <thead>
                         <tr className="text-gray-400"><th>Element</th><th>Value{format==='percent' ? ' (%)' : ''}</th></tr>
                     </thead>
                     <tbody>
                     {Object.entries(resultsData)
                         .sort(([,a],[,b]) => b - a) // Sort descending by value
                         .map(([el, conc]) => (
                         <tr key={el} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                             <td className="py-0.5">{el}</td>
                             <td className="py-0.5">{formatNumber(conc, format === 'percent' ? 2 : 3)}</td>
                         </tr>
                      ))}
                     </tbody>
                 </table>
            </div>
         );
     };

     const renderCfLibsExtras = () => {
         if (!cfLibsResults || cfLibsResults.status !== 'Success') return null;
          const temp = cfLibsResults.plasma_temperature_K;
          const ne = cfLibsResults['electron_density_cm-3'];
          const temp_err = cfLibsResults.temperature_uncertainty_K;
         return (
             <div className="text-xs space-y-0.5 px-2 mt-1">
                 {temp && <p><span className="text-gray-400 w-28 inline-block">Plasma Temp (K):</span> {formatNumber(temp, 0)} {temp_err ? `± ${formatNumber(temp_err, 0)}` : ''}</p>}
                 {ne && <p><span className="text-gray-400 w-28 inline-block">Electron Dens (cm⁻³):</span> {formatNumber(ne, 2)}</p>}
             </div>
         )
     }

     const renderMLResultsDisplay = () => {
        if (!mlResults || mlResults.status !== 'Success') return null;
        const output = mlResults.output || {}; // Use output key from apply_model structure
        return (
             <div className="mt-3 px-2">
                <h4 className="font-medium text-gray-300 border-t border-gray-700 pt-2 mb-1">ML Results ({mlResults.model_type || 'Unknown'})</h4>
                {output.predictions && (
                    <div><span className="text-gray-400">Predictions:</span> <pre className="text-xs bg-gray-900/50 p-1 rounded overflow-x-auto">{JSON.stringify(output.predictions, null, 2)}</pre></div>
                )}
                 {output.transformed_data && (
                     <div><span className="text-gray-400">Transformed Data (PCA/etc.):</span> <pre className="text-xs bg-gray-900/50 p-1 rounded overflow-x-auto">Shape: {output.transformed_data?.length} x {output.transformed_data?.[0]?.length} (Showing first few)</pre></div> // Just show shape example
                 )}
                 {output.explained_variance_ratio && (
                      <div><span className="text-gray-400">Explained Var % (PCA):</span> {output.explained_variance_ratio.map(v => formatNumber(v*100, 1)).join(', ')}</div>
                 )}
             </div>
         );
     }

    return (
        <div className="panel h-full overflow-y-auto flex flex-col text-xs">
            <h2 className="text-base font-semibold text-cyan-400 border-b border-gray-600 pb-1 mb-2 px-1">Results & Fit Details</h2>

            {/* Peak Fit Details Area */}
            <div className="mb-3 border-b border-gray-700 pb-2">
                <h3 className="text-sm font-medium text-gray-300 mb-1 px-1">Selected Peak Fit Details</h3>
                {renderFitDetails()}
            </div>

            {/* Analysis Results Area */}
             {cfLibsResults && cfLibsResults.status === 'Success' && renderCfLibsExtras()}
             {renderAnalysisResults("CF-LIBS Composition", cfLibsResults?.composition_atom_percent || cfLibsResults?.composition_weight_percent)}
             {renderAnalysisResults("Quantification", quantResults)}
             {renderMLResultsDisplay()}

        </div>
    );
};

export default React.memo(ResultsPanel);