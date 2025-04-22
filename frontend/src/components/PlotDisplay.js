// frontend/src/components/PlotDisplay.js
import React, { useState, useEffect, useRef, useMemo } from 'react';
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js'; // Import base plotly for potential direct manipulation

const PlotDisplay = ({
    spectrumData, // { wavelength: [], intensity: [] } or null
    peaksData,    // Array of peak objects { index, wavelength, intensity, ... } or null
    fitResults,   // Dict { peakIndex: results } or null
    selectedPeakIndex, // number or null
    onPeakClick, // function(peakIndex)
    baselineData, // Optional { wavelength: [], intensity: [] }
    roiHighlight, // Optional { startWl: number, endWl: number }
}) => {
    const plotRef = useRef(null); // Ref to access Plotly instance

    // Define Sci-Fi Colors (Consider moving to a theme file)
    const sciFiColors = useMemo(() => ({
        spectrum: 'rgb(0, 220, 220)', // Brighter Cyan
        peaks: 'rgb(255, 0, 255)',    // Magenta
        fits: 'rgb(0, 255, 0)',      // Bright Green
        selectedPeak: 'rgb(255, 255, 0)', // Yellow
        baseline: 'rgba(100, 116, 139, 0.7)', // slate-500/70
        roi: 'rgba(0, 255, 0, 0.1)', // Lighter green fill for ROI
        grid: 'rgba(55, 65, 81, 0.6)', // gray-700/60
        text: 'rgb(209, 213, 219)', // gray-300
        background: 'transparent',
        paper: 'transparent',
    }), []);

    // --- Generate Plot Traces ---
    const plotData = useMemo(() => {
        const traces = [];

        // 1. Spectrum Trace
        if (spectrumData?.wavelength?.length > 0) {
            traces.push({
                x: spectrumData.wavelength,
                y: spectrumData.intensity,
                type: 'scattergl', // Use WebGL for performance
                mode: 'lines',
                name: 'Spectrum',
                line: { color: sciFiColors.spectrum, width: 1.5 },
            });
        } else {
            traces.push({ x: [], y: [], type: 'scattergl', name:'Spectrum' }); // Placeholder
        }

        // 2. Baseline Trace (Optional)
        if (baselineData?.wavelength?.length > 0) {
             traces.push({
                x: baselineData.wavelength,
                y: baselineData.intensity,
                type: 'scattergl',
                mode: 'lines',
                name: 'Baseline',
                line: { color: sciFiColors.baseline, width: 1, dash: 'dash' },
             });
        }

        // 3. Peaks Trace
        if (peaksData?.length > 0) { // Check if peaksData exists and has items
            traces.push({
                x: peaksData.map(p => p.wavelength), // Maps wavelength from each peak object
                y: peaksData.map(p => p.intensity),  // Maps intensity from each peak object
                type: 'scattergl',
                mode: 'markers',
                name: 'Peaks',
                marker: { color: sciFiColors.peaks, size: 8, symbol: 'cross-thin-open' },
                customdata: peaksData.map(p => p.index), // Store original index for clicking
                hovertemplate: 'W: %{x:.3f}<br>I: %{y:.1f}<br>Index: %{customdata}<extra></extra>',
            });
        }

        // 4. Fit Traces (using data from backend)
        if (fitResults && spectrumData?.wavelength && Object.keys(fitResults).length > 0) {
            Object.entries(fitResults).forEach(([peakIdxStr, fitInfo]) => {
                if (fitInfo?.status !== 'Success' || !fitInfo.best_fit_type) return;

                const peakIndex = parseInt(peakIdxStr, 10);
                const fitFuncName = fitInfo.best_fit_type;
                const fitX = fitInfo.fit_curve_x; // Assume backend provided X array
                const fitY = fitInfo.fit_curve_y; // Assume backend provided Y array

                if (fitX && fitY && fitX.length === fitY.length && fitX.length > 1) {
                    traces.push({
                        x: fitX,
                        y: fitY,
                        type: 'scattergl',
                        mode: 'lines',
                        name: `Fit ${peakIndex} (${fitFuncName})`,
                        line: { color: sciFiColors.fits, width: 1.5, dash: 'dot' },
                        opacity: 0.85,
                        hoverinfo: 'skip',
                    });
                } else if (fitInfo.best_params?.center && fitInfo.best_params?.amplitude) {
                    // Fallback: Show marker if curve data missing but params exist
                     traces.push({
                         x: [fitInfo.best_params.center], y: [fitInfo.best_params.amplitude],
                         mode: 'markers', type: 'scattergl', name: `Fit ${peakIndex} (Pt)`,
                         marker: { color: sciFiColors.fits, size: 5, symbol: 'diamond-open' },
                         hoverinfo: 'skip'
                     });
                }
            });
        }

        // 5. Selected Peak Highlight
        if (selectedPeakIndex !== null && peaksData) {
             const selectedPeak = peaksData.find(p => p.index === selectedPeakIndex);
             if (selectedPeak) {
                 traces.push({
                     x: [selectedPeak.wavelength],
                     y: [selectedPeak.intensity],
                     type: 'scattergl', mode: 'markers', name: 'Selected',
                     marker: { color: 'rgba(0,0,0,0)', size: 16, symbol: 'circle-open', line: { width: 2.5, color: sciFiColors.selectedPeak } },
                     hoverinfo: 'skip',
                 });
             }
        }

        return traces;
    }, [spectrumData, peaksData, fitResults, selectedPeakIndex, baselineData, sciFiColors]);


    // --- Generate Plot Layout ---
    const plotLayout = useMemo(() => {
        const layout = {
            uirevision: 'data', // Keep zoom only when 'data' reference changes (i.e. loading new file)
            // uirevision: 'constant', // Alt: Keep zoom always unless layout explicitly changes
            hovermode: 'closest',
            showlegend: true,
            legend: { x: 1, xanchor: 'right', y: 1, yanchor: 'top', bgcolor: 'rgba(31, 41, 55, 0.8)', bordercolor: sciFiColors.grid, borderwidth: 1, font: { color: sciFiColors.text } },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            xaxis: { title: 'Wavelength (nm)', color: sciFiColors.text, gridcolor: sciFiColors.grid, zerolinecolor: sciFiColors.grid, automargin: true },
            yaxis: { title: 'Intensity (a.u.)', color: sciFiColors.text, gridcolor: sciFiColors.grid, zerolinecolor: sciFiColors.grid, exponentformat: 'e', automargin: true },
            plot_bgcolor: sciFiColors.background,
            paper_bgcolor: sciFiColors.paper,
            font: { family: 'monospace, Consolas, "Courier New"', color: sciFiColors.text },
            shapes: [], // For ROI highlight
            // Performance optimizations
            // datarevision: new Date().getTime(), // Force redraw on data change if uirevision isn't working well
        };

        // Zooming logic for selected peak
        if (selectedPeakIndex !== null && peaksData) {
            const selectedPeak = peaksData.find(p => p.index === selectedPeakIndex);
            const fitInfo = fitResults?.[selectedPeakIndex];
            let center = selectedPeak?.wavelength;
            let width = 5; // Default zoom width (nm)

             if (fitInfo?.status === 'Success' && fitInfo.roi_slice && spectrumData?.wavelength) {
                  const { start, stop } = fitInfo.roi_slice;
                  const roiWl = spectrumData.wavelength.slice(start, stop);
                  if (roiWl?.length > 1) {
                      center = fitInfo.best_params?.center ?? center; // Prefer fitted center
                      width = (roiWl[roiWl.length-1] - roiWl[0]); // Use ROI width
                  }
             } else if (selectedPeak?.width_fwhm_wl > 0) {
                 // Use FWHM from peak detection if fit/ROI unavailable
                 width = selectedPeak.width_fwhm_wl * 5; // Zoom ~5x FWHM
             }

             if(center) {
                 const zoomMargin = Math.max(width * 0.2, 0.5); // Add margin, at least 0.5nm
                 layout.xaxis.range = [center - width / 2 - zoomMargin, center + width / 2 + zoomMargin];
                 layout.yaxis.autorange = true; // Autorange Y on zoom
             } else {
                  layout.xaxis.autorange = true;
                  layout.yaxis.autorange = true;
             }

        } else {
             layout.xaxis.autorange = true;
             layout.yaxis.autorange = true;
        }

        // Add ROI highlight shape
        if (roiHighlight) {
             layout.shapes.push({
                 type: 'rect', xref: 'x', yref: 'paper', x0: roiHighlight.startWl, x1: roiHighlight.endWl,
                 y0: 0, y1: 1, fillcolor: sciFiColors.roi, line: { width: 0 }, layer: 'below',
             });
        }

        return layout;

    }, [selectedPeakIndex, fitResults, roiHighlight, peaksData, spectrumData, sciFiColors]);


    // --- Event Handlers ---
    const handlePlotClick = (eventData) => {
        if (!eventData?.points?.length || !onPeakClick) return;
        const point = eventData.points[0];
        // Check if clicking on the 'Peaks' trace by name or index might be safer
        if (point.customdata !== undefined && point.curveNumber < plotData.length && plotData[point.curveNumber]?.name === 'Peaks') {
            onPeakClick(point.customdata); // Pass the peak index back
        }
    };

    return (
        <div className="w-full h-full bg-gray-900/50 border border-gray-700 rounded-lg overflow-hidden shadow-inner">
            <Plot
                ref={plotRef}
                data={plotData}
                layout={plotLayout}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'], // Example removals
                    // modeBarButtonsToAdd: [] // Add custom buttons if needed
                }}
                onClick={handlePlotClick}
            />
        </div>
    );
};

export default React.memo(PlotDisplay); // Memoize to prevent unnecessary re-renders