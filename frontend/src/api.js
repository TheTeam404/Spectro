// frontend/src/api.js
import axios from 'axios';

// Use environment variable for API base URL if available, otherwise default
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '/api';

// Configure axios default timeout
axios.defaults.timeout = 45000; // 45 seconds (increase for potentially long analysis)

/**
 * Handles API requests, including error formatting.
 * @param {Promise<axios.AxiosResponse<any>>} requestPromise Axios request promise
 * @returns {Promise<any>} Resolves with response data or rejects with formatted error message.
 */
const handleRequest = async (requestPromise) => {
    try {
        const response = await requestPromise;
        // Check for successful backend status within the response data
        if (response.data && response.data.status === 'error') {
            console.warn("API call successful but returned error status:", response.data.message);
            throw new Error(response.data.message || 'Backend returned an error status.');
        }
        return response.data; // Return the full response data structure
    } catch (error) {
        let errorMessage = 'An unknown network error occurred.';
        if (error.response) {
            // Server responded with a status code outside the 2xx range
            errorMessage = `Error ${error.response.status}: ${error.response.data?.message || error.message || error.response.statusText}`;
        } else if (error.request) {
            errorMessage = 'Network error: No response received from server (check connection or timeout).';
        } else if (error.message?.includes('timeout')) {
            errorMessage = 'Request timed out. The operation may be taking too long.';
        }
         else {
            errorMessage = `Request setup error: ${error.message}`;
        }
        console.error("API Error:", error); // Log the full technical error
        // Throw the user-friendly message
        throw new Error(errorMessage);
    }
};

// --- API Functions ---

export const loadData = (formData) => {
    return handleRequest(axios.post(`${API_BASE_URL}/load`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    }));
};

export const getStatus = () => {
    return handleRequest(axios.get(`${API_BASE_URL}/status`));
};

export const smoothData = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/smooth`, params));
};

export const findPeaks = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/find_peaks`, params));
};

export const fitPeaks = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/fit_peaks`, params));
};

export const runAutoAnalysis = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/run_auto`, params));
};

export const runQuantification = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/quantify`, params));
};

export const runCFLibs = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/cf_libs`, params));
};

export const runML = (params) => {
    return handleRequest(axios.post(`${API_BASE_URL}/ml`, params));
};

// Add other specific API calls for features like noise analysis, NIST fetching trigger etc.+