// custom.js - Bridge between React Flow Editor iframe and Chainlit Python backend

// 1. Send the initial graph to the iframe on load
const observeDOM = () => {
    const observer = new MutationObserver((mutations) => {
        const iframe = document.getElementById('editor-iframe');
        const dataDiv = document.getElementById('graph-data');
        if (iframe && dataDiv && !iframe.dataset.initialized) {
            iframe.dataset.initialized = 'true';
            iframe.onload = () => {
                try {
                    const raw = dataDiv.getAttribute('data-payload');
                    const graphJson = JSON.parse(raw || '{}');
                    iframe.contentWindow.postMessage({ type: 'INIT_GRAPH', ...graphJson }, '*');
                } catch(e) {
                    console.error("Failed to parse initial graph JSON", e);
                }
            };
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
};

observeDOM();


window.addEventListener('message', async (e) => {
    // React app now uses GRAPH_CONFIRMED but we can just let FastApi handle it via App.tsx fetch directly.
    // So we don't need to do anything here for SYNC_GRAPH.
});
