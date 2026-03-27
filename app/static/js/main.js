// SPIKD — main.js
// Minimal JS for auth redirect handling on 401 responses

(function () {
  const originalFetch = window.fetch.bind(window);
  window.fetch = async function (...args) {
    const response = await originalFetch(...args);
    if (response.status === 401) {
      window.location.href = '/login';
    }
    return response;
  };
})();
