const log = document.querySelector('#log');

function append(message) {
  const now = new Date().toISOString();
  log.textContent += `[${now}] ${message}\n`;
}

function setStatus(id, value) {
  document.querySelector(id).textContent = value;
}

window.__agentStaticTest = {
  moduleLoaded: false,
  chunkLoaded: false,
  jsonLoaded: false,
};

append('module script entered');
setStatus('#module-status', 'loaded');
window.__agentStaticTest.moduleLoaded = true;

const chunk = await import('./dynamic-view.js?v=20260516');
setStatus('#chunk-status', chunk.dynamicStatus());
window.__agentStaticTest.chunkLoaded = true;
append(`dynamic chunk says: ${chunk.dynamicStatus()}`);

const response = await fetch('/assets/data.json?cache_bust=20260516');
if (!response.ok) {
  throw new Error(`data.json request failed: ${response.status}`);
}
const data = await response.json();
setStatus('#json-status', data.status);
window.__agentStaticTest.jsonLoaded = true;
append(`json payload: ${JSON.stringify(data)}`);

document.body.dataset.ready = 'true';
append('browser load test complete');

