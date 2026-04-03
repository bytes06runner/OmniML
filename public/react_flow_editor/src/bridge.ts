import type { Node, Edge } from '@xyflow/react';

export const SESSION_ID_KEY = 'omniml_session_id';

export function getSessionId(): string {
  return new URLSearchParams(window.location.search).get('session_id') 
    || localStorage.getItem(SESSION_ID_KEY) 
    || 'default';
}

export async function syncGraph(nodes: Node[], edges: Edge[]): Promise<boolean> {
  const graph = { nodes, edges, session_id: getSessionId() };
  
  window.parent.postMessage(
    { type: 'GRAPH_CONFIRMED', graph },
    '*'
  );
  
  try {
    const res = await fetch('/sync-graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graph)
    });
    return res.ok;
  } catch {
    return false;
  }
}

export function receiveInitialGraph(
  callback: (graph: { nodes: Node[], edges: Edge[] }) => void
) {
  window.addEventListener('message', (e) => {
    if (e.data?.type === 'LOAD_GRAPH' || e.data?.type === 'INIT_GRAPH') {
      callback(e.data.graph || e.data);
    }
  });
  window.parent.postMessage({ type: 'EDITOR_READY' }, '*');
}
