import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  useReactFlow,
  MarkerType,
  ConnectionLineType
} from '@xyflow/react';
import type { Connection, Edge, Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import CustomLayerNode from './CustomLayerNode';
import Sidebar from './Sidebar';
import { getLayoutedElements } from './utils/layout';

const nodeTypes = { customNode: CustomLayerNode };

type ValidationError = {
  id: string;
  message: string;
  severity: 'error' | 'warning';
};

type SyncState = 'idle' | 'syncing' | 'success' | 'error';

const AppFlow = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [syncState, setSyncState] = useState<SyncState>('idle');
  const [syncCount, setSyncCount] = useState({ nodes: 0, edges: 0 });
  const [suggesting, setSuggesting] = useState(false);
  const [rationale, setRationale] = useState('');
  const { fitView } = useReactFlow();

  const handleSuggestArchitecture = useCallback(async () => {
    setSuggesting(true);
    setRationale('');
    
    try {
      const params = new URLSearchParams(window.location.search);
      const session_id = params.get('session_id') || 'default';
      
      const res = await fetch(`/suggest-architecture?session_id=${session_id}`);
      // Check content type before parsing
      const contentType = res.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        const text = await res.text();
        console.error('[OmniML] /suggest-architecture returned non-JSON:', text.slice(0, 200));
        setSuggesting(false);
        return;
      }

      const data = await res.json();
      
      if (!data.ok || !data.graph?.nodes?.length) {
        console.error('[OmniML] Suggest failed:', data.error);
        setSuggesting(false);
        return;
      }
      
      // Wire onDelete into every node's data
      const nodesWithDelete = data.graph.nodes.map((n: Node) => ({
        ...n,
        data: {
          ...n.data,
          id: n.id,
          onDelete: (id: string) => {
            setNodes(prev => prev.filter(node => node.id !== id));
            setEdges(prev => prev.filter(
              e => e.source !== id && e.target !== id
            ));
          }
        }
      }));

      const { nodes: ln, edges: le } = getLayoutedElements(
        nodesWithDelete,
        data.graph.edges || [],
        'TB'
      );
      
      setNodes(ln);
      setEdges(le);
      setRationale(data.rationale || '');
      
      setTimeout(() => {
        fitView({ padding: 0.25, duration: 700 });
      }, 60);
      
    } catch (err) {
      console.error('Suggest architecture error:', err);
    } finally {
      setSuggesting(false);
    }
  }, [setNodes, setEdges, fitView]);

  const graphLoadedRef = useRef(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const session_id = params.get('session_id') || 'default';

    const loadGraph = async () => {
      try {
        const res = await fetch(
          `/get-architect-graph?session_id=${session_id}`
        );
        const data = await res.json();

        if (data.ok && data.graph?.nodes?.length > 0) {
          console.log('[OmniML] Graph fetched from server, nodes:',
                      data.graph.nodes.length);
          graphLoadedRef.current = true;

          // Wire onDelete into every node's data
          const nodesWithDelete = data.graph.nodes.map((n: Node) => ({
            ...n,
            data: {
              ...n.data,
              id: n.id,
              onDelete: (id: string) => {
                setNodes(prev => prev.filter(node => node.id !== id));
                setEdges(prev => prev.filter(
                  e => e.source !== id && e.target !== id
                ));
              }
            }
          }));

          const { nodes: ln, edges: le } = getLayoutedElements(
            nodesWithDelete,
            data.graph.edges || [],
            'TB'
          );
          setNodes(ln);
          setEdges(le);
          setTimeout(() => {
            fitView({ padding: 0.25, duration: 700 });
          }, 80);
        } else {
          console.warn('[OmniML] No graph from server:', data.error);
        }
      } catch (err) {
        console.error('[OmniML] Failed to fetch graph:', err);
      }
    };

    // Try immediately, then retry twice with backoff
    // in case the architect node hasn't stored yet
    loadGraph();
    setTimeout(loadGraph, 1000);
    setTimeout(loadGraph, 2500);

    // Keep postMessage listener as fallback only
    const handleMessage = (e: MessageEvent) => {
      if (e.data?.type === 'LOAD_GRAPH' && 
          e.data.graph?.nodes?.length > 0 &&
          !graphLoadedRef.current) {
        console.log('[OmniML] LOAD_GRAPH via postMessage, nodes:',
                    e.data.graph.nodes.length);
        graphLoadedRef.current = true;
        const { nodes: ln, edges: le } = getLayoutedElements(
          e.data.graph.nodes.map((n: any) => ({
            ...n,
            data: {
              ...n.data,
              id: n.id,
              onDelete: (id: string) => {
                setNodes(prev => prev.filter(node => node.id !== id));
                setEdges(prev => prev.filter(
                  e => e.source !== id && e.target !== id
                ));
              }
            }
          })),
          e.data.graph.edges || [],
          'TB'
        );
        setNodes(ln);
        setEdges(le);
        setTimeout(() => fitView({ padding: 0.25, duration: 700 }), 80);
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [setNodes, setEdges, fitView]);

  const onConnect = useCallback((connection: Connection) => {
    setEdges(prev => addEdge({
      ...connection,
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#6366f1',
      },
    } as any, prev));
  }, [setEdges]);

  const onNodesDelete = useCallback((deleted: Node[]) => {
    setEdges(prev => prev.filter(
      e => !deleted.find(
        d => d.id === e.source || d.id === e.target
      )
    ));
  }, [setEdges]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const { screenToFlowPosition } = useReactFlow();

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const nodeType = event.dataTransfer.getData('application/reactflow');
      if (!nodeType) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const defaultParams: Record<string, any> = {
        Dense:       { units: 128, activation: 'relu' },
        LSTM:        { units: 64, return_sequences: false },
        Conv1D:      { filters: 32, kernel_size: 3, activation: 'relu' },
        Dropout:     { rate: 0.3 },
        BatchNorm1d: {},
        ReLU:        {},
        Sigmoid:     {},
        Input:       { shape: '30,' },
        Output:      { units: 1, activation: 'sigmoid' },
      };

      const newId   = `${nodeType}_${Date.now()}`;
      const newNode: Node = {
        id:       newId,
        type:     'customNode',
        position,
        width:    220,
        height:   80,
        data: {
          label:    newId,
          nodeType: nodeType,
          params:   defaultParams[nodeType] || {},
          id:       newId,
          onDelete: (id: string) => {
            setNodes(prev => prev.filter(n => n.id !== id));
            setEdges(prev => prev.filter(
              e => e.source !== id && e.target !== id
            ));
          },
        },
      };
      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes, setEdges, screenToFlowPosition]
  );

  const handleValidate = useCallback(() => {
    const errors: ValidationError[] = [];

    // Rule 1: must have exactly one Input node
    const inputNodes = nodes.filter(n => n.data.nodeType === 'Input');
    if (inputNodes.length === 0) {
      errors.push({ id: 'no-input', message: 'Missing Input node — every network needs one', severity: 'error' });
    } else if (inputNodes.length > 1) {
      errors.push({ id: 'multi-input', message: `${inputNodes.length} Input nodes found — only 1 allowed`, severity: 'error' });
    }

    // Rule 2: must have exactly one Output node
    const outputNodes = nodes.filter(n => n.data.nodeType === 'Output');
    if (outputNodes.length === 0) {
      errors.push({ id: 'no-output', message: 'Missing Output node — every network needs one', severity: 'error' });
    } else if (outputNodes.length > 1) {
      errors.push({ id: 'multi-output', message: `${outputNodes.length} Output nodes found — only 1 allowed`, severity: 'error' });
    }

    // Rule 3: no disconnected islands
    if (nodes.length > 1) {
      const connectedIds = new Set<string>();
      edges.forEach(e => { connectedIds.add(e.source); connectedIds.add(e.target); });
      nodes.forEach(n => {
        if (!connectedIds.has(n.id)) {
          errors.push({ id: `island-${n.id}`, message: `"${n.data.label}" is disconnected — connect it or delete it`, severity: 'error' });
        }
      });
    }

    // Rule 4: no cycles (DAG check via DFS)
    const adj: Record<string, string[]> = {};
    nodes.forEach(n => { adj[n.id] = []; });
    edges.forEach(e => { if (adj[e.source]) adj[e.source].push(e.target); });
    const visited = new Set<string>();
    const recStack = new Set<string>();
    let hasCycle = false;
    const dfsCycle = (nid: string) => {
      visited.add(nid); recStack.add(nid);
      for (const nb of (adj[nid] || [])) {
        if (!visited.has(nb)) dfsCycle(nb);
        else if (recStack.has(nb)) hasCycle = true;
      }
      recStack.delete(nid);
    };
    nodes.forEach(n => { if (!visited.has(n.id)) dfsCycle(n.id); });
    if (hasCycle) {
      errors.push({ id: 'cycle', message: 'Cycle detected — architecture must be a feed-forward DAG (no loops)', severity: 'error' });
    }

    // Rule 5: warn if no hidden layers
    const hiddenNodes = nodes.filter(n => n.data.nodeType !== 'Input' && n.data.nodeType !== 'Output');
    if (hiddenNodes.length === 0 && nodes.length >= 2) {
      errors.push({ id: 'no-hidden', message: 'No hidden layers — consider adding Dense or Dropout layers', severity: 'warning' });
    }

    setValidationErrors(errors);
    const valid = errors.filter(e => e.severity === 'error').length === 0;
    setIsValid(valid);
    return valid;
  }, [nodes, edges]);

  // Auto-sanitize: deduplicate Input/Output nodes keeping the most-connected one
  const sanitizeGraph = useCallback(() => {
    let currentNodes = [...nodes];
    let currentEdges = [...edges];
    let changed = false;

    // Deduplicate Input nodes — keep the one with most connections
    const inputNodes = currentNodes.filter(n => n.data.nodeType === 'Input');
    if (inputNodes.length > 1) {
      const edgeCounts = inputNodes.map(n => ({
        node: n,
        count: currentEdges.filter(e => e.source === n.id || e.target === n.id).length
      }));
      edgeCounts.sort((a, b) => b.count - a.count);
      const toRemove = edgeCounts.slice(1).map(x => x.node.id);
      currentNodes = currentNodes.filter(n => !toRemove.includes(n.id));
      currentEdges = currentEdges.filter(e => !toRemove.includes(e.source) && !toRemove.includes(e.target));
      changed = true;
    }

    // Deduplicate Output nodes — keep the one with most connections
    const outputNodes = currentNodes.filter(n => n.data.nodeType === 'Output');
    if (outputNodes.length > 1) {
      const edgeCounts = outputNodes.map(n => ({
        node: n,
        count: currentEdges.filter(e => e.source === n.id || e.target === n.id).length
      }));
      edgeCounts.sort((a, b) => b.count - a.count);
      const toRemove = edgeCounts.slice(1).map(x => x.node.id);
      currentNodes = currentNodes.filter(n => !toRemove.includes(n.id));
      currentEdges = currentEdges.filter(e => !toRemove.includes(e.source) && !toRemove.includes(e.target));
      changed = true;
    }

    // Auto-wire isolated nodes into the chain (between last connected and Output)
    if (currentNodes.length > 1) {
      const connectedIds = new Set<string>();
      currentEdges.forEach(e => { connectedIds.add(e.source); connectedIds.add(e.target); });
      const orphans = currentNodes.filter(n => !connectedIds.has(n.id));

      if (orphans.length > 0) {
        // Find the node just before Output (the current last source before Output)
        const outputNode = currentNodes.find(n => n.data.nodeType === 'Output');
        if (outputNode) {
          const edgeToOutput = currentEdges.find(e => e.target === outputNode.id);
          orphans.forEach((orphan, i) => {
            // Insert orphan before Output: prev → orphan → Output
            const prevId = i === 0 ? edgeToOutput?.source : orphans[i - 1].id;
            if (prevId) {
              // Remove the old edge to Output
              if (i === 0 && edgeToOutput) {
                currentEdges = currentEdges.filter(e => e.id !== edgeToOutput.id);
              }
              currentEdges.push({ id: `e-${prevId}-${orphan.id}`, source: prevId, target: orphan.id, animated: true } as Edge);
              currentEdges.push({ id: `e-${orphan.id}-${outputNode.id}`, source: orphan.id, target: outputNode.id, animated: true } as Edge);
              changed = true;
            }
          });
        }
      }
    }

    if (changed) {
      setNodes(currentNodes);
      setEdges(currentEdges);
    }
    return { nodes: currentNodes, edges: currentEdges };
  }, [nodes, edges, setNodes, setEdges]);

  const handleSync = useCallback(async () => {
    // Step 1: auto-sanitize (remove duplicate Input/Output, wire orphans)
    const { nodes: cleanNodes, edges: cleanEdges } = sanitizeGraph();

    // Step 2: validate the cleaned graph inline (don't update state to avoid
    // render cycle — compute errors directly)
    const inputCount  = cleanNodes.filter(n => n.data.nodeType === 'Input').length;
    const outputCount = cleanNodes.filter(n => n.data.nodeType === 'Output').length;
    const connectedIds = new Set<string>();
    cleanEdges.forEach(e => { connectedIds.add(e.source); connectedIds.add(e.target); });
    const islandCount = cleanNodes.filter(n => cleanNodes.length > 1 && !connectedIds.has(n.id)).length;

    const syncErrors: ValidationError[] = [];
    if (inputCount === 0)  syncErrors.push({ id: 'no-input',  message: 'Missing Input node — add one from the Layer Palette', severity: 'error' });
    if (outputCount === 0) syncErrors.push({ id: 'no-output', message: 'Missing Output node — add one from the Layer Palette', severity: 'error' });
    if (islandCount > 0)   syncErrors.push({ id: 'islands',   message: `${islandCount} unconnected node(s) detected — every node must be wired`, severity: 'error' });

    if (syncErrors.length > 0) {
      setValidationErrors(syncErrors);
      setIsValid(false);
      setSyncState('error');
      setTimeout(() => setSyncState('idle'), 2500);
      // Scroll error panel into view
      setTimeout(() => {
        document.getElementById('validation-panel')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 50);
      return;
    }

    // Step 3: POST to /sync-graph with cleaned data
    setSyncState('syncing');
    const params = new URLSearchParams(window.location.search);
    const session_id = params.get('session_id') || 'default';

    try {
      const res = await fetch('/sync-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id,
          nodes: cleanNodes.map(n => ({
            id: n.id,
            type: n.type,
            position: n.position,
            width: n.width || 220,
            height: n.height || 80,
            data: {
              label:    n.data.label,
              nodeType: n.data.nodeType,
              params:   n.data.params || {},
            },
          })),
          edges: cleanEdges.map(e => ({
            id:       e.id,
            source:   e.source,
            target:   e.target,
            animated: true,
          })),
        }),
      });

      const data = await res.json();

      if (data.ok) {
        setSyncState('success');
        setSyncCount({ nodes: data.nodes, edges: data.edges });
        setValidationErrors([]);
        setIsValid(true);
        window.parent.postMessage({
          type: 'GRAPH_CONFIRMED',
          graph: { nodes: cleanNodes, edges: cleanEdges },
        }, '*');
        setTimeout(() => setSyncState('idle'), 3000);
      } else {
        setSyncState('error');
        setTimeout(() => setSyncState('idle'), 2000);
      }
    } catch (err) {
      console.error('[OmniML] Sync failed:', err);
      setSyncState('error');
      setTimeout(() => setSyncState('idle'), 2000);
    }
  }, [nodes, edges, sanitizeGraph]);

  return (
    <div style={{ display: 'flex', flexDirection: 'row', width: '100vw', height: '100vh', overflow: 'hidden', background: '#0a0a0f' }} className="font-['JetBrains_Mono'] text-[#e6edf3]">
      <Sidebar />
      <div style={{ flex: 1, height: '100vh', minWidth: 0, position: 'relative', background: '#0a0a0f' }} ref={reactFlowWrapper}>
        {/* Toolbar */}
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '52px', zIndex: 10, display: 'flex', alignItems: 'center', padding: '0 16px', gap: '12px', background: 'rgba(13,17,23,0.85)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)', borderBottom: '1px solid #21262d' }}>
          {/* Left Group */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <h1 className="text-[#6366f1] font-bold tracking-wide flex items-center gap-2 whitespace-nowrap">
              <span>🤖</span> OmniML Architect
            </h1>
            <div className="flex gap-2">
              <span className="bg-[#6366f1]/20 text-[#6366f1] px-3 py-1 rounded-full text-xs font-semibold border border-[#6366f1]/30 whitespace-nowrap">
                {nodes.length} layers
              </span>
              <span className="bg-[#21262d] text-[#8b949e] px-3 py-1 rounded-full text-xs font-semibold border border-[#30363d] whitespace-nowrap">
                {edges.length} connections
              </span>
            </div>
          </div>
          {/* Right Group */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <button
              onClick={handleSuggestArchitecture}
              disabled={suggesting}
              style={{
                padding: '8px 16px',
                fontSize: '12px',
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 600,
                whiteSpace: 'nowrap',
                background: suggesting
                  ? 'rgba(99,102,241,0.15)'
                  : 'transparent',
                color: suggesting ? '#a5b4fc' : '#8b949e',
                border: '1px solid',
                borderColor: suggesting ? '#6366f1' : '#30363d',
                borderRadius: '8px',
                cursor: suggesting ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                flexShrink: 0,
              }}
            >
              {suggesting ? (
                <>
                  <span style={{
                    display: 'inline-block',
                    width: 10, height: 10,
                    border: '2px solid #6366f1',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 0.7s linear infinite',
                  }}/>
                  Thinking...
                </>
              ) : (
                '✦ Suggest Architecture'
              )}
            </button>
            <button
              onClick={handleValidate}
              style={{
                padding: '8px 16px',
                fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 600,
                whiteSpace: 'nowrap',
                background: 'transparent',
                color: isValid === true
                  ? '#10b981'
                  : isValid === false
                    ? '#ef4444'
                    : '#8b949e',
                border: '1px solid',
                borderColor: isValid === true
                  ? '#10b981'
                  : isValid === false
                    ? '#ef4444'
                    : '#30363d',
                borderRadius: 8,
                cursor: 'pointer',
                flexShrink: 0,
                transition: 'all 0.2s ease',
              }}
            >
              {isValid === true
                ? '✓ Valid'
                : isValid === false
                  ? `✗ ${validationErrors.filter(e => e.severity === 'error').length} error${validationErrors.filter(e => e.severity === 'error').length > 1 ? 's' : ''}`
                  : 'Validate Graph'}
            </button>
            <button
              onClick={handleSync}
              disabled={syncState === 'syncing'}
              style={{
                padding: '8px 20px',
                fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 700,
                whiteSpace: 'nowrap',
                letterSpacing: '0.05em',
                background: syncState === 'success'
                  ? '#059669'
                  : syncState === 'error'
                    ? '#dc2626'
                    : '#6366f1',
                color: '#ffffff',
                border: 'none',
                borderRadius: 8,
                cursor: syncState === 'syncing' ? 'not-allowed' : 'pointer',
                flexShrink: 0,
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                animation: syncState === 'error' ? 'shake 0.3s ease' : 'none',
              }}
            >
              {syncState === 'syncing' && (
                <span style={{
                  width: 10, height: 10,
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderTopColor: '#fff',
                  borderRadius: '50%',
                  display: 'inline-block',
                  animation: 'spin 0.6s linear infinite',
                  flexShrink: 0,
                }}/>
              )}
              {syncState === 'idle'    && 'SYNC ARCHITECTURE'}
              {syncState === 'syncing' && 'Syncing...'}
              {syncState === 'success' && `✓ Synced  ${syncCount.nodes} layers · ${syncCount.edges} edges`}
              {syncState === 'error'   && '✗ Fix errors first'}
            </button>
          </div>
        </div>

        {/* Rationale Banner */}
        {rationale && (
          <div style={{
            position: 'absolute',
            top: 52,
            left: 0,
            right: 0,
            zIndex: 9,
            padding: '8px 20px',
            background: 'rgba(99,102,241,0.08)',
            borderBottom: '1px solid rgba(99,102,241,0.2)',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '11px',
            color: '#a5b4fc',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <span style={{ color: '#6366f1', flexShrink: 0 }}>✦</span>
            <span>{rationale}</span>
            <button
              onClick={() => setRationale('')}
              style={{
                marginLeft: 'auto',
                background: 'none',
                border: 'none',
                color: '#6366f1',
                cursor: 'pointer',
                fontSize: '14px',
                flexShrink: 0,
                padding: 0,
              }}
            >✕</button>
          </div>
        )}

        {/* Validation Errors Panel */}
        {validationErrors.length > 0 && (
          <div
            id="validation-panel"
            style={{
              position: 'absolute',
              top: rationale ? 88 : 52,
              left: 0,
              right: 0,
              zIndex: 10,
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              padding: '8px 16px',
              background: 'rgba(13,17,23,0.97)',
              borderBottom: '1px solid #21262d',
              backdropFilter: 'blur(8px)',
            }}>
            {validationErrors.map(err => (
              <div
                key={err.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 11,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: err.severity === 'error' ? '#fca5a5' : '#fde68a',
                  padding: '4px 0',
                }}
              >
                <span style={{
                  color: err.severity === 'error' ? '#ef4444' : '#f59e0b',
                  flexShrink: 0,
                }}>
                  {err.severity === 'error' ? '✗' : '⚠'}
                </span>
                <span style={{ flex: 1 }}>{err.message}</span>
              </div>
            ))}
            <button
              onClick={() => {
                setValidationErrors([]);
                setIsValid(null);
              }}
              style={{
                alignSelf: 'flex-end',
                background: 'none',
                border: 'none',
                color: '#6366f1',
                cursor: 'pointer',
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
                padding: '2px 0',
              }}
            >
              dismiss
            </button>
          </div>
        )}

        {/* React Flow Canvas */}
        <div 
          ref={reactFlowWrapper}
          onDrop={onDrop}
          onDragOver={onDragOver}
          style={{ paddingTop: '52px', height: '100%', width: '100%' }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodesDelete={onNodesDelete}
            nodeTypes={nodeTypes}
            connectionLineStyle={{ stroke: '#6366f1', strokeWidth: 2 }}
            connectionLineType={ConnectionLineType.SmoothStep}
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: '#6366f1', strokeWidth: 2 },
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: '#6366f1',
              },
            }}
            snapToGrid={true}
            snapGrid={[20, 20]}
            fitView
            style={{ width: '100%', height: '100%' }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1a1a2e" />
            <Controls style={{ backgroundColor: '#0d1117', fill: '#e6edf3', border: '1px solid #30363d' }} />
            <MiniMap 
              style={{ backgroundColor: '#0d1117', border: '1px solid #30363d' }}
              nodeColor={(n: any) => {
                const colors: Record<string,string> = {
                  Input:'#10b981', Dense:'#6366f1', LSTM:'#8b5cf6',
                  Conv1D:'#ec4899', Dropout:'#f59e0b',
                  BatchNorm1d:'#06b6d4', Output:'#10b981',
                };
                return colors[n.data?.nodeType] || '#30363d';
              }}
              maskColor="rgba(10, 10, 15, 0.7)"
            />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <ReactFlowProvider>
      <AppFlow />
    </ReactFlowProvider>
  );
}
