import { Handle, Position, type NodeProps } from '@xyflow/react';

export default function CustomLayerNode({ data }: NodeProps) {
  const typeColors: Record<string, string> = {
    Input:       '#10b981',
    Dense:       '#6366f1',
    LSTM:        '#8b5cf6',
    Conv1D:      '#ec4899',
    Dropout:     '#f59e0b',
    BatchNorm1d: '#06b6d4',
    ReLU:        '#84cc16',
    Sigmoid:     '#f97316',
    Output:      '#10b981',
  };
  const color = typeColors[data.nodeType as string] || '#6366f1';
  const isInput  = data.nodeType === 'Input';
  const isOutput = data.nodeType === 'Output';

  const params = data.params || {};
  const paramLine = Object.entries(params)
    .map(([k, v]) => `${k}: ${v}`)
    .join('  ·  ');

  return (
    <div style={{
      width: 220,
      minHeight: 80,
      background: '#0d1117',
      border: `1.5px solid ${color}`,
      borderRadius: 12,
      overflow: 'hidden',
      fontFamily: "'JetBrains Mono', monospace",
      position: 'relative',
      boxShadow: `0 0 16px ${color}22`,
    }}>

      {/* TOP HANDLE — hidden for Input nodes */}
      {!isInput && (
        <Handle
          type="target"
          position={Position.Top}
          style={{
            width: 14,
            height: 14,
            background: color,
            border: '2px solid #0d1117',
            borderRadius: '50%',
            top: -7,
            left: '50%',
            transform: 'translateX(-50%)',
            cursor: 'crosshair',
            zIndex: 10,
          }}
        />
      )}

      {/* COLOR BAR */}
      <div style={{
        height: 4,
        background: color,
        width: '100%',
      }}/>

      {/* NODE TYPE LABEL */}
      <div style={{
        padding: '8px 12px 2px',
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        color: color,
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>
        {(data.nodeType as string) || (data.label as string)}
      </div>

      {/* NODE NAME */}
      <div style={{
        padding: '0 12px 4px',
        fontSize: 13,
        fontWeight: 600,
        color: '#e6edf3',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>
        {data.label as string}
      </div>

      {/* PARAMS LINE */}
      {paramLine && (
        <div style={{
          padding: '0 12px 10px',
          fontSize: 11,
          color: '#8b949e',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          {paramLine}
        </div>
      )}

      {/* DELETE BUTTON — top right, hover only via CSS class */}
      <button
        className="node-delete-btn"
        onClick={(e) => {
          e.stopPropagation();
          (data.onDelete as any)?.(data.id);
        }}
        style={{
          position: 'absolute',
          top: 6,
          right: 6,
          width: 18,
          height: 18,
          background: 'rgba(239,68,68,0.15)',
          border: '1px solid rgba(239,68,68,0.4)',
          borderRadius: 4,
          color: '#f87171',
          fontSize: 11,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          lineHeight: 1,
          padding: 0,
          opacity: 0,
          transition: 'opacity 0.15s',
        }}
      >
        ✕
      </button>

      {/* BOTTOM HANDLE — hidden for Output nodes */}
      {!isOutput && (
        <Handle
          type="source"
          position={Position.Bottom}
          style={{
            width: 14,
            height: 14,
            background: color,
            border: '2px solid #0d1117',
            borderRadius: '50%',
            bottom: -7,
            left: '50%',
            transform: 'translateX(-50%)',
            cursor: 'crosshair',
            zIndex: 10,
          }}
        />
      )}
    </div>
  );
}
