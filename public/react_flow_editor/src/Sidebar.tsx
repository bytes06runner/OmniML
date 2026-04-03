import React from 'react';

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

const NODE_TYPES = Object.keys(typeColors);

export default function Sidebar() {
  const onDragStart = (
    e: React.DragEvent, 
    nodeType: string
  ) => {
    e.dataTransfer.setData('application/reactflow', nodeType);
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside style={{
        width: 180,
        height: '100%',
        minHeight: '100vh',
        background: '#0d1117',
        borderRight: '1px solid #21262d',
        display: 'flex',
        flexDirection: 'column',
        padding: 16,
        flexShrink: 0,
        overflowY: 'auto'
    }}>
      <h3 style={{
        color: '#6366f1',
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        marginBottom: 16
      }}>
        Layer Palette
      </h3>
      
      <div style={{ flexGrow: 1, overflowY: 'auto' }}>
        {NODE_TYPES.map((type) => (
          <div
            key={type}
            draggable
            onDragStart={(e) => onDragStart(e, type)}
            style={{
              padding: '8px 12px',
              cursor: 'grab',
              fontSize: 12,
              color: typeColors[type],
              borderBottom: '1px solid #21262d',
              fontFamily: "'JetBrains Mono', monospace",
              userSelect: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              transition: 'background 0.15s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#161b22'}
            onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
          >
            <span style={{
              width: 8, height: 8,
              borderRadius: '50%',
              background: typeColors[type],
              flexShrink: 0,
              display: 'inline-block',
            }}/>
            {type}
          </div>
        ))}
      </div>

      <div style={{
        marginTop: 16,
        paddingTop: 16,
        borderTop: '1px solid #30363d',
        fontSize: 11,
        color: '#8b949e',
        lineHeight: 1.4,
        fontFamily: "'JetBrains Mono', monospace"
      }}>
        Drag layers onto canvas → connect handles → click Sync
      </div>
    </aside>
  );
}
