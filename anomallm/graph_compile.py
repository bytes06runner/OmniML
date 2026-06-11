from __future__ import annotations

from typing import List, Tuple


def build_mlp_layers(graph_nodes: List[dict]) -> Tuple[str, str]:
    """Compile React Flow graph nodes into PyTorch nn.Module init and forward lines."""
    sorted_nodes = sorted(graph_nodes, key=lambda n: n.get("position", {}).get("y", 0))

    init_lines: List[str] = []
    forward_lines: List[str] = []
    layer_idx = 0
    prev_dim = "input_dim"

    for node in sorted_nodes:
        d = node.get("data", {})
        ntype = d.get("nodeType", "")
        params = d.get("params", {})

        if ntype == "Input":
            continue
        if ntype == "Dense":
            units = params.get("units", 128)
            act = params.get("activation", "relu")
            init_lines.append(f"        self.fc{layer_idx} = nn.Linear({prev_dim}, {units})")
            if act == "relu":
                init_lines.append(f"        self.act{layer_idx} = nn.ReLU()")
            elif act == "tanh":
                init_lines.append(f"        self.act{layer_idx} = nn.Tanh()")
            elif act == "selu":
                init_lines.append(f"        self.act{layer_idx} = nn.SELU()")
            else:
                init_lines.append(f"        self.act{layer_idx} = nn.ReLU()")
            forward_lines.append(f"        x = self.fc{layer_idx}(x)")
            forward_lines.append(f"        x = self.act{layer_idx}(x)")
            prev_dim = str(units)
            layer_idx += 1
        elif ntype == "BatchNorm1d":
            init_lines.append(f"        self.bn{layer_idx} = nn.BatchNorm1d({prev_dim})")
            forward_lines.append(f"        x = self.bn{layer_idx}(x)")
            layer_idx += 1
        elif ntype == "Dropout":
            rate = params.get("rate", 0.3)
            init_lines.append(f"        self.drop{layer_idx} = nn.Dropout({rate})")
            forward_lines.append(f"        x = self.drop{layer_idx}(x)")
            layer_idx += 1
        elif ntype == "Output":
            init_lines.append(f"        self.output_layer = nn.Linear({prev_dim}, num_classes)")
            forward_lines.append(f"        x = self.output_layer(x)")
            prev_dim = "num_classes"

    if not init_lines:
        init_lines = [
            "        self.fc0 = nn.Linear(input_dim, 128)",
            "        self.act0 = nn.ReLU()",
            "        self.drop0 = nn.Dropout(0.3)",
            "        self.fc1 = nn.Linear(128, 64)",
            "        self.act1 = nn.ReLU()",
            "        self.output_layer = nn.Linear(64, num_classes)",
        ]
        forward_lines = [
            "        x = self.fc0(x)",
            "        x = self.act0(x)",
            "        x = self.drop0(x)",
            "        x = self.fc1(x)",
            "        x = self.act1(x)",
            "        x = self.output_layer(x)",
        ]

    return "\n".join(init_lines), "\n".join(forward_lines)


def describe_architecture_layers(graph_nodes: List[dict]) -> str:
    """Human-readable layer string for HPT / logs."""
    parts = []
    for node in sorted(graph_nodes, key=lambda n: n.get("position", {}).get("y", 0)):
        t = node.get("data", {}).get("nodeType", "Dense")
        p = node.get("data", {}).get("params", {})
        if t == "Dense":
            parts.append(f"Dense({p.get('units', '?')},{p.get('activation', 'relu')})")
        elif t == "Dropout":
            parts.append(f"Dropout({p.get('rate', '?')})")
        elif t == "Output":
            parts.append(f"Output({p.get('units', '?')},{p.get('activation', 'sigmoid')})")
        elif t == "BatchNorm1d":
            parts.append("BatchNorm1d")
    return " → ".join(parts) if parts else "pytorch_mlp_fallback"
