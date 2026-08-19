import { GitGraph, MousePointer2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { EvidenceItem } from "../types/query";
import { compactId, metadataText, titleCase } from "../features/format";

interface GraphTabProps {
  evidence: EvidenceItem[];
}

interface GraphNode {
  id: string;
  label: string;
  type: string;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  evidenceId: string;
  metadata: Record<string, unknown>;
}

export function GraphTab({ evidence }: GraphTabProps) {
  const graph = useMemo(() => buildGraph(evidence), [evidence]);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(graph.edges[0] || null);

  if (graph.nodes.length === 0) {
    return <p className="muted">No graph relationships or paths were returned for this query.</p>;
  }

  const positions = graph.nodes.map((node, index) => ({
    node,
    x: 80 + index * 170,
    y: index % 2 === 0 ? 80 : 150
  }));
  const positionById = new Map(positions.map((item) => [item.node.id, item]));

  return (
    <div className="graph-layout">
      <div className="graph-canvas" aria-label="Graph evidence visualization">
        <svg viewBox="0 0 760 240" role="img" aria-label="Current query graph">
          {graph.edges.map((edge) => {
            const source = positionById.get(edge.source);
            const target = positionById.get(edge.target);
            if (!source || !target) {
              return null;
            }
            return (
              <g key={edge.id}>
                <line x1={source.x + 48} y1={source.y} x2={target.x - 48} y2={target.y} className="graph-edge" />
                <g
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedEdge(edge)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedEdge(edge);
                    }
                  }}
                  aria-label={`Select ${edge.label}`}
                >
                  <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2 - 8} className="edge-label">
                    {titleCase(edge.label)}
                  </text>
                </g>
              </g>
            );
          })}
          {positions.map(({ node, x, y }) => (
            <g key={node.id}>
              <circle cx={x} cy={y} r="42" className={`graph-node node-${node.type}`} />
              <text x={x} y={y - 4} className="node-label">
                {truncate(node.label, 17)}
              </text>
              <text x={x} y={y + 14} className="node-type">
                {titleCase(node.type)}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <aside className="edge-panel">
        <p className="eyebrow">
          <MousePointer2 size={14} aria-hidden="true" />
          Selected edge
        </p>
        {selectedEdge ? (
          <>
            <h3>{titleCase(selectedEdge.label)}</h3>
            <dl className="detail-grid compact">
              <div>
                <dt>Evidence</dt>
                <dd>{compactId(selectedEdge.evidenceId)}</dd>
              </div>
              <div>
                <dt>Source</dt>
                <dd>{compactId(selectedEdge.source)}</dd>
              </div>
              <div>
                <dt>Target</dt>
                <dd>{compactId(selectedEdge.target)}</dd>
              </div>
            </dl>
            <pre>{metadataText(selectedEdge.metadata)}</pre>
          </>
        ) : (
          <p className="muted">Select an edge to inspect relationship metadata.</p>
        )}
      </aside>
      <p className="graph-note">
        <GitGraph size={15} aria-hidden="true" />
        Visualized from graph evidence in this response only.
      </p>
    </div>
  );
}

function buildGraph(evidence: EvidenceItem[]): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodes = new Map<string, GraphNode>();
  const edges = new Map<string, GraphEdge>();
  for (const item of evidence) {
    if (item.evidence_type !== "graph_relationship" && item.evidence_type !== "graph_path") {
      continue;
    }
    const metadataNodes = asArray<Record<string, unknown>>(item.metadata?.nodes);
    const metadataRelationships = asArray<Record<string, unknown>>(item.metadata?.relationships);
    for (const node of metadataNodes) {
      const id = String(node.entity_id || "");
      if (!id) {
        continue;
      }
      nodes.set(id, {
        id,
        label: String(node.canonical_name || propertyValue(node.properties, "title") || id),
        type: String(node.entity_type || "entity")
      });
    }
    for (const relationship of metadataRelationships) {
      const id = String(relationship.relationship_id || `${item.evidence_id}:${edges.size}`);
      const source = String(relationship.source_entity_id || "");
      const target = String(relationship.target_entity_id || "");
      if (!source || !target) {
        continue;
      }
      nodes.set(source, nodes.get(source) || { id: source, label: source, type: "entity" });
      nodes.set(target, nodes.get(target) || { id: target, label: target, type: "entity" });
      edges.set(id, {
        id,
        source,
        target,
        label: String(relationship.relationship_type || "related_to"),
        evidenceId: item.evidence_id,
        metadata: relationship
      });
    }
  }
  return { nodes: [...nodes.values()].slice(0, 8), edges: [...edges.values()].slice(0, 10) };
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function truncate(value: string, maxLength: number): string {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

function propertyValue(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return (value as Record<string, unknown>)[key];
}
