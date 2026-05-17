// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS atomic coordination as a directed graph.
//!
//! Represents the relationships between NUCLEUS atomics (Tower, Node,
//! Nest) and the substrates they run on as a directed graph. This is
//! the local evolution that biomeOS will absorb for ecosystem-wide
//! coordination.
//!
//! # Model
//!
//! - **Nodes** are `AtomicType` instances (Tower, Node, Nest) bound
//!   to `SubstrateKind` (GPU, NPU, CPU).
//! - **Edges** are `ChannelKind` connections representing data flow
//!   between atomics.
//! - **Queries** support reachability ("which substrates can reach this
//!   atomic?") and path finding ("optimal path from GPU to NPU?").

use crate::nucleus::AtomicType;
use crate::pipeline::ChannelKind;
use crate::substrate::SubstrateKind;
use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the biome coordination graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: u32,
    /// NUCLEUS atomic type.
    pub atomic: AtomicType,
    /// Substrate hosting this atomic.
    pub substrate: SubstrateKind,
    /// Human-readable label.
    pub label: String,
}

/// A directed edge in the biome coordination graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID.
    pub from: u32,
    /// Destination node ID.
    pub to: u32,
    /// Channel type for data transfer.
    pub channel: ChannelKind,
}

/// Directed graph representing NUCLEUS atomic coordination.
///
/// Nodes are atomic instances; edges are data channels. The graph
/// supports queries about reachability and substrate topology.
#[derive(Debug)]
pub struct BiomeGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    next_id: u32,
}

impl BiomeGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(
        &mut self,
        atomic: AtomicType,
        substrate: SubstrateKind,
        label: impl Into<String>,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(GraphNode {
            id,
            atomic,
            substrate,
            label: label.into(),
        });
        id
    }

    /// Add a directed edge between two nodes.
    pub fn add_edge(&mut self, from: u32, to: u32, channel: ChannelKind) {
        self.edges.push(GraphEdge { from, to, channel });
    }

    /// Get all nodes.
    #[must_use]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Get all edges.
    #[must_use]
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Find a node by ID.
    #[must_use]
    pub fn node(&self, id: u32) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Find all nodes of a given atomic type.
    #[must_use]
    pub fn nodes_by_atomic(&self, atomic: AtomicType) -> Vec<&GraphNode> {
        self.nodes.iter().filter(|n| n.atomic == atomic).collect()
    }

    /// Find all nodes on a given substrate kind.
    #[must_use]
    pub fn nodes_by_substrate(&self, kind: SubstrateKind) -> Vec<&GraphNode> {
        self.nodes.iter().filter(|n| n.substrate == kind).collect()
    }

    /// Which substrates can reach a given node (via BFS on reverse edges)?
    #[must_use]
    pub fn reachable_from(&self, target_id: u32) -> HashSet<SubstrateKind> {
        let reverse: HashMap<u32, Vec<u32>> = {
            let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
            for edge in &self.edges {
                map.entry(edge.to).or_default().push(edge.from);
            }
            map
        };

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(target_id);
        visited.insert(target_id);

        while let Some(current) = queue.pop_front() {
            if let Some(predecessors) = reverse.get(&current) {
                for &pred in predecessors {
                    if visited.insert(pred) {
                        queue.push_back(pred);
                    }
                }
            }
        }

        visited
            .iter()
            .filter_map(|&id| self.node(id).map(|n| n.substrate))
            .collect()
    }

    /// Find shortest path from `from` to `to` by node ID.
    ///
    /// Returns the ordered list of node IDs, or `None` if unreachable.
    #[must_use]
    pub fn shortest_path(&self, from: u32, to: u32) -> Option<Vec<u32>> {
        let adjacency: HashMap<u32, Vec<u32>> = {
            let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
            for edge in &self.edges {
                map.entry(edge.from).or_default().push(edge.to);
            }
            map
        };

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<u32, u32> = HashMap::new();

        queue.push_back(from);
        visited.insert(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                let mut path = vec![to];
                let mut cursor = to;
                while cursor != from {
                    cursor = *parent.get(&cursor)?;
                    path.push(cursor);
                }
                path.reverse();
                return Some(path);
            }
            if let Some(neighbors) = adjacency.get(&current) {
                for &next in neighbors {
                    if visited.insert(next) {
                        parent.insert(next, current);
                        queue.push_back(next);
                    }
                }
            }
        }

        None
    }

    /// Count how many PCIe direct hops are in a path.
    #[must_use]
    pub fn pcie_direct_hops(&self, path: &[u32]) -> usize {
        path.windows(2)
            .filter(|w| {
                self.edges
                    .iter()
                    .any(|e| e.from == w[0] && e.to == w[1] && e.channel == ChannelKind::PcieDirect)
            })
            .count()
    }
}

impl Default for BiomeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a standard NUCLEUS coordination graph.
///
/// Creates the canonical Tower → Node → Nest topology with
/// appropriate substrate assignments and channel types.
#[must_use]
pub fn standard_nucleus_graph() -> BiomeGraph {
    let mut g = BiomeGraph::new();

    let tower = g.add_node(AtomicType::Tower, SubstrateKind::Cpu, "Tower (trust)");
    let node = g.add_node(AtomicType::Node, SubstrateKind::Gpu, "Node (compute)");
    let nest = g.add_node(AtomicType::Nest, SubstrateKind::Cpu, "Nest (provenance)");

    g.add_edge(tower, node, ChannelKind::SharedMemory);
    g.add_edge(node, nest, ChannelKind::Pcie);
    g.add_edge(tower, nest, ChannelKind::SharedMemory);

    g
}

/// Build a NUCLEUS graph with PCIe direct GPU→NPU bypass.
///
/// Extends the standard graph with an NPU inference node that
/// receives GPU output directly via PCIe peer-to-peer.
#[must_use]
pub fn pcie_direct_nucleus_graph() -> BiomeGraph {
    let mut g = BiomeGraph::new();

    let tower = g.add_node(AtomicType::Tower, SubstrateKind::Cpu, "Tower (trust)");
    let node_gpu = g.add_node(AtomicType::Node, SubstrateKind::Gpu, "Node (GPU compute)");
    let node_npu = g.add_node(AtomicType::Node, SubstrateKind::Npu, "Node (NPU inference)");
    let nest = g.add_node(AtomicType::Nest, SubstrateKind::Cpu, "Nest (provenance)");

    g.add_edge(tower, node_gpu, ChannelKind::SharedMemory);
    g.add_edge(node_gpu, node_npu, ChannelKind::PcieDirect);
    g.add_edge(node_npu, nest, ChannelKind::Pcie);
    g.add_edge(tower, nest, ChannelKind::SharedMemory);

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_graph_structure() {
        let g = standard_nucleus_graph();
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.edges().len(), 3);
    }

    #[test]
    fn node_lookup_by_atomic() {
        let g = standard_nucleus_graph();
        let towers = g.nodes_by_atomic(AtomicType::Tower);
        assert_eq!(towers.len(), 1);
        assert_eq!(towers[0].substrate, SubstrateKind::Cpu);
    }

    #[test]
    fn node_lookup_by_substrate() {
        let g = standard_nucleus_graph();
        let cpus = g.nodes_by_substrate(SubstrateKind::Cpu);
        assert_eq!(cpus.len(), 2);
    }

    #[test]
    fn shortest_path_tower_to_nest() {
        let g = standard_nucleus_graph();
        let tower_id = g.nodes_by_atomic(AtomicType::Tower)[0].id;
        let nest_id = g.nodes_by_atomic(AtomicType::Nest)[0].id;
        let path = g.shortest_path(tower_id, nest_id);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], tower_id);
        assert_eq!(path[1], nest_id);
    }

    #[test]
    fn reachable_substrates() {
        let g = standard_nucleus_graph();
        let nest_id = g.nodes_by_atomic(AtomicType::Nest)[0].id;
        let reachable = g.reachable_from(nest_id);
        assert!(reachable.contains(&SubstrateKind::Cpu));
        assert!(reachable.contains(&SubstrateKind::Gpu));
    }

    #[test]
    fn pcie_direct_graph() {
        let g = pcie_direct_nucleus_graph();
        assert_eq!(g.nodes().len(), 4);
        let pcie_direct_count = g
            .edges()
            .iter()
            .filter(|e| e.channel == ChannelKind::PcieDirect)
            .count();
        assert_eq!(pcie_direct_count, 1);
    }

    #[test]
    fn pcie_direct_hop_count() {
        let g = pcie_direct_nucleus_graph();
        let gpu_id = g.nodes_by_substrate(SubstrateKind::Gpu)[0].id;
        let npu_id = g.nodes_by_substrate(SubstrateKind::Npu)[0].id;
        let nest_id = g.nodes_by_atomic(AtomicType::Nest)[0].id;
        let path = g.shortest_path(gpu_id, nest_id).unwrap();
        assert_eq!(g.pcie_direct_hops(&path), 1);
        assert_eq!(path, vec![gpu_id, npu_id, nest_id]);
    }
}
