// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A pipeline as a graph: nodes in creation order, so children come before
//! their parents, and structurally equal nodes shared.

use std::collections::HashMap;

use super::{
    BinaryOp,
    Function,
};

/// A node of a pipeline under construction, see
/// [`Builder`](super::Builder). Only valid with the builder that made it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
    pub(super) fn index(self) -> usize {
        usize::try_from(self.0).expect("u32 fits in usize on supported targets")
    }
}

/// One computation of a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Node {
    /// The input series with this index.
    Input(usize),
    /// A finite constant.
    Const(f64),
    /// `-a`
    Neg(NodeId),
    /// `a op b`
    Binary(BinaryOp, NodeId, NodeId),
    /// A function applied to a node, with checked parameters.
    Call(Function, NodeId, Box<[f64]>),
}

impl Node {
    /// The nodes this one reads, left to right.
    pub(super) fn children(&self) -> impl Iterator<Item = NodeId> + use<> {
        let (a, b) = match self {
            Node::Input(_) | Node::Const(_) => (None, None),
            Node::Neg(a) | Node::Call(_, a, _) => (Some(*a), None),
            Node::Binary(_, a, b) => (Some(*a), Some(*b)),
        };
        a.into_iter().chain(b)
    }

    fn key(&self) -> Key {
        match self {
            Node::Input(i) => Key::Input(*i),
            Node::Const(c) => Key::Const(c.to_bits()),
            Node::Neg(a) => Key::Neg(*a),
            Node::Binary(op, a, b) => Key::Binary(*op, *a, *b),
            Node::Call(function, a, params) => {
                Key::Call(*function, *a, params.iter().map(|p| p.to_bits()).collect())
            }
        }
    }
}

/// A [`Node`] with its numbers as bits, so it can be hashed: equal keys
/// compute the same values, so one node serves both.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Input(usize),
    Const(u64),
    Neg(NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    Call(Function, NodeId, Box<[u64]>),
}

/// The nodes of a pipeline, each computation once.
#[derive(Debug, Clone, Default)]
pub(super) struct Graph {
    nodes: Vec<Node>,
    index: HashMap<Key, NodeId>,
}

impl Graph {
    /// The id of `node`: the existing equal node, or `node` added. `None`
    /// if adding it would exceed `max_nodes`.
    pub(super) fn intern(&mut self, node: Node, max_nodes: usize) -> Option<NodeId> {
        for child in node.children() {
            assert!(child.index() < self.nodes.len(), "a node of this graph");
        }
        if let Node::Const(c) = node {
            assert!(c.is_finite(), "constants are finite");
        }
        let key = node.key();
        if let Some(id) = self.index.get(&key) {
            return Some(*id);
        }
        if self.nodes.len() >= max_nodes {
            return None;
        }
        let id = NodeId(u32::try_from(self.nodes.len()).ok()?);
        self.nodes.push(node);
        self.index.insert(key, id);
        assert_eq!(self.nodes.len(), self.index.len(), "every node is indexed");
        Some(id)
    }

    /// The node `id`.
    pub(super) fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    /// The number of nodes.
    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Every node with its id, children before parents.
    pub(super) fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (NodeId(u32::try_from(i).expect("ids fit u32")), node))
    }

    /// Which nodes the `roots` read, directly or not, roots included.
    pub(super) fn reachable(&self, roots: impl IntoIterator<Item = NodeId>) -> Vec<bool> {
        let mut reachable = vec![false; self.nodes.len()];
        for root in roots {
            reachable[root.index()] = true;
        }
        // Parents come after their children, so one backward pass suffices.
        for i in (0..self.nodes.len()).rev() {
            if reachable[i] {
                for child in self.nodes[i].children() {
                    assert!(child.index() < i, "children come first");
                    reachable[child.index()] = true;
                }
            }
        }
        reachable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_nodes_are_shared() {
        let mut g = Graph::default();
        let x = g.intern(Node::Input(0), 10).unwrap();
        let a = g.intern(Node::Call(Function::Sma, x, [3.0].into()), 10);
        let b = g.intern(Node::Call(Function::Sma, x, [3.0].into()), 10);
        let c = g.intern(Node::Call(Function::Sma, x, [4.0].into()), 10);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn the_node_limit_counts_distinct_nodes() {
        let mut g = Graph::default();
        let x = g.intern(Node::Input(0), 2).unwrap();
        let neg = g.intern(Node::Neg(x), 2).unwrap();
        assert_eq!(g.intern(Node::Neg(x), 2), Some(neg), "shared: no new node");
        assert_eq!(g.intern(Node::Neg(neg), 2), None);
    }

    #[test]
    fn reachability() {
        let mut g = Graph::default();
        let x = g.intern(Node::Input(0), 10).unwrap();
        let y = g.intern(Node::Input(1), 10).unwrap();
        let neg = g.intern(Node::Neg(x), 10).unwrap();
        let sum = g.intern(Node::Binary(BinaryOp::Add, neg, x), 10).unwrap();
        assert_eq!(g.reachable([sum]), vec![true, false, true, true]);
        assert_eq!(g.reachable([y]), vec![false, true, false, false]);
        assert_eq!(Vec::from_iter(g.get(sum).children()), vec![neg, x]);
    }

    #[test]
    #[should_panic(expected = "a node of this graph")]
    fn foreign_nodes_are_rejected() {
        let mut g = Graph::default();
        let _ = g.intern(Node::Neg(NodeId(5)), 10);
    }
}
