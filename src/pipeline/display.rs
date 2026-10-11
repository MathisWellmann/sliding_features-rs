//! The text of a node: canonical, so equal texts mean equal computations.

use std::fmt::Write as _;

use super::{
    Input,
    graph::{
        Graph,
        Node,
        NodeId,
    },
};

/// Writes the expression of nodes, referring to named nodes by name.
pub(super) struct Printer<'a, F> {
    graph: &'a Graph,
    inputs: &'a [Input],
    /// The name to print instead of a node's expression, if any.
    name_of: F,
}

impl<'a, F> Printer<'a, F>
where
    F: Fn(NodeId) -> Option<&'a str>,
{
    pub(super) fn new(graph: &'a Graph, inputs: &'a [Input], name_of: F) -> Self {
        Self {
            graph,
            inputs,
            name_of,
        }
    }

    /// The expression of `id`, with its children by name where they have
    /// one.
    pub(super) fn expr(&self, id: NodeId) -> String {
        let mut out = String::new();
        self.write(&mut out, id);
        out
    }

    /// The name of `id` if it has one, else its expression.
    pub(super) fn name_or_expr(&self, id: NodeId) -> String {
        let mut out = String::new();
        self.write_child(&mut out, id);
        out
    }

    /// The name of `id` if it is an input.
    pub(super) fn input_name(&self, id: NodeId) -> Option<&'a str> {
        match self.graph.get(id) {
            Node::Input(i) => Some(self.inputs[*i].name()),
            _ => None,
        }
    }

    fn write_child(&self, out: &mut String, id: NodeId) {
        // An input keeps its own name even if a binding aliases it.
        if matches!(self.graph.get(id), Node::Input(_)) {
            return self.write(out, id);
        }
        match (self.name_of)(id) {
            Some(name) => out.push_str(name),
            None => self.write(out, id),
        }
    }

    fn write(&self, out: &mut String, id: NodeId) {
        match self.graph.get(id) {
            Node::Input(i) => out.push_str(self.inputs[*i].name()),
            Node::Const(c) => {
                let _ = write!(out, "{c}");
            }
            Node::Neg(a) => {
                out.push('-');
                self.write_child(out, *a);
            }
            Node::Binary(op, a, b) => {
                out.push('(');
                self.write_child(out, *a);
                let _ = write!(out, " {} ", op.symbol());
                self.write_child(out, *b);
                out.push(')');
            }
            Node::Call(function, a, params) => {
                out.push_str(function.name());
                out.push('(');
                self.write_child(out, *a);
                for p in params {
                    let _ = write!(out, ", {p}");
                }
                out.push(')');
            }
        }
    }
}
