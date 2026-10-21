// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Checks before a pipeline runs: the domain and warm-up of every node,
//! and what is likely a mistake, worded so the author can fix it.

use super::{
    BinaryOp,
    Diagnostic,
    Domain,
    Input,
    Location,
    Requirement,
    builder::Binding,
    display::Printer,
    graph::{
        Graph,
        Node,
        NodeId,
    },
};

/// What the compiler knows about every node before any data is seen.
#[derive(Debug, Clone)]
pub(super) struct Analysis {
    domains: Vec<Domain>,
    warm_ups: Vec<usize>,
    diagnostics: Vec<Diagnostic>,
}

impl Analysis {
    /// Derive domains and warm-ups in one pass over the nodes (children
    /// come first), then check the bindings. Diagnostics point at the
    /// node's location in the text form, if any.
    pub(super) fn run(
        graph: &Graph,
        inputs: &[Input],
        bindings: &[Binding],
        locations: &[Option<Location>],
    ) -> Self {
        assert_eq!(locations.len(), graph.len(), "a location per node");
        let name_of = |id: NodeId| {
            bindings
                .iter()
                .find(|b| b.is_named() && b.node() == id)
                .map(Binding::name)
        };
        let printer = Printer::new(graph, inputs, name_of);
        let mut analysis = Self {
            domains: Vec::with_capacity(graph.len()),
            warm_ups: Vec::with_capacity(graph.len()),
            diagnostics: Vec::new(),
        };
        for (id, node) in graph.iter() {
            let (domain, warm_up, diagnostic) = analysis.node(&printer, id, node, inputs);
            analysis.domains.push(domain);
            analysis.warm_ups.push(warm_up);
            let at = locations[id.index()];
            analysis.diagnostics.extend(diagnostic.map(|d| d.or_at(at)));
        }
        assert_eq!(analysis.domains.len(), graph.len(), "a domain per node");
        let findings = analysis.check_bindings(graph, bindings);
        analysis.diagnostics.extend(findings);
        analysis
    }

    /// The values node `id` can take.
    pub(super) fn domain(&self, id: NodeId) -> Domain {
        self.domains[id.index()]
    }

    /// An estimate of the updates node `id` needs before it is ready.
    pub(super) fn warm_up(&self, id: NodeId) -> usize {
        self.warm_ups[id.index()]
    }

    /// Every problem found, in node order, then the bindings'.
    pub(super) fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// The domain and warm-up of one node, and a problem with it.
    fn node<'a>(
        &self,
        printer: &Printer<'a, impl Fn(NodeId) -> Option<&'a str>>,
        id: NodeId,
        node: &Node,
        inputs: &[Input],
    ) -> (Domain, usize, Option<Diagnostic>) {
        let d = |n: &NodeId| self.domains[n.index()];
        let w = |n: &NodeId| self.warm_ups[n.index()];
        match node {
            Node::Input(i) => (inputs[*i].domain(), 0, None),
            Node::Const(c) => (Domain::point(*c), 0, None),
            Node::Neg(a) => (d(a).neg(), w(a), None),
            Node::Binary(op, a, b) => {
                let warm_up = w(a).max(w(b));
                match binary_domain(*op, a == b, d(a), d(b)) {
                    Some(domain) => (domain, warm_up, None),
                    None => {
                        let message = format!(
                            "`{}` divides by zero, so it is never ready",
                            printer.expr(id)
                        );
                        (Domain::REAL, warm_up, Some(Diagnostic::error(message)))
                    }
                }
            }
            Node::Call(function, a, params) => {
                let warm_up = w(a) + function.warm_up(params);
                let (arg, requires) = (d(a), function.requires());
                let Some(domain) = function.output_domain(arg, params) else {
                    let message = format!(
                        "`{}` is never ready: `{}` needs {} values, but `{}` lies in {arg}",
                        printer.expr(id),
                        function.name(),
                        requirement_words(requires),
                        printer.name_or_expr(*a),
                    );
                    return (Domain::REAL, warm_up, Some(Diagnostic::error(message)));
                };
                let partly = arg.intersect(requires.domain()) != Some(arg);
                // The entropy of signs counts values of at least zero.
                let one_sign = requires == Requirement::BothSigns
                    && (arg.lo() >= 0.0 || arg.intersect(Domain::non_negative()).is_none());
                let warning = if partly {
                    let hint = match printer.input_name(*a) {
                        Some(name) => format!(
                            "; declare the domain of input `{name}` if it is always {}",
                            requirement_words(requires)
                        ),
                        None => String::new(),
                    };
                    Some(format!(
                        "`{}` is not ready while `{}` is not {} (it lies in {arg}){hint}",
                        printer.expr(id),
                        printer.name_or_expr(*a),
                        requirement_words(requires),
                    ))
                } else if one_sign {
                    Some(format!(
                        "`{}` is constant: `{}` never changes sign (it lies in {arg})",
                        printer.expr(id),
                        printer.name_or_expr(*a),
                    ))
                } else {
                    None
                };
                (domain, warm_up, warning.map(Diagnostic::warning))
            }
        }
    }

    /// Constant and duplicate outputs, and helpers no output uses.
    fn check_bindings(&self, graph: &Graph, bindings: &[Binding]) -> Vec<Diagnostic> {
        let outputs = Vec::from_iter(bindings.iter().filter(|b| b.is_output()));
        let reachable = graph.reachable(outputs.iter().map(|b| b.node()));
        let mut found = Vec::new();
        for (i, binding) in bindings.iter().enumerate() {
            let name = binding.name();
            let at = binding.location();
            if !binding.is_output() {
                if !reachable[binding.node().index()] {
                    found.push(
                        Diagnostic::warning(format!("`{name}` is not used by any output"))
                            .or_at(at),
                    );
                }
                continue;
            }
            if let Some(c) = self.domain(binding.node()).as_point() {
                found.push(
                    Diagnostic::error(format!("output `{name}` is constant: it is always {c}"))
                        .or_at(at),
                );
            }
            let earlier = bindings[..i]
                .iter()
                .find(|b| b.is_output() && b.node() == binding.node());
            if let Some(earlier) = earlier {
                let message = format!(
                    "output `{name}` computes the same as output `{}`",
                    earlier.name()
                );
                found.push(Diagnostic::warning(message).or_at(at));
            }
        }
        found
    }
}

/// The domain of `a op b`, `None` if it is never finite. `same` says
/// that both operands are one node, so `x - x` is exactly zero.
fn binary_domain(op: BinaryOp, same: bool, lhs: Domain, rhs: Domain) -> Option<Domain> {
    if same {
        match op {
            BinaryOp::Sub => return Some(Domain::point(0.0)),
            // Where `a` is zero the quotient is not ready, elsewhere 1.
            BinaryOp::Div if lhs.as_point() != Some(0.0) => return Some(Domain::point(1.0)),
            BinaryOp::Mul => return Some(lhs.abs().map_increasing(|v| v * v)),
            BinaryOp::Add | BinaryOp::Div => {}
        }
    }
    op.output_domain(lhs, rhs)
}

fn requirement_words(requires: Requirement) -> &'static str {
    match requires {
        Requirement::Positive => "positive",
        Requirement::NonNegative => "non-negative",
        Requirement::Any | Requirement::BothSigns => "finite",
    }
}
