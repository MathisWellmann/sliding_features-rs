// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The runtime form of a pipeline: a flat list of operations evaluated in
//! one pass, each writing one slot of a value buffer.

use super::{
    BinaryOp,
    Requirement,
    function::Params,
    graph::{
        Graph,
        Node,
        NodeId,
    },
    state::{
        Kind,
        Pure,
        State,
    },
};

/// One operation. Operands index the value buffer, always below the
/// operation's own slot.
#[derive(Debug, Clone, Copy)]
enum Op {
    Input(u32),
    Const(f64),
    Pure(Pure, u32),
    Binary(BinaryOp, u32, u32),
    Stateful {
        state: u32,
        arg: u32,
        requires: Requirement,
    },
}

/// The operations of the nodes the outputs read, their states and their
/// last values; NaN marks a value that is not ready.
#[derive(Debug, Clone)]
pub(super) struct Program {
    ops: Vec<Op>,
    /// Kept apart from `ops`, so the loop over the small operations stays
    /// dense and a large view does not widen every slot.
    states: Vec<State>,
    values: Vec<f64>,
    outputs: Vec<u32>,
}

impl Program {
    /// Lower the nodes that `outputs` read into operations, with fresh
    /// states.
    pub(super) fn lower(graph: &Graph, outputs: &[NodeId]) -> Self {
        assert!(!outputs.is_empty(), "a pipeline has an output");
        let reachable = graph.reachable(outputs.iter().copied());
        let mut slot_of = vec![u32::MAX; graph.len()];
        let mut ops = Vec::new();
        let mut states = Vec::new();
        for (id, node) in graph.iter() {
            if !reachable[id.index()] {
                continue;
            }
            let slot = |child: &NodeId| {
                let s = slot_of[child.index()];
                assert_ne!(s, u32::MAX, "children are lowered first");
                s
            };
            let op = match node {
                Node::Input(i) => Op::Input(to_u32(*i)),
                Node::Const(c) => Op::Const(*c),
                Node::Neg(a) => Op::Pure(Pure::Neg, slot(a)),
                Node::Binary(op, a, b) => Op::Binary(*op, slot(a), slot(b)),
                Node::Call(function, a, params) => {
                    match function.instantiate(Params::new(*function, params)) {
                        Kind::Pure(pure) => Op::Pure(pure, slot(a)),
                        Kind::Stateful(state) => {
                            states.push(state);
                            Op::Stateful {
                                state: to_u32(states.len() - 1),
                                arg: slot(a),
                                requires: function.requires(),
                            }
                        }
                    }
                }
            };
            slot_of[id.index()] = to_u32(ops.len());
            ops.push(op);
        }
        let outputs = Vec::from_iter(outputs.iter().map(|o| slot_of[o.index()]));
        assert!(outputs.iter().all(|o| (*o as usize) < ops.len()));
        Self {
            values: vec![f64::NAN; ops.len()],
            ops,
            states,
            outputs,
        }
    }

    /// Feed one value per input series.
    #[inline]
    pub(super) fn update(&mut self, inputs: &[f64]) {
        run(&self.ops, &mut self.states, &mut self.values, inputs);
    }

    /// The `i`-th output, `None` while not ready.
    #[inline]
    pub(super) fn output(&self, i: usize) -> Option<f64> {
        let v = self.values[self.outputs[i] as usize];
        (!v.is_nan()).then_some(v)
    }

    /// The number of operations run per update.
    pub(super) fn len(&self) -> usize {
        self.ops.len()
    }
}

fn to_u32(i: usize) -> u32 {
    u32::try_from(i).expect("node counts fit u32")
}

/// Evaluate every operation once, in order. Free of `self`, so the slices
/// stay in registers.
#[inline]
fn run(ops: &[Op], states: &mut [State], values: &mut [f64], inputs: &[f64]) {
    assert_eq!(ops.len(), values.len(), "one value per operation");
    for (i, op) in ops.iter().enumerate() {
        let v = match *op {
            Op::Input(j) => inputs[j as usize],
            Op::Const(c) => c,
            Op::Pure(f, a) => f.eval(values[a as usize]),
            Op::Binary(op, a, b) => op.eval(values[a as usize], values[b as usize]),
            Op::Stateful {
                state,
                arg,
                requires,
            } => {
                let a = values[arg as usize];
                let state = &mut states[state as usize];
                // A view only sees admitted values and otherwise holds its
                // last output, like a chained view whose input is not ready.
                if admits(requires, a) {
                    state.update(a)
                } else {
                    state.last()
                }
            }
        };
        // Not ready instead of an infinity: one non-finite value would stay
        // in every window downstream for its whole length.
        values[i] = if v.is_finite() { v } else { f64::NAN };
    }
}

/// Whether a value meets the requirement; NaN (not ready) never does.
#[inline]
fn admits(requires: Requirement, v: f64) -> bool {
    match requires {
        Requirement::Any | Requirement::BothSigns => !v.is_nan(),
        Requirement::Positive => v > 0.0,
        Requirement::NonNegative => v >= 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::Function,
        *,
    };

    #[test]
    fn admits_rejects_not_ready_values() {
        for requires in [
            Requirement::Any,
            Requirement::BothSigns,
            Requirement::Positive,
            Requirement::NonNegative,
        ] {
            assert!(!admits(requires, f64::NAN));
        }
        assert!(admits(Requirement::NonNegative, 0.0));
        assert!(!admits(Requirement::Positive, 0.0));
        assert!(admits(Requirement::Any, -1.0));
    }

    #[test]
    fn only_reachable_nodes_are_lowered() {
        let mut g = Graph::default();
        let x = g.intern(Node::Input(0), 10).unwrap();
        let unused = g
            .intern(Node::Call(Function::Sma, x, [3.0].into()), 10)
            .unwrap();
        let ema = g
            .intern(Node::Call(Function::Ema, x, [3.0].into()), 10)
            .unwrap();
        let program = Program::lower(&g, &[ema]);
        assert_eq!(program.len(), 2);
        assert_eq!(program.states.len(), 1);
        assert_ne!(unused, ema);
    }
}
