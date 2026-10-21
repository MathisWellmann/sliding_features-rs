// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Assembling a pipeline node by node: the programmatic way to build one,
//! e.g. for a search, and the target of the text form.

use super::{
    BinaryOp,
    CompileError,
    Domain,
    Function,
    Location,
    Pipeline,
    Severity,
    analysis::Analysis,
    display::Printer,
    function::Params,
    graph::{
        Graph,
        Node,
        NodeId,
    },
    state::Kind,
};

/// Resource bounds of a pipeline, so one from an untrusted author (e.g. an
/// LLM) cannot make it arbitrarily slow or large. Plain data: any values
/// are valid, small ones just reject more pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The largest window parameter.
    pub max_window: usize,
    /// The most distinct nodes, inputs and constants included; a shared
    /// subexpression counts once.
    pub max_nodes: usize,
    /// The longest chain of nodes from an input or constant to any node,
    /// and the deepest nesting of parentheses and calls in the text form.
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_window: 100_000,
            max_nodes: 256,
            max_depth: 32,
        }
    }
}

impl Limits {
    /// No bounds besides the function signatures.
    pub fn unbounded() -> Self {
        Self {
            max_window: usize::MAX,
            max_nodes: usize::MAX,
            max_depth: usize::MAX,
        }
    }
}

/// An input series of a pipeline: its name and the values it takes. The
/// domain lets the compiler check functions with requirements, e.g. that
/// `ln(price)` is defined when prices are declared positive.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    name: String,
    domain: Domain,
}

impl Input {
    /// An input called `name` whose values lie in `domain`.
    pub fn new(name: impl Into<String>, domain: Domain) -> Self {
        Self {
            name: name.into(),
            domain,
        }
    }

    /// An input called `name` that takes any finite value.
    pub fn real(name: impl Into<String>) -> Self {
        Self::new(name, Domain::REAL)
    }

    /// The name expressions refer to the input by.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The values the input takes.
    pub fn domain(&self) -> Domain {
        self.domain
    }
}

/// What a binding names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BindingKind {
    /// A node later statements refer to by name (`let name = ...`).
    Helper,
    /// An output (`name = ...`).
    Output,
    /// An output written as a bare expression, named by its text.
    Anonymous,
}

/// A named node, see [`BindingKind`].
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Binding {
    name: String,
    node: NodeId,
    kind: BindingKind,
    location: Option<Location>,
}

impl Binding {
    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn node(&self) -> NodeId {
        self.node
    }

    pub(super) fn kind(&self) -> BindingKind {
        self.kind
    }

    pub(super) fn is_output(&self) -> bool {
        self.kind != BindingKind::Helper
    }

    /// Whether expressions can refer to the node by the binding's name.
    pub(super) fn is_named(&self) -> bool {
        self.kind != BindingKind::Anonymous
    }

    /// Where the binding is in the text form.
    pub(super) fn location(&self) -> Option<Location> {
        self.location
    }
}

/// Assembles a [`Pipeline`] node by node.
///
/// Equal computations are shared: asking twice for `ema(x, 10)` returns
/// the same node, so it is computed once per update. Calls on constants
/// are folded into constants.
///
/// ```
/// use sliding_features::pipeline::{BinaryOp, Builder, Function, Input, Limits};
///
/// let mut b = Builder::new(vec![Input::real("x")], Limits::default()).unwrap();
/// let x = b.lookup("x").unwrap();
/// let fast = b.call(Function::Ema, x, &[10.0]).unwrap();
/// let slow = b.call(Function::Ema, x, &[50.0]).unwrap();
/// let cross = b.binary(BinaryOp::Sub, fast, slow).unwrap();
/// b.output("cross", cross).unwrap();
/// let mut pipeline = b.build().unwrap();
/// pipeline.update(&[100.0]);
/// assert_eq!(pipeline.output(0), None, "the EMAs warm up");
/// for _ in 0..50 {
///     pipeline.update(&[100.0]);
/// }
/// assert_eq!(pipeline.output(0), Some(0.0));
/// ```
#[derive(Debug, Clone)]
pub struct Builder {
    graph: Graph,
    /// The depth of every node, see [`Limits::max_depth`].
    depths: Vec<usize>,
    inputs: Vec<Input>,
    /// The node of each input.
    input_nodes: Vec<NodeId>,
    bindings: Vec<Binding>,
    limits: Limits,
    /// Where in the text form each node first appears.
    locations: Vec<Option<Location>>,
    /// Where the text form is, for new nodes, bindings and errors.
    location: Option<Location>,
}

impl Builder {
    /// A builder over `inputs`, which are fed in this order.
    pub fn new(inputs: Vec<Input>, limits: Limits) -> Result<Self, CompileError> {
        if inputs.is_empty() {
            return Err(CompileError::single("a pipeline needs at least one input"));
        }
        let mut builder = Self {
            graph: Graph::default(),
            depths: Vec::new(),
            inputs: Vec::new(),
            input_nodes: Vec::new(),
            bindings: Vec::new(),
            limits,
            locations: Vec::new(),
            location: None,
        };
        for (i, input) in inputs.into_iter().enumerate() {
            builder.check_name(input.name())?;
            let id = builder.add(Node::Input(i))?;
            assert_eq!(id.index(), i, "inputs are the first nodes");
            builder.inputs.push(input);
            builder.input_nodes.push(id);
        }
        Ok(builder)
    }

    /// The input or binding called `name`.
    pub fn lookup(&self, name: &str) -> Option<NodeId> {
        if let Some(i) = self.inputs.iter().position(|input| input.name() == name) {
            return Some(self.input_nodes[i]);
        }
        self.bindings
            .iter()
            .find(|binding| binding.is_named() && binding.name == name)
            .map(|binding| binding.node)
    }

    /// Every name expressions can refer to: inputs, then bindings.
    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        let inputs = self.inputs.iter().map(Input::name);
        let bindings = self
            .bindings
            .iter()
            .filter(|b| b.is_named())
            .map(Binding::name);
        inputs.chain(bindings)
    }

    /// The limits the builder enforces.
    pub(super) fn limits(&self) -> Limits {
        self.limits
    }

    /// Attribute new nodes, bindings and errors to `location` in the text
    /// form.
    pub(super) fn set_location(&mut self, location: Location) {
        self.location = Some(location);
    }

    /// The constant `value`.
    pub fn constant(&mut self, value: f64) -> Result<NodeId, CompileError> {
        if !value.is_finite() {
            return Err(self.error(format!("the constant {value} is not finite")));
        }
        // Adding zero turns -0 into 0, so both are one node.
        self.add(Node::Const(value + 0.0))
    }

    /// `-arg`.
    pub fn neg(&mut self, arg: NodeId) -> Result<NodeId, CompileError> {
        match *self.graph.get(arg) {
            Node::Const(c) => self.constant(-c),
            _ => self.add(Node::Neg(arg)),
        }
    }

    /// `lhs op rhs`.
    pub fn binary(
        &mut self,
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Result<NodeId, CompileError> {
        if let (Node::Const(a), Node::Const(b)) = (self.graph.get(lhs), self.graph.get(rhs)) {
            let (a, b) = (*a, *b);
            let value = op.eval(a, b);
            if !value.is_finite() {
                return Err(self.error(format!("`{a} {} {b}` is not finite", op.symbol())));
            }
            return self.constant(value);
        }
        self.add(Node::Binary(op, lhs, rhs))
    }

    /// `function(arg, params...)`, with parameters checked against the
    /// signature and [`Limits::max_window`].
    pub fn call(
        &mut self,
        function: Function,
        arg: NodeId,
        params: &[f64],
    ) -> Result<NodeId, CompileError> {
        function
            .check_params(params, self.limits.max_window)
            .map_err(|message| self.error(message))?;
        if let Node::Const(c) = *self.graph.get(arg)
            && let Kind::Pure(pure) = function.instantiate(Params::new(function, params))
        {
            let value = pure.eval(c);
            if !value.is_finite() {
                return Err(self.error(format!(
                    "`{}` of the constant {c} is not finite",
                    function.name()
                )));
            }
            return self.constant(value);
        }
        self.add(Node::Call(function, arg, params.into()))
    }

    /// Make `node` an output called `name`. Outputs are read in the order
    /// they are added.
    pub fn output(&mut self, name: &str, node: NodeId) -> Result<(), CompileError> {
        self.check_name(name)?;
        self.push_binding(name.to_string(), node, BindingKind::Output);
        Ok(())
    }

    /// Name `node` without making it an output, so the text form refers to
    /// it by `name` (`let name = ...`).
    pub fn bind(&mut self, name: &str, node: NodeId) -> Result<(), CompileError> {
        self.check_name(name)?;
        self.push_binding(name.to_string(), node, BindingKind::Helper);
        Ok(())
    }

    /// Make `node` an output named by its expression, as written by a bare
    /// expression in the text form.
    pub(super) fn anonymous_output(&mut self, node: NodeId) -> Result<(), CompileError> {
        let name = self.expr_text(node);
        let taken = self
            .bindings
            .iter()
            .any(|b| b.is_output() && b.name() == name);
        if taken {
            return Err(self.error(format!("the output `{name}` is already defined")));
        }
        self.push_binding(name, node, BindingKind::Anonymous);
        Ok(())
    }

    /// The text of `node`, referring to named nodes by name.
    fn expr_text(&self, node: NodeId) -> String {
        let name_of = |id: NodeId| {
            self.bindings
                .iter()
                .find(|b| b.is_named() && b.node() == id)
                .map(Binding::name)
        };
        Printer::new(&self.graph, &self.inputs, name_of).expr(node)
    }

    /// Check the pipeline and return it with fresh state.
    ///
    /// The check derives the [`Domain`] of every node from the inputs'.
    /// It rejects calls that are never ready (e.g. `ln` of a negative
    /// argument), divisions by zero and constant outputs, and warns about
    /// calls that are not always ready, duplicate outputs and unused
    /// helpers, see [`Pipeline::warnings`]. The error lists every problem.
    pub fn build(self) -> Result<Pipeline, CompileError> {
        if !self.bindings.iter().any(Binding::is_output) {
            return Err(CompileError::single("a pipeline needs at least one output"));
        }
        let analysis = Analysis::run(&self.graph, &self.inputs, &self.bindings, &self.locations);
        let diagnostics = analysis.diagnostics();
        if diagnostics.iter().any(|d| d.severity() == Severity::Error) {
            return Err(CompileError::new(diagnostics.to_vec()));
        }
        Ok(Pipeline::new(
            self.graph,
            self.inputs,
            self.bindings,
            &analysis,
        ))
    }

    fn push_binding(&mut self, name: String, node: NodeId, kind: BindingKind) {
        assert!(node.index() < self.graph.len(), "a node of this builder");
        self.bindings.push(Binding {
            name,
            node,
            kind,
            location: self.location,
        });
    }

    /// An error at the current location.
    fn error(&self, message: impl Into<String>) -> CompileError {
        CompileError::single(message).or_at(self.location)
    }

    /// Names are identifiers, so the text form can refer to them, and
    /// unique among inputs and bindings. A name may equal a function's:
    /// calls are followed by `(`, so `rsi = rsi(x, 14)` is unambiguous.
    fn check_name(&self, name: &str) -> Result<(), CompileError> {
        let mut chars = name.chars();
        let identifier = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
        let problem = if !identifier {
            "is not an identifier (a letter or `_`, then letters, digits or `_`)"
        } else if name == "let" {
            "is a keyword"
        } else if self.lookup(name).is_some() {
            "is already defined"
        } else {
            return Ok(());
        };
        Err(self.error(format!("the name `{name}` {problem}")))
    }

    /// Intern `node` within the limits.
    fn add(&mut self, node: Node) -> Result<NodeId, CompileError> {
        let depth = 1 + node
            .children()
            .map(|c| self.depths[c.index()])
            .max()
            .unwrap_or(0);
        if depth > self.limits.max_depth {
            return Err(self.error(format!(
                "the expression is nested {depth} deep, the limit is {}",
                self.limits.max_depth
            )));
        }
        let id = self
            .graph
            .intern(node, self.limits.max_nodes)
            .ok_or_else(|| {
                self.error(format!(
                    "the pipeline needs more than {} distinct nodes",
                    self.limits.max_nodes
                ))
            })?;
        if id.index() == self.depths.len() {
            self.depths.push(depth);
            self.locations.push(self.location);
        }
        assert_eq!(self.depths.len(), self.graph.len(), "a depth per node");
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder() -> Builder {
        Builder::new(vec![Input::real("x")], Limits::default()).unwrap()
    }

    #[test]
    fn equal_calls_share_one_node() {
        let mut b = builder();
        let x = b.lookup("x").unwrap();
        let a = b.call(Function::Ema, x, &[10.0]).unwrap();
        let again = b.call(Function::Ema, x, &[10.0]).unwrap();
        assert_eq!(a, again);
        let other = b.call(Function::Ema, x, &[11.0]).unwrap();
        assert_ne!(a, other);
    }

    #[test]
    fn constants_are_folded() {
        let mut b = builder();
        let two = b.constant(2.0).unwrap();
        let three = b.constant(3.0).unwrap();
        let six = b.binary(BinaryOp::Mul, two, three).unwrap();
        assert_eq!(six, b.constant(6.0).unwrap());
        let neg = b.neg(six).unwrap();
        assert_eq!(neg, b.constant(-6.0).unwrap());
        let abs = b.call(Function::Abs, neg, &[]).unwrap();
        assert_eq!(abs, six);
        assert_eq!(b.constant(-0.0).unwrap(), b.constant(0.0).unwrap());
        let zero = b.constant(0.0).unwrap();
        let err = b.binary(BinaryOp::Div, two, zero).unwrap_err();
        assert_eq!(err.to_string(), "error: `2 / 0` is not finite");
        let err = b.call(Function::Ln, neg, &[]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "error: `ln` of the constant -6 is not finite"
        );
        // Stateful calls on constants are not folded: they are not ready
        // until their window fills.
        let sma = b.call(Function::Sma, two, &[3.0]).unwrap();
        assert_ne!(sma, two);
    }

    #[test]
    fn names_are_checked() {
        let mut b = builder();
        let x = b.lookup("x").unwrap();
        for (name, problem) in [
            ("x", "is already defined"),
            ("let", "is a keyword"),
            ("1a", "is not an identifier"),
            ("a-b", "is not an identifier"),
            ("", "is not an identifier"),
        ] {
            let err = b.output(name, x).unwrap_err().to_string();
            assert!(err.contains(problem), "{name}: {err}");
        }
        b.bind("helper", x).unwrap();
        assert_eq!(b.lookup("helper"), Some(x));
        b.output("sma", x).unwrap();
        assert!(b.output("helper", x).is_err());
        assert!(Builder::new(vec![Input::real("x"), Input::real("x")], Limits::default()).is_err());
        assert!(Builder::new(Vec::new(), Limits::default()).is_err());
    }

    #[test]
    fn limits_are_enforced() {
        let limits = Limits {
            max_window: 10,
            max_nodes: 3,
            max_depth: 3,
        };
        let mut b = Builder::new(vec![Input::real("x")], limits).unwrap();
        let x = b.lookup("x").unwrap();
        assert!(b.call(Function::Sma, x, &[11.0]).is_err());
        let a = b.call(Function::Sma, x, &[10.0]).unwrap();
        let c = b.call(Function::Ema, a, &[10.0]).unwrap();
        let err = b.call(Function::Ema, c, &[10.0]).unwrap_err().to_string();
        assert!(err.contains("nested 4 deep"), "{err}");
        let err = b.call(Function::Ema, x, &[10.0]).unwrap_err().to_string();
        assert!(err.contains("more than 3 distinct nodes"), "{err}");
    }

    #[test]
    fn a_pipeline_needs_an_output() {
        let mut b = builder();
        let x = b.lookup("x").unwrap();
        b.bind("helper", x).unwrap();
        assert!(b.build().is_err());
    }

    #[test]
    #[should_panic(expected = "a node of this builder")]
    fn foreign_nodes_are_rejected() {
        let mut other = builder();
        let x = other.lookup("x").unwrap();
        let foreign = other.call(Function::Sma, x, &[2.0]).unwrap();
        let _ = builder().output("a", foreign);
    }
}
