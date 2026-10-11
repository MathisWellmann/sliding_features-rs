//! A compiled pipeline and its use as a single view.

use super::{
    CompileError,
    Diagnostic,
    Domain,
    Input,
    Severity,
    analysis::Analysis,
    builder::Binding,
    graph::{
        Graph,
        NodeId,
    },
    program::Program,
};
use crate::View;

/// Features computed from input series by one flat program.
///
/// Every update feeds one value per input and runs each distinct node
/// once, in order, with no allocation, recursion or dynamic dispatch. An
/// output is `None` while not ready: until its windows fill, or while an
/// argument is outside what a function accepts (a division by zero, the
/// logarithm of a non-positive value). Ready outputs are always finite.
///
/// Clone a warmed-up pipeline to branch it; [`Pipeline::reset`] starts it
/// over.
#[derive(Debug, Clone)]
pub struct Pipeline {
    graph: Graph,
    inputs: Vec<Input>,
    bindings: Vec<Binding>,
    /// The domain and warm-up of each output.
    output_domains: Vec<Domain>,
    output_warm_ups: Vec<usize>,
    warnings: Vec<Diagnostic>,
    program: Program,
}

impl Pipeline {
    pub(super) fn new(
        graph: Graph,
        inputs: Vec<Input>,
        bindings: Vec<Binding>,
        analysis: &Analysis,
    ) -> Self {
        assert!(!inputs.is_empty(), "a pipeline has an input");
        let outputs = output_nodes(&bindings);
        let program = Program::lower(&graph, &outputs);
        let warnings = analysis.diagnostics().to_vec();
        assert!(
            warnings.iter().all(|d| d.severity() == Severity::Warning),
            "a pipeline with errors is rejected"
        );
        Self {
            output_domains: Vec::from_iter(outputs.iter().map(|o| analysis.domain(*o))),
            output_warm_ups: Vec::from_iter(outputs.iter().map(|o| analysis.warm_up(*o))),
            warnings,
            graph,
            inputs,
            bindings,
            program,
        }
    }

    /// Feed one value per input, in the order of [`Pipeline::inputs`]. A
    /// non-finite value counts as not ready.
    ///
    /// # Panics
    ///
    /// If the number of values differs from the number of inputs.
    #[inline]
    pub fn update(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.inputs.len(), "one value per input");
        self.program.update(inputs);
    }

    /// The value of the `i`-th output, `None` while not ready.
    ///
    /// # Panics
    ///
    /// If there is no `i`-th output.
    #[inline]
    pub fn output(&self, i: usize) -> Option<f64> {
        self.program.output(i)
    }

    /// The value of every output, in order.
    pub fn outputs(&self) -> impl ExactSizeIterator<Item = Option<f64>> + '_ {
        (0..self.output_count()).map(|i| self.output(i))
    }

    /// The number of outputs.
    pub fn output_count(&self) -> usize {
        self.output_bindings().count()
    }

    /// The name of every output, in order.
    pub fn output_names(&self) -> impl Iterator<Item = &str> {
        self.output_bindings().map(Binding::name)
    }

    /// The index of the output called `name`.
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.output_names().position(|n| n == name)
    }

    /// The values the `i`-th output can take, derived from the inputs'
    /// domains before any data is seen.
    pub fn output_domain(&self, i: usize) -> Domain {
        self.output_domains[i]
    }

    /// An estimate of the updates the `i`-th output needs before it is
    /// ready: its window lengths summed along its longest chain. Use it to
    /// size the history replayed before the output is read.
    pub fn output_warm_up(&self, i: usize) -> usize {
        self.output_warm_ups[i]
    }

    /// The largest warm-up of any output, see [`Pipeline::output_warm_up`].
    pub fn warm_up(&self) -> usize {
        self.output_warm_ups.iter().copied().max().unwrap_or(0)
    }

    /// What the compiler found suspicious but allowed, e.g. a logarithm
    /// of an argument that can be negative.
    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }

    /// The inputs, in the order values are fed.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// The number of operations an update runs: the distinct nodes the
    /// outputs read.
    pub fn op_count(&self) -> usize {
        self.program.len()
    }

    /// Start over: every output is not ready until fed again.
    pub fn reset(&mut self) {
        self.program = Program::lower(&self.graph, &output_nodes(&self.bindings));
    }

    fn output_bindings(&self) -> impl Iterator<Item = &Binding> {
        self.bindings.iter().filter(|b| b.is_output())
    }
}

fn output_nodes(bindings: &[Binding]) -> Vec<NodeId> {
    Vec::from_iter(bindings.iter().filter(|b| b.is_output()).map(Binding::node))
}

/// A pipeline with one input and one output, used as a [`View`], e.g. to
/// chain it with other views or to replace the spec module's `DynView`.
#[derive(Debug, Clone)]
pub struct Feature(Pipeline);

impl Feature {
    /// The pipeline behind the view.
    pub fn pipeline(&self) -> &Pipeline {
        &self.0
    }
}

impl TryFrom<Pipeline> for Feature {
    type Error = CompileError;

    fn try_from(pipeline: Pipeline) -> Result<Self, Self::Error> {
        let (inputs, outputs) = (pipeline.inputs().len(), pipeline.output_count());
        if inputs != 1 || outputs != 1 {
            return Err(CompileError::single(format!(
                "a feature has one input and one output, the pipeline has {inputs} and {outputs}"
            )));
        }
        Ok(Self(pipeline))
    }
}

impl View<f64> for Feature {
    #[inline]
    fn update(&mut self, val: f64) {
        self.0.update(&[val]);
    }

    #[inline]
    fn last(&self) -> Option<f64> {
        self.0.output(0)
    }
}
