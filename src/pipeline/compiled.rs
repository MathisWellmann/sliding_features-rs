// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A compiled pipeline, its canonical text form and its use as a single
//! view.

use std::{
    fmt,
    str::FromStr,
};

use super::{
    Builder,
    CompileError,
    Diagnostic,
    Domain,
    Input,
    Limits,
    Severity,
    analysis::Analysis,
    builder::{
        Binding,
        BindingKind,
    },
    display::Printer,
    graph::{
        Graph,
        NodeId,
    },
    parse::parse,
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
///
/// The [`fmt::Display`] form is canonical: compiling it yields the same
/// pipeline, so two pipelines that print the same compute the same.
///
/// ```
/// use sliding_features::pipeline::Pipeline;
///
/// let mut p: Pipeline = "zscore(ema(x, 10) - ema(x, 50), 200)".parse().unwrap();
/// assert_eq!(p.to_string(), "zscore(ema(x, 10) - ema(x, 50), 200)");
/// p.update(&[100.0]);
/// assert_eq!(p.output(0), None, "warming up");
/// ```
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

    /// Compile the text form of a pipeline over `inputs`, see
    /// [`reference()`](super::reference()) for the language.
    ///
    /// The error lists every problem with its line and column, worded to
    /// be fixed by the author, e.g. an LLM asked to retry.
    pub fn compile(source: &str, inputs: Vec<Input>, limits: Limits) -> Result<Self, CompileError> {
        let builder = Builder::new(inputs, limits)?;
        parse(source, builder)?.build()
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

impl FromStr for Pipeline {
    type Err = CompileError;

    /// Compile a pipeline over one input `x` that takes any finite value,
    /// within the default [`Limits`].
    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Self::compile(source, vec![Input::real("x")], Limits::default())
    }
}

impl fmt::Display for Pipeline {
    /// One statement per line, in the order written. A node named by an
    /// earlier statement is referred to by that name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, binding) in self.bindings.iter().enumerate() {
            let name_of = |id: NodeId| {
                self.bindings[..k]
                    .iter()
                    .find(|b| b.is_named() && b.node() == id)
                    .map(Binding::name)
            };
            let expr = Printer::new(&self.graph, &self.inputs, name_of).expr(binding.node());
            if k > 0 {
                f.write_str("\n")?;
            }
            match binding.kind() {
                BindingKind::Helper => write!(f, "let {} = {expr}", binding.name())?,
                BindingKind::Output => write!(f, "{} = {expr}", binding.name())?,
                BindingKind::Anonymous => f.write_str(&expr)?,
            }
        }
        Ok(())
    }
}

fn output_nodes(bindings: &[Binding]) -> Vec<NodeId> {
    Vec::from_iter(bindings.iter().filter(|b| b.is_output()).map(Binding::node))
}

/// A pipeline with one input and one output, used as a [`View`], e.g. to
/// chain it with other views.
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
