//! The functions of the spec language and their signatures.

use std::fmt::Write as _;

/// The kind of a parameter that follows the (expression) first argument.
#[deprecated(
    note = "use the `pipeline` module: its text form compiles every spec to the same values"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    /// A window length, counted in updates of the argument: a positive
    /// integer.
    Window,
    /// A finite real number.
    Real,
}

impl Param {
    fn name(self) -> &'static str {
        match self {
            Param::Window => "window",
            Param::Real => "real",
        }
    }
}

/// Declares [`Function`] with its name, parameters, minimum window and
/// description from one table, so they cannot drift apart.
macro_rules! functions {
    ($($variant:ident => $name:literal, [$($param:ident),*], $min:literal, $doc:literal;)*) => {
        /// A function of the spec language, wrapping one `View` of this crate.
        ///
        /// Every function takes an expression as its first argument and then
        /// its [`Function::params`].
        #[deprecated(note = "use `pipeline::Function`")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Function {
            $(#[doc = $doc] $variant,)*
        }

        impl Function {
            /// Every function, in declaration order.
            pub const ALL: &'static [Function] = &[$(Function::$variant,)*];

            /// The name the function is called by in a spec.
            pub fn name(self) -> &'static str {
                match self { $(Function::$variant => $name,)* }
            }

            /// The parameters after the expression argument.
            pub fn params(self) -> &'static [Param] {
                match self { $(Function::$variant => &[$(Param::$param),*],)* }
            }

            /// The smallest window length the function accepts: shorter
            /// windows are degenerate (e.g. an RSI over one value).
            pub fn min_window(self) -> usize {
                match self { $(Function::$variant => $min,)* }
            }

            /// What the function computes, one line.
            pub fn description(self) -> &'static str {
                match self { $(Function::$variant => $doc,)* }
            }
        }
    };
}

functions! {
    Abs => "abs", [], 1, "Absolute value.";
    Ln => "ln", [], 1, "Natural logarithm; not ready for non-positive values.";
    Tanh => "tanh", [], 1, "Hyperbolic tangent, squashes into (-1, 1).";
    LnReturn => "ln_return", [], 1, "Log return between consecutive updates of a positive argument.";
    Drawdown => "drawdown", [], 1, "Largest relative peak-to-valley decline of a positive argument so far.";
    Gte => "gte", [Real], 1, "Clips values below the real to it: max(a, real).";
    Lte => "lte", [Real], 1, "Clips values above the real to it: min(a, real).";
    Sma => "sma", [Window], 1, "Simple moving average.";
    Ema => "ema", [Window], 1, "Exponential moving average, alpha = 2 / (window + 1).";
    Alma => "alma", [Window], 2, "Arnaud Legoux moving average (sigma 6, offset 0.85).";
    Roc => "roc", [Window], 1, "Rate of change in percent against the value `window` updates ago.";
    Lag => "lag", [Window], 1, "The value `window` updates ago.";
    Min => "min", [Window], 1, "Minimum over the window.";
    Max => "max", [Window], 1, "Maximum over the window.";
    MinMax => "min_max", [Window], 2, "Position in the [min, max] range of the previous window, scaled to [-1, 1].";
    Mad => "mad", [Window], 2, "Robust z-score: (a - median) / (1.4826 MAD) of the previous window.";
    ZScore => "zscore", [Window], 2, "Z-score against the mean and standard deviation of the previous window.";
    Std => "std", [Window], 2, "Standard deviation over the window (Welford).";
    Cumulative => "cumulative", [Window], 1, "Sum over the window.";
    Rsi => "rsi", [Window], 2, "Relative strength index.";
    MyRsi => "my_rsi", [Window], 2, "John Ehlers' MyRSI, in [-1, 1].";
    LaguerreRsi => "laguerre_rsi", [Window], 2, "John Ehlers' Laguerre RSI.";
    LaguerreFilter => "laguerre_filter", [Real], 1, "John Ehlers' Laguerre filter with damping gamma in (0, 1).";
    SuperSmoother => "super_smoother", [Window], 2, "John Ehlers' two-pole super smoother low-pass filter.";
    RoofingFilter => "roofing_filter", [Window, Window], 2, "John Ehlers' roofing filter (low-pass window, high-pass window): a band-pass.";
    ReFlex => "re_flex", [Window], 2, "John Ehlers' ReFlex zero-lag cycle indicator.";
    TrendFlex => "trend_flex", [Window], 2, "John Ehlers' TrendFlex zero-lag trend indicator.";
    CyberCycle => "cyber_cycle", [Window], 3, "John Ehlers' cyber cycle.";
    CenterOfGravity => "center_of_gravity", [Window], 2, "John Ehlers' center of gravity oscillator.";
    Cti => "cti", [Window], 2, "John Ehlers' correlation trend indicator: correlation with a straight line, in [-1, 1].";
    Pfe => "pfe", [Window, Window], 3, "Polarized fractal efficiency (window, EMA smoothing window), in [-1, 1].";
    Fisher => "fisher", [Window, Window], 2, "John Ehlers' Fisher transform (window, EMA smoothing window).";
    BinaryEntropy => "binary_entropy", [Window], 2, "Shannon entropy of the signs over the window.";
    Vst => "vst", [Window], 2, "Variance stabilizing transform: change scaled by the rolling standard deviation.";
    Net => "net", [Window], 2, "John Ehlers' noise elimination technology (Kendall correlation).";
}

impl Function {
    /// The function called `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// The call signature, e.g. `sma(a, window)`.
    pub fn signature(self) -> String {
        let mut out = format!("{}(a", self.name());
        for param in self.params() {
            let _ = write!(out, ", {}", param.name());
        }
        out.push(')');
        out
    }

    /// A markdown table of every function: its signature and description.
    /// Meant for documentation and for prompting an agent that writes specs.
    pub fn reference() -> String {
        let mut out = String::from("| function | description |\n|---|---|\n");
        for function in Self::ALL {
            let _ = writeln!(
                out,
                "| `{}` | {} |",
                function.signature(),
                function.description()
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_round_trip() {
        for function in Function::ALL {
            assert_eq!(Function::from_name(function.name()), Some(*function));
            assert!(function.min_window() >= 1);
        }
        let mut names = Vec::from_iter(Function::ALL.iter().map(|f| f.name()));
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Function::ALL.len());
        assert_eq!(Function::from_name("nope"), None);
    }

    #[test]
    fn the_reference_lists_every_signature() {
        let reference = Function::reference();
        assert!(reference.contains("| `pfe(a, window, window)` |"));
        assert!(reference.contains("| `abs(a)` |"));
        assert_eq!(reference.lines().count(), Function::ALL.len() + 2);
    }
}
