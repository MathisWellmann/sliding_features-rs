//! What a call becomes at runtime: a stateless function evaluated inline,
//! or a sliding view of this crate held in one enum.

use super::function::sigmoid;
use crate::{
    View,
    pure_functions::Echo,
    rolling::{
        Drawdown,
        LnReturn,
    },
    sliding_windows::{
        AdaptiveLaguerreFilter,
        AdaptiveRsi,
        AdaptiveSuperSmoother,
        Alma,
        AutoTuneFilter,
        AutocorrelationPeriodogram,
        BandPass,
        BinaryEntropy,
        Butterworth,
        CenterOfGravity,
        Cmo,
        ConnorsRsi,
        ContinuationIndex,
        Coppock,
        CorrelationTrendIndicator,
        Cumulative,
        CyberCycle,
        Decycler,
        DecyclerOscillator,
        Dema,
        Dpo,
        Dsma,
        EhlersFilter,
        EhlersFisherTransform,
        EhlersStochastic,
        Ehma,
        Ema,
        EmpiricalModeDecomposition,
        EvenBetterSinewave,
        GaussianFilter,
        HighPass,
        HilbertPhasor,
        HilbertTrendMode,
        Hma,
        HomodyneDominantCycle,
        InstantaneousTrendline,
        Jma,
        Kama,
        Kst,
        Lag,
        LaguerreFilter,
        LaguerreRSI,
        Macd,
        MadScaler,
        Mama,
        Max,
        McGinleyDynamic,
        Min,
        MinMaxNormalizer,
        MyRSI,
        NoiseEliminationTechnology,
        PolarizedFractalEfficiency,
        Ppo,
        ReFlex,
        ReversionIndex,
        Rma,
        Roc,
        RoofingFilter,
        Rsi,
        Rsx,
        SchaffTrendCycle,
        SineWave,
        Sma,
        StochRsi,
        SuperSmoother,
        T3,
        Tema,
        TrendFlex,
        Trima,
        Trix,
        Tsi,
        UltimateBands,
        UltimateSmoother,
        UniversalOscillator,
        Vidya,
        Vst,
        WelfordOnline,
        Wma,
        ZScoreStandardization,
        Zlema,
    },
};

/// A stateless function of one value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Pure {
    Neg,
    Abs,
    Sign,
    Sqrt,
    Square,
    Exp,
    Ln,
    Tanh,
    Sigmoid,
    Asinh,
    /// John Ehlers' inverse Fisher transform `tanh(gain * v)` with the
    /// gain, as [`crate::pure_functions::InverseFisherTransform`].
    InverseFisher(f64),
    Gte(f64),
    Lte(f64),
}

impl Pure {
    /// The function of `v`. A NaN (not ready) argument gives NaN; outside
    /// its domain a function gives NaN or an infinity, which the caller
    /// turns into "not ready".
    #[inline]
    pub(super) fn eval(self, v: f64) -> f64 {
        // `f64::max` and `f64::min` would replace a NaN by the bound.
        if v.is_nan() {
            return v;
        }
        match self {
            Pure::Neg => -v,
            Pure::Abs => v.abs(),
            // Not `f64::signum`, which is 1 for 0.
            Pure::Sign => {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Pure::Sqrt => v.sqrt(),
            Pure::Square => v * v,
            Pure::Exp => v.exp(),
            Pure::Ln => v.ln(),
            Pure::Tanh => v.tanh(),
            Pure::Sigmoid => sigmoid(v),
            Pure::Asinh => v.asinh(),
            Pure::InverseFisher(gain) => (gain * v).tanh(),
            Pure::Gte(c) => v.max(c),
            Pure::Lte(c) => v.min(c),
        }
    }
}

/// What a call becomes at runtime.
#[derive(Debug, Clone)]
pub(super) enum Kind {
    Pure(Pure),
    Stateful(State),
}

/// Declares [`State`] over the views of this crate, each fed by an [`Echo`]
/// so the pipeline pushes values into it, with uniform `update` and `last`.
macro_rules! states {
    ($($variant:ident($ty:ty),)*) => {
        /// The state of a call to a sliding view.
        #[derive(Debug, Clone)]
        pub(super) enum State {
            $($variant($ty),)*
        }

        impl State {
            /// Push `v` and return the output, NaN while not ready.
            #[inline]
            pub(super) fn update(&mut self, v: f64) -> f64 {
                match self {
                    $(State::$variant(s) => {
                        s.update(v);
                        s.last().unwrap_or(f64::NAN)
                    })*
                }
            }

            /// The last output, NaN while not ready.
            #[inline]
            pub(super) fn last(&self) -> f64 {
                match self {
                    $(State::$variant(s) => s.last().unwrap_or(f64::NAN),)*
                }
            }
        }

        $(impl From<$ty> for Kind {
            fn from(view: $ty) -> Self {
                Kind::Stateful(State::$variant(view))
            }
        })*
    };
}

type E = Echo<f64>;

// Views above about 200 bytes are boxed (clippy's `large_enum_variant`), so
// they do not inflate the slot of every state in a pipeline.
states! {
    LnReturn(LnReturn<f64, E>),
    Drawdown(Drawdown<f64, E>),
    Sma(Sma<f64, E>),
    Ema(Ema<f64, E>),
    Alma(Alma<f64, E>),
    Roc(Roc<f64, E>),
    Lag(Lag<f64, E>),
    Min(Min<f64, E>),
    Max(Max<f64, E>),
    MinMax(MinMaxNormalizer<f64, E>),
    Mad(MadScaler<f64, E>),
    ZScore(ZScoreStandardization<f64, E>),
    Std(WelfordOnline<f64, E>),
    Cumulative(Cumulative<f64, E>),
    Rsi(Rsi<f64, E>),
    MyRsi(MyRSI<f64, E>),
    LaguerreRsi(LaguerreRSI<f64, E>),
    LaguerreFilter(LaguerreFilter<f64, E>),
    SuperSmoother(SuperSmoother<f64, E>),
    RoofingFilter(RoofingFilter<f64, E>),
    ReFlex(ReFlex<f64, E>),
    TrendFlex(TrendFlex<f64, E>),
    CyberCycle(CyberCycle<f64, E>),
    CenterOfGravity(CenterOfGravity<f64, E>),
    Cti(CorrelationTrendIndicator<f64, E>),
    Pfe(PolarizedFractalEfficiency<f64, E, Ema<f64, E>>),
    Fisher(EhlersFisherTransform<f64, E, Ema<f64, E>>),
    BinaryEntropy(BinaryEntropy<f64, E>),
    Vst(Vst<f64, E>),
    Net(NoiseEliminationTechnology<f64, E>),
    HighPass(HighPass<f64, E>),
    BandPass(BandPass<f64, E>),
    UltimateSmoother(UltimateSmoother<f64, E>),
    UltimateBands(UltimateBands<f64, E>),
    Decycler(Decycler<f64, E>),
    DecyclerOscillator(DecyclerOscillator<f64, E>),
    GaussianFilter(GaussianFilter<f64, E>),
    Butterworth(Butterworth<f64, E>),
    EhlersFilter(EhlersFilter<f64, E>),
    Dsma(Dsma<f64, E>),
    ContinuationIndex(Box<ContinuationIndex<f64, E>>),
    Emd(Box<EmpiricalModeDecomposition<f64, E>>),
    HomodyneDominantCycle(Box<HomodyneDominantCycle<f64, E>>),
    HilbertPhasor(Box<HilbertPhasor<f64, E>>),
    SineWave(Box<SineWave<f64, E>>),
    InstantaneousTrendline(Box<InstantaneousTrendline<f64, E>>),
    HilbertTrendMode(Box<HilbertTrendMode<f64, E>>),
    Mama(Box<Mama<f64, E>>),
    AutocorrelationPeriodogram(Box<AutocorrelationPeriodogram<f64, E>>),
    EvenBetterSinewave(Box<EvenBetterSinewave<f64, E>>),
    AdaptiveRsi(Box<AdaptiveRsi<f64, E>>),
    AdaptiveLaguerreFilter(AdaptiveLaguerreFilter<f64, E>),
    AutoTuneFilter(AutoTuneFilter<f64, E>),
    AdaptiveSuperSmoother(AdaptiveSuperSmoother<f64, E>),
    EhlersStochastic(Box<EhlersStochastic<f64, E>>),
    UniversalOscillator(UniversalOscillator<f64, E>),
    ReversionIndex(ReversionIndex<f64, E>),
    Wma(Wma<f64, E>),
    Rma(Rma<f64, E>),
    Dema(Dema<f64, E>),
    Tema(Box<Tema<f64, E>>),
    T3(Box<T3<f64, E>>),
    Zlema(Zlema<f64, E>),
    Hma(Box<Hma<f64, E>>),
    Ehma(Box<Ehma<f64, E>>),
    Trima(Trima<f64, E>),
    McGinleyDynamic(McGinleyDynamic<f64, E>),
    Kama(Kama<f64, E>),
    Vidya(Vidya<f64, E>),
    Jma(Jma<f64, E>),
    Cmo(Cmo<f64, E>),
    Macd(Box<Macd<f64, E>>),
    Ppo(Box<Ppo<f64, E>>),
    Trix(Box<Trix<f64, E>>),
    Tsi(Box<Tsi<f64, E>>),
    StochRsi(Box<StochRsi<f64, E>>),
    Rsx(Rsx<f64, E>),
    SchaffTrendCycle(Box<SchaffTrendCycle<f64, E>>),
    ConnorsRsi(Box<ConnorsRsi<f64, E>>),
    Dpo(Dpo<f64, E>),
    Coppock(Box<Coppock<f64, E>>),
    Kst(Box<Kst<f64, E>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_functions_propagate_not_ready() {
        for f in [
            Pure::Neg,
            Pure::Abs,
            Pure::Sign,
            Pure::Sqrt,
            Pure::Square,
            Pure::Exp,
            Pure::Ln,
            Pure::Tanh,
            Pure::Sigmoid,
            Pure::Asinh,
            Pure::InverseFisher(2.0),
            Pure::Gte(0.0),
            Pure::Lte(0.0),
        ] {
            assert!(f.eval(f64::NAN).is_nan(), "{f:?}");
        }
    }

    #[test]
    fn pure_function_values() {
        assert_eq!(Pure::Sign.eval(0.0), 0.0);
        assert_eq!(Pure::Sign.eval(-3.0), -1.0);
        assert_eq!(Pure::Gte(1.0).eval(0.5), 1.0);
        assert_eq!(Pure::Lte(1.0).eval(0.5), 0.5);
        assert!(Pure::Ln.eval(-1.0).is_nan());
        assert_eq!(Pure::Ln.eval(0.0), f64::NEG_INFINITY);
        assert_eq!(Pure::Square.eval(-3.0), 9.0);
        assert_eq!(Pure::InverseFisher(2.0).eval(0.5), 1.0_f64.tanh());
        assert_eq!(Pure::InverseFisher(1e300).eval(1e300), 1.0);
    }
}
