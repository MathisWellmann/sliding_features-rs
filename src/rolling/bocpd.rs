//! Bayesian online changepoint detection.

use num::Float;

use crate::View;

/// Normal-Inverse-Gamma posterior parameters of one run length.
#[derive(Debug, Clone, Copy)]
struct NigStats<T> {
    mu: T,
    kappa: T,
    alpha: T,
    beta: T,
    /// `ln Gamma(alpha + 1/2) - ln Gamma(alpha)`, updated with `alpha` so the Student-t
    /// density needs no `ln Gamma` per update.
    ln_gamma_ratio: T,
}

impl<T: Float> NigStats<T> {
    /// Log density of the Student-t posterior predictive at `x`:
    /// `nu = 2 alpha` degrees of freedom, location `mu`, `scale^2 = beta (kappa + 1) / (alpha kappa)`.
    #[inline]
    fn log_predictive(&self, x: T) -> T {
        let two = T::from(2).expect("can convert");
        let nu = two * self.alpha;
        let scale2 = self.beta * (self.kappa + T::one()) / (self.alpha * self.kappa);
        let d = x - self.mu;
        let z2 = d * d / (nu * scale2);
        let pi = T::from(std::f64::consts::PI).expect("can convert");
        self.ln_gamma_ratio - (nu * pi * scale2).ln() / two - (nu + T::one()) / two * z2.ln_1p()
    }

    /// The conjugate posterior after observing `x`.
    #[inline]
    fn updated(&self, x: T) -> Self {
        let two = T::from(2).expect("can convert");
        let kappa = self.kappa + T::one();
        let d = x - self.mu;
        Self {
            mu: (self.kappa * self.mu + x) / kappa,
            kappa,
            alpha: self.alpha + T::one() / two,
            beta: self.beta + self.kappa * d * d / (two * kappa),
            // ln G(a + 1) - ln G(a + 1/2) = ln a + ln G(a) - ln G(a + 1/2)
            ln_gamma_ratio: self.alpha.ln() - self.ln_gamma_ratio,
        }
    }
}

/// `ln Gamma(x)` for `x > 0`: Stirling series after shifting `x` to at least 10 with
/// `ln Gamma(x) = ln Gamma(x + 1) - ln x` (absolute error below 1e-13 in `f64`).
fn ln_gamma<T: Float>(x: T) -> T {
    assert!(x > T::zero(), "ln_gamma needs x > 0");
    let ten = T::from(10).expect("can convert");
    let mut x = x;
    let mut shift = T::zero();
    while x < ten {
        shift = shift + x.ln();
        x = x + T::one();
    }
    let c = |v: f64| T::from(v).expect("can convert");
    let inv = x.recip();
    let inv2 = inv * inv;
    let series = inv
        * (c(1.0 / 12.0)
            - inv2 * (c(1.0 / 360.0) - inv2 * (c(1.0 / 1260.0) - inv2 * c(1.0 / 1680.0))));
    let half_ln_two_pi = c(0.5 * (2.0 * std::f64::consts::PI).ln());
    (x - c(0.5)) * x.ln() - x + half_ln_two_pi + series - shift
}

/// `ln(sum(exp(v)))` without overflow or underflow.
fn log_sum_exp<T: Float>(vals: impl Iterator<Item = T> + Clone) -> T {
    let max = vals.clone().fold(T::neg_infinity(), T::max);
    if max == T::neg_infinity() {
        return max;
    }
    max + vals
        .map(|v| (v - max).exp())
        .fold(T::zero(), |a, b| a + b)
        .ln()
}

/// Bayesian online changepoint detection (BOCPD) for changes of mean and variance of
/// Gaussian data.
///
/// Reference: R. P. Adams, D. J. C. MacKay (2007), "Bayesian Online Changepoint
/// Detection", arXiv:0710.3742.
///
/// Keeps the posterior distribution of the run length `r_t` (number of values since the
/// last changepoint) with a constant hazard `H = 1 / lambda` and, per run length, a
/// Normal-Inverse-Gamma conjugate posterior of the unknown mean and variance; the
/// predictive density of a new value is Student-t. Probabilities are kept in log space,
/// so long runs do not underflow. A value whose squared deviation overflows (beyond
/// about `1e154` on the scale of the prior) is impossible under every run, so it starts
/// a new run with certainty.
///
/// Parameters:
/// - `lambda`: the expected run length between changepoints (`> 1`).
/// - `mu0, kappa0, alpha0, beta0`: the Normal-Inverse-Gamma prior of a new segment, with
///   prior mean `mu0` (worth `kappa0` observations) and an Inverse-Gamma(`alpha0`, `beta0`)
///   prior of the variance (prior variance guess `beta0 / alpha0`).
///   Choose them on the scale of the input.
/// - `max_run_length`: run lengths are truncated at this value. Mass that would grow
///   beyond it is merged into the last bin (meaning "at least `max_run_length`"), which
///   keeps the sufficient statistics of the more probable of the two merged runs; the
///   distribution is renormalized after each update. Merging instead of dropping keeps
///   the MAP at `max_run_length` on long stationary segments.
///
/// Output: the MAP (most probable) run length. Secondary outputs:
/// [`Bocpd::changepoint_probability`] and [`Bocpd::expected_run_length`].
///
/// Warm-up: `None` until the chained `View` yields its first value.
/// Complexity: O(`max_run_length`) time per update, O(`max_run_length`) memory,
/// preallocated in `new`, so updates never allocate.
#[derive(Debug, Clone)]
pub struct Bocpd<T, V> {
    view: V,
    lambda: T,
    log_hazard: T,
    log_growth: T,
    prior: NigStats<T>,
    max_run_length: usize,
    /// `ln P(r_t = r | x_1..t)` for `r < len`.
    log_probs: Vec<T>,
    /// Scratch space for the log predictive densities of the latest value.
    log_preds: Vec<T>,
    stats: Vec<NigStats<T>>,
    len: usize,
    map_run_length: Option<usize>,
    expected_run_length: T,
    changepoint_probability: T,
}

impl<T, V> Bocpd<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new BOCPD with a chained `View`, the expected run length `lambda`, the
    /// Normal-Inverse-Gamma prior `mu0, kappa0, alpha0, beta0` and the truncation
    /// `max_run_length`, see the type docs.
    ///
    /// # Panics
    /// If `lambda <= 1`, `mu0` is not finite, `kappa0`, `alpha0` or `beta0` is not finite
    /// and positive, or `max_run_length == 0`.
    pub fn new(
        view: V,
        lambda: T,
        mu0: T,
        kappa0: T,
        alpha0: T,
        beta0: T,
        max_run_length: usize,
    ) -> Self {
        assert!(lambda.is_finite(), "lambda must be finite");
        assert!(lambda > T::one(), "lambda must be > 1");
        assert!(mu0.is_finite(), "mu0 must be finite");
        for (p, msg) in [
            (kappa0, "kappa0 must be finite and > 0"),
            (alpha0, "alpha0 must be finite and > 0"),
            (beta0, "beta0 must be finite and > 0"),
        ] {
            assert!(p.is_finite(), "{msg}");
            assert!(p > T::zero(), "{msg}");
        }
        assert!(max_run_length > 0, "max_run_length must be > 0");
        let half = T::from(0.5).expect("can convert");
        let prior = NigStats {
            mu: mu0,
            kappa: kappa0,
            alpha: alpha0,
            beta: beta0,
            ln_gamma_ratio: ln_gamma(alpha0 + half) - ln_gamma(alpha0),
        };
        let hazard = lambda.recip();
        let n = max_run_length + 1;
        let mut log_probs = vec![T::neg_infinity(); n];
        // Before the first value, a new run starts with certainty.
        log_probs[0] = T::zero();
        Self {
            view,
            lambda,
            log_hazard: hazard.ln(),
            log_growth: (-hazard).ln_1p(),
            prior,
            max_run_length,
            log_probs,
            log_preds: vec![T::zero(); n],
            stats: vec![prior; n],
            len: 1,
            map_run_length: None,
            expected_run_length: T::zero(),
            changepoint_probability: T::one(),
        }
    }

    /// The expected run length between changepoints (inverse hazard).
    #[inline]
    pub fn lambda(&self) -> T {
        self.lambda
    }

    /// The truncation of the run length.
    #[inline]
    pub fn max_run_length(&self) -> usize {
        self.max_run_length
    }

    /// `P(r_t <= 1 | x_1..t)`: the probability that a changepoint occurred right before the
    /// latest value (`r_t = 1`: the latest value is the first of a new segment) or right
    /// after it (`r_t = 0`).
    /// `P(r_t = 0)` alone is always the hazard `1 / lambda` with a constant hazard, so it
    /// carries no information. `1` before the first update.
    #[inline]
    pub fn changepoint_probability(&self) -> T {
        self.changepoint_probability
    }

    /// The posterior mean of the run length, `0` before the first update.
    #[inline]
    pub fn expected_run_length(&self) -> T {
        self.expected_run_length
    }

    /// Shift the run length distribution by one (growth), merging the mass and
    /// statistics that would exceed `max_run_length` into the last bin.
    fn grow(&mut self, val: T) {
        let n = self.len;
        let top = self.max_run_length;
        let mut start = n;
        if n == top + 1 {
            let from_below = self.log_probs[top - 1] + self.log_preds[top - 1];
            let from_top = self.log_probs[top] + self.log_preds[top];
            let source = if from_below > from_top { top - 1 } else { top };
            self.stats[top] = self.stats[source].updated(val);
            self.log_probs[top] = log_sum_exp([from_below, from_top].into_iter()) + self.log_growth;
            start = top - 1;
        } else {
            self.len = n + 1;
        }
        // Iterate downwards, so every bin is read before it is overwritten.
        for r in (0..start).rev() {
            self.log_probs[r + 1] = self.log_probs[r] + self.log_preds[r] + self.log_growth;
            self.stats[r + 1] = self.stats[r].updated(val);
        }
    }

    /// Normalize the distribution and compute the summary outputs.
    fn normalize_and_summarize(&mut self) {
        let mut total = log_sum_exp(self.log_probs[..self.len].iter().copied());
        if !total.is_finite() {
            // Every run length found the latest value impossible (its
            // squared deviation overflowed): start over with a new run.
            self.len = 1;
            self.log_probs[0] = T::zero();
            self.stats[0] = self.prior;
            total = T::zero();
        }
        let probs = &mut self.log_probs[..self.len];
        let mut map = 0;
        let mut map_lp = T::neg_infinity();
        let mut expected = T::zero();
        for (r, lp) in probs.iter_mut().enumerate() {
            *lp = *lp - total;
            expected = expected + T::from(r).expect("can convert") * lp.exp();
            if *lp > map_lp {
                map = r;
                map_lp = *lp;
            }
        }
        self.map_run_length = Some(map);
        self.expected_run_length = expected;
        self.changepoint_probability = self.log_probs[..self.len.min(2)]
            .iter()
            .map(|lp| lp.exp())
            .fold(T::zero(), |a, b| a + b);
        debug_assert!(expected >= T::zero(), "expected run length is non-negative");
        debug_assert!(
            self.changepoint_probability <= T::one() + T::epsilon().sqrt(),
            "probability is at most one"
        );
    }
}

impl<T, V> View<T> for Bocpd<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let n = self.len;
        for (pred, stats) in self.log_preds[..n].iter_mut().zip(&self.stats[..n]) {
            let log_pred = stats.log_predictive(val);
            // NaN once a squared deviation overflowed (values beyond about
            // 1e154): the value is impossible under that run's posterior.
            *pred = if log_pred.is_nan() {
                T::neg_infinity()
            } else {
                log_pred
            };
        }
        let joint = self.log_probs[..n]
            .iter()
            .zip(&self.log_preds[..n])
            .map(|(lp, pred)| *lp + *pred);
        let log_changepoint = log_sum_exp(joint) + self.log_hazard;
        self.grow(val);
        self.log_probs[0] = log_changepoint;
        self.stats[0] = self.prior;
        self.normalize_and_summarize();
        debug_assert!(
            self.len <= self.max_run_length + 1,
            "run length is truncated"
        );
    }

    #[inline]
    fn last(&self) -> Option<T> {
        let out = T::from(self.map_run_length?).expect("can convert");
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::Sma,
        test_data::TEST_DATA,
    };

    /// Standard normal sample via the Box-Muller transform on romu uniforms.
    fn gaussian(rng: &romu::Rng) -> f64 {
        let u1 = 1.0 - rng.f64(); // in (0, 1], so `ln` is finite
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn bocpd(lambda: f64, max_run_length: usize) -> Bocpd<f64, Echo<f64>> {
        Bocpd::new(Echo::new(), lambda, 0.0, 1.0, 1.0, 1.0, max_run_length)
    }

    fn probs(b: &Bocpd<f64, Echo<f64>>) -> Vec<f64> {
        b.log_probs[..b.len].iter().map(|lp| lp.exp()).collect()
    }

    fn map(b: &Bocpd<f64, Echo<f64>>) -> usize {
        b.last().unwrap() as usize
    }

    /// Values whose squared deviation overflows (beyond about 1e154) make
    /// every run length impossible, and posteriors that saw them have
    /// non-finite scales.
    #[test]
    fn bocpd_survives_overflowing_values() {
        let mut b = bocpd(10.0, 20);
        for v in [
            1.0, 1e200, -1e200, 3e200, 1e300, -1e300, 0.5, 1.0, 1e200, 2.0,
        ] {
            b.update(v);
            let out = b.last().expect("ready after the first value");
            assert!(out.is_finite());
            assert!(b.changepoint_probability().is_finite());
            assert!(b.expected_run_length().is_finite());
        }
        for _ in 0..50 {
            b.update(1.0);
        }
        assert!(map(&b) > 10, "the run grows again after the extreme values");
    }

    #[test]
    fn bocpd_plot() {
        let mut b = Bocpd::new(Echo::new(), 100.0, 100.0, 1.0, 1.0, 1.0, 500);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            b.update(*x);
            out.push(b.last().unwrap());
        }
        plot_values(out, "img/bocpd.png").unwrap();
    }

    #[test]
    fn bocpd_ln_gamma() {
        let pi = std::f64::consts::PI;
        for (x, expected) in [
            (0.5, 0.5 * pi.ln()),
            (1.0, 0.0),
            (2.0, 0.0),
            (5.0, 24.0_f64.ln()),
            (10.5, 13.940625219403763),
            (30.0, 71.25703896716801),
        ] {
            let got = ln_gamma(x);
            assert!((got - expected).abs() < 1e-12, "x {x}: {got} vs {expected}");
        }
    }

    #[test]
    fn bocpd_warmup_and_first_update() {
        let mut b = Bocpd::new(
            Sma::new(Echo::new(), NonZeroUsize::new(2).unwrap()),
            10.0,
            0.0,
            1.0,
            1.0,
            1.0,
            50,
        );
        assert_eq!(b.lambda(), 10.0);
        assert_eq!(b.max_run_length(), 50);
        assert_eq!(b.last(), None);
        assert_eq!(b.changepoint_probability(), 1.0);
        assert_eq!(b.expected_run_length(), 0.0);
        b.update(1.0);
        assert_eq!(b.last(), None);
        b.update(1.0);
        // After one value: P(r = 0) = H, P(r = 1) = 1 - H.
        assert_eq!(b.last(), Some(1.0));
        assert!((b.expected_run_length() - 0.9).abs() < 1e-12);
        assert!((b.changepoint_probability() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bocpd_distribution_invariants() {
        let rng = romu::Rng::from_seed_with_64bit(111);
        let mut b = bocpd(50.0, 64);
        for i in 0..1_000 {
            let level = if (i / 200) % 2 == 0 { 0.0 } else { 3.0 };
            b.update(level + gaussian(&rng));
            let p = probs(&b);
            assert_eq!(p.len(), (i + 2).min(65));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            // With a constant hazard, P(r_t = 0) is the hazard.
            assert!((p[0] - 1.0 / 50.0).abs() < 1e-9);
            let expected: f64 = p.iter().enumerate().map(|(r, p)| r as f64 * p).sum();
            assert!((b.expected_run_length() - expected).abs() < 1e-9);
            assert!((b.changepoint_probability() - p[0] - p[1]).abs() < 1e-12);
            let argmax = (0..p.len()).max_by(|a, c| p[*a].total_cmp(&p[*c])).unwrap();
            assert_eq!(map(&b), argmax);
        }
    }

    #[test]
    fn bocpd_grows_on_stationary_data_and_resets_after_shift() {
        let rng = romu::Rng::from_seed_with_64bit(112);
        let mut b = bocpd(250.0, 1_000);
        for t in 1..=300 {
            b.update(gaussian(&rng));
            if t > 20 {
                assert!(b.changepoint_probability() < 0.5, "false alarm at {t}");
            }
        }
        assert!(map(&b) >= 250, "MAP {}", map(&b));
        assert!(b.expected_run_length() >= 200.0);

        let mut alarm_at = None;
        for t in 1..=5 {
            b.update(4.0 + gaussian(&rng));
            if alarm_at.is_none() && b.changepoint_probability() > 0.5 {
                alarm_at = Some(t);
            }
        }
        assert!(alarm_at.expect("the shift must be detected") <= 3);
        assert!(map(&b) <= 5, "MAP {} after the shift", map(&b));

        for _ in 0..100 {
            b.update(4.0 + gaussian(&rng));
        }
        assert!(map(&b) >= 80, "MAP {} grows again", map(&b));
    }

    #[test]
    fn bocpd_truncated_run_length() {
        // Compare with an effectively untruncated detector: the truncated MAP should be the
        // clamped untruncated MAP, also when a new run reaches the truncation. It can differ
        // when the full posterior is ambiguous, as the merged bin keeps only one run.
        let rng = romu::Rng::from_seed_with_64bit(113);
        let mut b = bocpd(100.0, 20);
        let mut full = bocpd(100.0, 2_000);
        let mut map_mismatches = 0;
        for t in 0..1_000 {
            let level = if (t / 250) % 2 == 0 { 0.0 } else { -4.0 };
            let x = level + gaussian(&rng);
            b.update(x);
            full.update(x);
            assert!(b.len <= 21);
            map_mismatches += usize::from(map(&b) != map(&full).min(20));
            // Truncation only perturbs the change probability slightly (max 0.013 here).
            let diff = b.changepoint_probability() - full.changepoint_probability();
            assert!(diff.abs() < 0.02, "step {t}: {diff}");
            if t % 250 == 5 && t > 5 {
                assert!(map(&b) <= 6, "MAP {} after the shift at {t}", map(&b));
            }
        }
        // 9 of 1000 steps for this seed.
        assert!(map_mismatches <= 25, "{map_mismatches} MAP mismatches");
        assert_eq!(b.len, 21);
        assert_eq!(map(&b), 20);
    }

    #[test]
    fn bocpd_f32() {
        let rng = romu::Rng::from_seed_with_64bit(114);
        let mut b = Bocpd::<f32, _>::new(Echo::new(), 100.0, 0.0, 1.0, 1.0, 1.0, 300);
        for _ in 0..1_000 {
            b.update(gaussian(&rng) as f32);
            assert!(b.last().unwrap().is_finite());
        }
        assert_eq!(b.last(), Some(300.0));
    }

    #[test]
    #[should_panic(expected = "max_run_length must be > 0")]
    fn bocpd_zero_max_run_length() {
        let _ = bocpd(100.0, 0);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 1")]
    fn bocpd_lambda_one() {
        let _ = bocpd(1.0, 10);
    }

    #[test]
    #[should_panic(expected = "kappa0 must be finite and > 0")]
    fn bocpd_zero_kappa() {
        let _ = Bocpd::new(Echo::<f64>::new(), 10.0, 0.0, 0.0, 1.0, 1.0, 10);
    }

    #[test]
    #[should_panic(expected = "alpha0 must be finite and > 0")]
    fn bocpd_negative_alpha() {
        let _ = Bocpd::new(Echo::<f64>::new(), 10.0, 0.0, 1.0, -1.0, 1.0, 10);
    }

    #[test]
    #[should_panic(expected = "beta0 must be finite and > 0")]
    fn bocpd_nan_beta() {
        let _ = Bocpd::new(Echo::<f64>::new(), 10.0, 0.0, 1.0, 1.0, f64::NAN, 10);
    }
}
