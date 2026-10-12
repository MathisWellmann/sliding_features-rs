//! Rolling Hurst exponent via rescaled range (R/S) analysis.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Smallest sub-window size, R/S of fewer points is dominated by noise.
const MIN_SUB_WINDOW: usize = 8;
/// Smallest window, so the regression has at least three sizes (8, 16, 32).
const MIN_WINDOW: usize = 4 * MIN_SUB_WINDOW;

/// Rolling Hurst exponent of the last `window_len` values via classical rescaled range (R/S)
/// analysis after H. E. Hurst, "Long-term storage capacity of reservoirs" (1951) and
/// B. Mandelbrot and J. R. Wallis, "Robustness of the rescaled range R/S" (1969).
///
/// Input convention: the chained input must be increments (returns, price differences),
/// NOT price levels. Chain a differencing / return view in front when working with prices.
///
/// Algorithm, for every sub-window size `s = 8, 16, 32, ...` (powers of two, `s <= window_len`):
/// - split the window into `floor(window_len / s)` non-overlapping chunks, aligned to the end
///   so the newest value is always used (the oldest `window_len mod s` values are skipped),
/// - per chunk, `R` is the range of the cumulative mean-adjusted sum and `S` the population
///   standard deviation; the chunk's `R / S` is averaged over chunks,
/// - `H` is the OLS slope of `ln(mean R/S)` over `ln(s)`.
///
/// Output: typically in `[0, 1]` (not clamped). About 0.5 for uncorrelated increments,
/// above 0.5 for persistent (trending), below 0.5 for anti-persistent (mean-reverting) increments.
/// Classical R/S has a known small-sample upward bias (no Anis-Lloyd correction is applied),
/// so iid increments give slightly above 0.5 for short windows.
///
/// Zero variance: chunks with (numerically) zero standard deviation are skipped. If fewer than
/// two sizes have a valid chunk (e.g. constant input), the output is `None` after warm-up.
///
/// Complexity per update: O(window_len * log(window_len)) (every size level touches the window
/// twice), allocation-free.
#[derive(Debug, Clone, CopyGetters)]
pub struct HurstExponent<T: Float, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    window: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> HurstExponent<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `HurstExponent` with a chained `View` that yields increments.
    ///
    /// # Panics
    /// If `window_len < 32`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= MIN_WINDOW,
            "window_len must be at least 32"
        );
        Self {
            view,
            window_len,
            window: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

/// `R / S` of one chunk, `None` if its standard deviation is numerically zero.
fn rescaled_range<T: Float>(chunk: &[T]) -> Option<T> {
    debug_assert!(chunk.len() >= MIN_SUB_WINDOW, "chunk too short");
    let n = T::from(chunk.len()).expect("can convert");
    let mean = chunk.iter().fold(T::zero(), |acc, &v| acc + v) / n;
    let (mut cum, mut lo, mut hi) = (T::zero(), T::zero(), T::zero());
    let (mut ss, mut max_abs) = (T::zero(), T::zero());
    for &v in chunk {
        let d = v - mean;
        cum = cum + d;
        lo = lo.min(cum);
        hi = hi.max(cum);
        ss = ss + d * d;
        max_abs = max_abs.max(v.abs());
    }
    let std = (ss / n).sqrt();
    // Rounding of the mean leaves residual deviations of order eps * |x| in a constant chunk.
    let noise_floor = T::from(16.0).expect("can convert") * T::epsilon() * max_abs;
    if std <= noise_floor {
        return None;
    }
    let rs = (hi - lo) / std;
    debug_assert!(rs >= T::zero(), "R/S is non-negative");
    Some(rs)
}

/// Mean `R / S` over the end-aligned chunks of size `s`, `None` if no chunk is valid.
fn mean_rescaled_range<T: Float>(x: &[T], s: usize) -> Option<T> {
    let num_chunks = x.len() / s;
    debug_assert!(num_chunks >= 1, "at least one chunk per size");
    let offset = x.len() - num_chunks * s;
    let (sum, count) = x[offset..]
        .chunks_exact(s)
        .filter_map(rescaled_range)
        .fold((T::zero(), 0_usize), |(acc, c), rs| (acc + rs, c + 1));
    (count > 0).then(|| sum / T::from(count).expect("can convert"))
}

/// OLS slope of `ln(mean R/S)` over `ln(s)`, `None` with fewer than two valid sizes.
fn hurst<T: Float>(x: &[T]) -> Option<T> {
    let (mut sx, mut sy, mut sxx, mut sxy, mut k) = (T::zero(), T::zero(), T::zero(), T::zero(), 0);
    let mut s = MIN_SUB_WINDOW;
    while s <= x.len() {
        if let Some(rs) = mean_rescaled_range(x, s).filter(|rs| *rs > T::zero()) {
            let lx = T::from(s).expect("can convert").ln();
            let ly = rs.ln();
            sx = sx + lx;
            sy = sy + ly;
            sxx = sxx + lx * lx;
            sxy = sxy + lx * ly;
            k += 1;
        }
        s *= 2;
    }
    if k < 2 {
        return None;
    }
    let k = T::from(k).expect("can convert");
    let denom = k * sxx - sx * sx;
    debug_assert!(
        denom > T::zero(),
        "distinct sizes give positive denominator"
    );
    Some((k * sxy - sx * sy) / denom)
}

impl<T, V> View<T> for HurstExponent<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.window.len() == self.window_len.get() {
            self.window.pop_front();
        }
        self.window.push_back(val);
        if self.window.len() < self.window_len.get() {
            return;
        }
        self.out = hurst(self.window.make_contiguous());
        if let Some(out) = self.out {
            debug_assert!(out.is_finite(), "value must be finite");
        }
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn new_f64(window_len: usize) -> HurstExponent<f64, Echo<f64>> {
        HurstExponent::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Straightforward reference with explicit vectors.
    fn naive(x: &[f64]) -> f64 {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut s = 8;
        while s <= x.len() {
            let k = x.len() / s;
            let start = x.len() - k * s;
            let mut rs_vals = Vec::new();
            for c in 0..k {
                let chunk = &x[start + c * s..start + (c + 1) * s];
                let mean = chunk.iter().sum::<f64>() / s as f64;
                let cum: Vec<f64> = chunk
                    .iter()
                    .scan(0.0, |acc, v| {
                        *acc += v - mean;
                        Some(*acc)
                    })
                    .collect();
                let r = cum.iter().cloned().fold(f64::MIN, f64::max)
                    - cum.iter().cloned().fold(f64::MAX, f64::min);
                let sd = (chunk.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / s as f64).sqrt();
                rs_vals.push(r / sd);
            }
            let mean_rs = rs_vals.iter().sum::<f64>() / k as f64;
            pts.push(((s as f64).ln(), mean_rs.ln()));
            s *= 2;
        }
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let cov: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let var: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        cov / var
    }

    /// Mean output over a long series, to reduce estimator noise in property tests.
    fn mean_hurst(window_len: usize, increments: &[f64]) -> f64 {
        let mut h = new_f64(window_len);
        let mut outs = Vec::new();
        for (i, v) in increments.iter().enumerate() {
            h.update(*v);
            if i % 64 == 0
                && let Some(o) = h.last()
            {
                outs.push(o);
            }
        }
        outs.iter().sum::<f64>() / outs.len() as f64
    }

    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let rng = romu::Rng::from_seed_with_64bit(seed);
        (0..n).map(|_| rng.f64() - 0.5).collect()
    }

    #[test]
    fn hurst_exponent_warmup() {
        let mut h = new_f64(32);
        let vals = noise(1, 32);
        for v in &vals[..31] {
            h.update(*v);
            assert_eq!(h.last(), None);
        }
        h.update(vals[31]);
        assert!(h.last().is_some());
    }

    #[test]
    fn hurst_exponent_matches_naive() {
        for window_len in [32, 64, 100, 257] {
            let vals = noise(window_len as u64, 600);
            let mut h = new_f64(window_len);
            for (i, v) in vals.iter().enumerate() {
                h.update(*v);
                if i + 1 < window_len {
                    continue;
                }
                let expected = naive(&vals[i + 1 - window_len..=i]);
                let got = h.last().unwrap();
                assert!((got - expected).abs() < 1e-9, "{got} vs {expected}");
            }
        }
    }

    #[test]
    fn hurst_exponent_iid_near_half() {
        let h = mean_hurst(1024, &noise(3, 20_000));
        // Classical R/S is biased upward for small sub-windows.
        assert!(h > 0.45, "{h}");
        assert!(h < 0.65, "{h}");
    }

    #[test]
    fn hurst_exponent_persistent_above_half() {
        // AR(1) increments with positive feedback trend locally.
        let e = noise(4, 20_000);
        let mut x = 0.0;
        let incs: Vec<f64> = e
            .iter()
            .map(|v| {
                x = 0.9 * x + v;
                x
            })
            .collect();
        let h = mean_hurst(1024, &incs);
        let h_iid = mean_hurst(1024, &e);
        assert!(h > 0.7, "{h}");
        assert!(h > h_iid + 0.1, "{h} vs iid {h_iid}");
    }

    #[test]
    fn hurst_exponent_anti_persistent_below_half() {
        // Differenced white noise: every move is undone by the next one.
        let e = noise(5, 20_001);
        let incs: Vec<f64> = e.windows(2).map(|w| w[1] - w[0]).collect();
        let h = mean_hurst(1024, &incs);
        assert!(h < 0.3, "{h}");
    }

    #[test]
    fn hurst_exponent_constant_is_none() {
        let mut h = new_f64(64);
        for _ in 0..100 {
            h.update(0.01);
        }
        assert_eq!(h.last(), None);
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 32")]
    fn hurst_exponent_window_too_small() {
        let _ = new_f64(31);
    }

    #[test]
    fn hurst_exponent_plot() {
        let mut h = new_f64(64);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for w in TEST_DATA.windows(2) {
            h.update(w[1] - w[0]);
            if let Some(val) = h.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/hurst_exponent.png").unwrap();
    }
}
