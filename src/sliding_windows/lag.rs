use std::{collections::VecDeque, num::NonZeroUsize};

use num::Float;

use crate::View;

/// Lags a value such that it appears n ticks later.
pub struct Lag<T, V> {
    view: V,
    buffer: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Lag<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a `view` and a `window_len`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            buffer: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> usize {
        self.buffer.capacity()
    }
}

impl<T, V> View<T> for Lag<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.buffer.push_back(val);
        if self.buffer.len() >= self.window_len() {
            self.out = self.buffer.pop_front();
        }
    }

    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod test {
    use crate::pure_functions::Echo;

    use super::*;

    #[test]
    fn lag() {
        let mut lag = Lag::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        lag.update(1.0);
        assert_eq!(lag.last(), None);
        lag.update(2.0);
        assert_eq!(lag.last(), None);
        lag.update(3.0);
        assert_eq!(lag.last(), Some(1.0));
        lag.update(4.0);
        assert_eq!(lag.last(), Some(2.0));
        lag.update(5.0);
        assert_eq!(lag.last(), Some(3.0));
    }
}
