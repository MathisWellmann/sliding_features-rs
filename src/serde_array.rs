// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Serde for arrays of any length, for `#[serde(with = "crate::serde_array")]`.
//! Serde itself covers arrays of up to 32 elements only.

use std::{
    fmt,
    marker::PhantomData,
};

use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
    de::{
        Error,
        SeqAccess,
        Visitor,
    },
    ser::SerializeTuple,
};

/// Write the elements as a tuple, as serde does for short arrays.
pub(crate) fn serialize<S, T, const N: usize>(
    array: &[T; N],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    let mut tuple = serializer.serialize_tuple(N)?;
    for element in array {
        tuple.serialize_element(element)?;
    }
    tuple.end()
}

/// Read what [`serialize`] wrote: exactly `N` elements.
pub(crate) fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
}

struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
where
    T: Deserialize<'de>,
{
    type Value = [T; N];

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "an array of {N} elements")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<[T; N], A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut elements = Vec::with_capacity(N);
        while let Some(element) = seq.next_element()? {
            if elements.len() == N {
                return Err(A::Error::invalid_length(N + 1, &self));
            }
            elements.push(element);
        }
        let len = elements.len();
        elements
            .try_into()
            .map_err(|_| A::Error::invalid_length(len, &self))
    }
}

#[cfg(test)]
mod tests {
    /// An array longer than serde covers.
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Long {
        #[serde(with = "super")]
        values: [f64; 50],
    }

    #[test]
    fn a_long_array_round_trips() {
        let long = Long {
            values: std::array::from_fn(|i| i as f64 * 0.1),
        };
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&long, config).unwrap();
        let (back, read): (Long, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(read, bytes.len());
        assert_eq!(back, long);
    }

    #[test]
    fn a_wrong_length_does_not_deserialize() {
        #[derive(serde::Serialize)]
        struct Short {
            values: [f64; 3],
        }
        let bytes =
            bincode::serde::encode_to_vec(Short { values: [1.0; 3] }, bincode::config::standard())
                .unwrap();
        let back: Result<(Long, usize), _> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard());
        assert!(back.is_err());
    }
}
