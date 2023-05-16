use std::array;

use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::Rng;

pub mod scope;

pub fn gen<T: SampleUniform, const N: usize, const M: usize, R: SampleRange<T> + Clone>(range: R) -> [[T; M]; N] {
    array::from_fn(|_|
        array::from_fn(|_|
            rand::thread_rng().gen_range(range.clone())
        )
    )
}
