use std::array;
use std::iter::Sum;
use std::ops::Mul;

use rand::distributions::uniform::{SampleRange, SampleUniform};

use opencl_mm::{
    gen,
    scope::{MMul, scope},
};

pub fn mul<T: Mul<Output=T> + Sum<T> + Copy, const N: usize, const M: usize, const K: usize>(
    a: &[[T; M]; N],
    b: &[[T; K]; M],
) -> [[T; K]; N] {
    array::from_fn(|i|
        array::from_fn(|j|
            (0..M).map(|k|
                a[i][k] * b[k][j]
            ).sum()
        )
    )
}

fn test<
    T: SampleUniform + Mul<Output=T> + Sum<T> + MMul + Default + Copy,
    const N: usize,
    const M: usize,
    const K: usize,
    R: SampleRange<T> + Clone,
    F: FnMut(&T, &T) -> bool
>(range: R, mut eq: F) {
    let a = gen::<_, N, M, _>(range.clone());
    let b = gen::<_, M, K, _>(range);
    let c = mul(&a, &b);

    assert!(scope(|s| {
        let p = s.create_with(a).unwrap();
        let q = s.create_with(b).unwrap();
        (p * q).host()
            .iter()
            .flatten()
            .zip(c.iter().flatten())
            .all(|(a, b)| eq(a, b))
    }).unwrap(), "multiplication error")
}

#[test]
fn test0() {
    test::<_, 100, 100, 100, _, _>(0..100, |a, b| a == b)
}

#[test]
fn test1() {
    test::<_, 100, 90, 80, _, _>(0f64..100., |a, b| (a - b).abs() < 1e-4);
}
