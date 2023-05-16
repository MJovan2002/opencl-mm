#![feature(test)]

extern crate test;

use test::Bencher;

use rand::distributions::uniform::{SampleRange, SampleUniform};

use opencl_mm::gen;
use opencl_mm::scope::{MMul, scope};

fn bench<
    T: SampleUniform + MMul + Default,
    const N: usize,
    const M: usize,
    const K: usize,
    R: SampleRange<T> + Clone
>(b: &mut Bencher, range: R) {
    let time = scope(|s| {
        let p = s.create_with(gen::<T, N, M, _>(range.clone())).unwrap();
        let q = s.create_with(gen::<T, M, K, _>(range)).unwrap();

        struct Avg {
            sum: u64,
            count: u64,
        }

        impl Avg {
            fn push(&mut self, t: u64) {
                self.sum += t;
                self.count += 1;
            }

            fn avg(&self) -> f64 {
                self.sum as f64 / self.count as f64
            }
        }

        let mut time = Avg { sum: 0, count: 0 };
        b.iter(|| p.try_mul(&q, |t| time.push(t)).unwrap());
        time.avg()
    }).unwrap();
}

#[bench]
fn bench0(b: &mut Bencher) {
    bench::<_, 100, 100, 100, _>(b, 0..100);
}

#[bench]
fn bench1(b: &mut Bencher) {
    bench::<_, 100, 100, 100, _>(b, 0f64..100.);
}