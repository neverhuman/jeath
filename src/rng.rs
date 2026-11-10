//! Versioned, dependency-free SplitMix64, with Box--Muller normal draws.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        // Strictly inside (0, 1), so logarithms are defined.
        ((self.next_u64() >> 12) as f64 + 0.5) / (1u64 << 52) as f64
    }
    pub fn index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        // Rejection avoids modulo bias.
        let limit = u64::MAX - u64::MAX % n as u64;
        loop {
            let x = self.next_u64();
            if x < limit {
                return (x % n as u64) as usize;
            }
        }
    }
    pub fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.index(i + 1));
        }
    }
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let sum: f64 = weights.iter().sum();
        assert!(sum.is_finite() && sum > 0.0);
        let mut u = self.uniform() * sum;
        for (i, w) in weights.iter().enumerate() {
            u -= w;
            if u <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}
