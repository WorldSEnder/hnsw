use bitarray::BitArray;
use hnsw::details::Metric;

#[derive(Default)]
pub struct Hamming;

impl Metric<u8> for Hamming {
    type Unit = u8;

    fn distance(&self, &a: &u8, &b: &u8) -> u8 {
        (a ^ b).count_ones() as u8
    }
}

impl<const N: usize> Metric<BitArray<N>> for Hamming {
    type Unit = <bitarray::Hamming as space::Metric<BitArray<N>>>::Unit;

    fn distance(&self, query: &BitArray<N>, point: &BitArray<N>) -> Self::Unit {
        space::Metric::distance(&bitarray::Hamming, query, point)
    }
}
