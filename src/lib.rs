pub use graph::Hnsw;
pub use searcher::Searcher;

pub mod details {
    pub use super::graph::{DefaultComponents, DefaultParams as Params, Metric};
    pub use super::graph::{HnswComponents, LayerStrategy, NeighborNodes, Node};
    pub use space::Neighbor;
}
pub mod compat {
    /// Type alias for compatbility with the implementation in https://github.com/rust-cv/hnsw
    pub type Hnsw<Met, T, R, const M: usize, const M0: usize> =
        crate::Hnsw<T, crate::details::DefaultComponents<Met, M, M0, R>>;
    pub use crate::Searcher;
    // No details struct!
}

mod graph;
mod searcher;
#[cfg(feature = "serde")]
mod serde_impl;
