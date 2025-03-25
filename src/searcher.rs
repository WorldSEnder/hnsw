use space::Neighbor;
use std::collections::HashSet;
use std::hash::RandomState;
use std::{vec, vec::Vec};

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
pub struct Searcher<Unit> {
    pub(super) candidates: Vec<usize>,
    pub(super) nearest: Vec<Neighbor<Unit>>,
    pub(super) seen: HashSet<usize, RandomState>,
}

impl<Unit> Searcher<Unit> {
    pub fn new() -> Self {
        Default::default()
    }

    pub(super) fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<Unit> Default for Searcher<Unit> {
    fn default() -> Self {
        Self {
            candidates: vec![],
            nearest: vec![],
            seen: HashSet::with_hasher(RandomState::new()),
        }
    }
}
