use crate::*;

use alloc::{vec, vec::Vec};
use hashbrown::HashSet;
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;
use rustc_hash::FxHasher;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use space::{CandidatesVec, MetricPoint, Neighbor};

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement `FloatingDistance` to get implementations.
#[derive(Clone)]
#[cfg_attr(
    feature = "serde1",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: Serialize, R: Serialize",
        deserialize = "T: Deserialize<'de>, R: Deserialize<'de>"
    ))
)]
pub struct HNSW<
    T,
    M: ArrayLength<u32> = typenum::U12,
    M0: ArrayLength<u32> = typenum::U24,
    R = Pcg64,
> {
    /// Contains the zero layer.
    zero: Vec<ZeroNode<M0>>,
    /// Contains the features of the zero layer.
    /// These are stored separately to allow SIMD speedup in the future by
    /// grouping small worlds of features together.
    features: Vec<T>,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Node<M>>>,
    /// This needs to create resonably random outputs to determine the levels of insertions.
    prng: R,
    /// The parameters for the HNSW.
    params: Params,
}

/// A node in the zero layer
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize), serde(bound = ""))]
struct ZeroNode<N: ArrayLength<u32>> {
    /// The neighbors of this node.
    neighbors: GenericArray<u32, N>,
}

impl<N: ArrayLength<u32>> ZeroNode<N> {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize), serde(bound = ""))]
struct Node<N: ArrayLength<u32>> {
    /// The node in the zero layer this refers to.
    zero_node: u32,
    /// The node in the layer below this one that this node corresponds to.
    next_node: u32,
    /// The neighbors in the graph of this node.
    neighbors: GenericArray<u32, N>,
}

impl<N: ArrayLength<u32>> Node<N> {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Searcher {
    candidates: Vec<Neighbor>,
    nearest: CandidatesVec,
    seen: HashSet<u32, core::hash::BuildHasherDefault<FxHasher>>,
}

impl Searcher {
    pub fn new() -> Self {
        Default::default()
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> HNSW<T, M, M0, R>
where
    R: RngCore + SeedableRng,
{
    /// Creates a new HNSW with a PRNG which is default seeded to produce deterministic behavior.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new HNSW with a default seeded PRNG and with the specified params.
    pub fn new_params(params: Params) -> Self {
        Self {
            params,
            ..Default::default()
        }
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> HNSW<T, M, M0, R>
where
    R: RngCore,
    T: MetricPoint,
{
    /// Creates a HNSW with the passed `prng`.
    pub fn new_prng(prng: R) -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params: Default::default(),
        }
    }

    /// Creates a HNSW with the passed `params` and `prng`.
    pub fn new_params_and_prng(params: Params, prng: R) -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params,
        }
    }

    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher) -> u32 {
        // Get the level of this feature.
        let level = self.random_level();

        // If this is empty, none of this will work, so just add it manually.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(ZeroNode {
                neighbors: core::iter::repeat(!0).collect(),
            });
            self.features.push(q);

            // Add all the layers its in.
            while self.layers.len() < level {
                // It's always index 0 with no neighbors since its the first feature.
                let node = Node {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: core::iter::repeat(!0).collect(),
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(
            &q,
            searcher,
            if level >= self.layers.len() {
                self.params.ef_construction
            } else {
                1
            },
        );

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, &self.layers[ix]);
            // Then lower the search only after we create the node.
            self.lower_search(
                &self.layers[ix],
                searcher,
                if ix == level {
                    self.params.ef_construction
                } else {
                    1
                },
            );
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, &self.layers[ix]);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(&q, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher, self.params.ef_construction);
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&q, searcher);
        self.create_node(&q, &searcher.nearest, 0);
        // Add the feature to the zero layer.
        self.features.push(q);

        // Add all level vectors needed to be able to add this level.
        let zero_node = (self.zero.len() - 1) as u32;
        while self.layers.len() < level {
            let node = Node {
                zero_node,
                next_node: self
                    .layers
                    .last()
                    .map(|l| (l.len() - 1) as u32)
                    .unwrap_or(zero_node),
                neighbors: core::iter::repeat(!0).collect(),
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    /// `ef` is the candidate pool size. `ef` can be increased to get better recall at the expense of speed.
    /// If `ef` is less than `dest.len()` then `dest` will only be filled with `ef` elements.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'a>(
        &self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher,
        dest: &'a mut [Neighbor],
    ) -> &'a mut [Neighbor] {
        self.search_layer(q, ef, 0, searcher, dest)
    }

    /// Extract the feature for a given item returned by [`HNSW::nearest`].
    ///
    /// The `item` must be retrieved from [`HNSW::search_layer`].
    pub fn feature(&self, item: u32) -> &T {
        &self.features[item as usize]
    }

    /// Extract the feature from a particular level for a given item returned by [`HNSW::search_layer`].
    pub fn layer_feature(&self, level: usize, item: u32) -> &T {
        &self.features[self.layer_item_id(level, item) as usize]
    }

    /// Retrieve the item ID for a given layer item returned by [`HNSW::search_layer`].
    pub fn layer_item_id(&self, level: usize, item: u32) -> u32 {
        if level == 0 {
            item
        } else {
            self.layers[level][item as usize].zero_node
        }
    }

    pub fn layers(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn layer_len(&self, level: usize) -> usize {
        if level == 0 {
            self.features.len()
        } else if level < self.layers() {
            self.layers[level - 1].len()
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    pub fn layer_is_empty(&self, level: usize) -> bool {
        self.layer_len(level) == 0
    }

    /// Performs the same algorithm as [`HNSW::nearest`], but stops on a particular layer of the network
    /// and returns the unique index on that layer rather than the item index.
    ///
    /// If this is passed a `level` of `0`, then this has the exact same functionality as [`HNSW::nearest`]
    /// since the unique indices at layer `0` are the item indices.
    pub fn search_layer<'a>(
        &self,
        q: &T,
        ef: usize,
        level: usize,
        searcher: &mut Searcher,
        dest: &'a mut [Neighbor],
    ) -> &'a mut [Neighbor] {
        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() || level >= self.layers() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher, if self.layers.is_empty() { ef } else { 1 });

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_single_layer(q, searcher, layer);
            if ix + 1 == level {
                return searcher.nearest.fill_slice(dest);
            }
            self.lower_search(layer, searcher, if ix == 0 { ef } else { 1 });
        }

        self.search_zero_layer(q, searcher);

        searcher.nearest.fill_slice(dest)
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_single_layer(&self, q: &T, searcher: &mut Searcher, layer: &[Node<M>]) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            for neighbor in layer[index as usize].neighbors() {
                let neighbor_node = &layer[neighbor as usize];
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor_node.zero_node) {
                    // Compute the distance of this neighbor.
                    let distance = T::distance(q, &self.features[neighbor_node.zero_node as usize]);
                    // Attempt to insert into nearest queue.
                    let candidate = Neighbor {
                        index: neighbor as usize,
                        distance,
                    };
                    if searcher.nearest.push(candidate) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&self, q: &T, searcher: &mut Searcher) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            for neighbor in self.zero[index as usize].neighbors() {
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor) {
                    // Compute the distance of this neighbor.
                    let distance = T::distance(q, &self.features[neighbor as usize]);
                    // Attempt to insert into nearest queue.
                    let candidate = Neighbor {
                        index: neighbor as usize,
                        distance,
                    };
                    if searcher.nearest.push(candidate) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search(&self, layer: &[Node<M>], searcher: &mut Searcher, m: usize) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Only preserve the best candidate. The original paper's algorithm uses `1` every time.
        // See Algorithm 5 line 5 of the paper. The paper makes no further comment on why `1` was chosen.
        let Neighbor { index, distance } = searcher.nearest.best().unwrap();
        searcher.nearest.clear();
        // Set the capacity on the nearest to `m`.
        searcher.nearest.set_cap(m);
        // Update the node to the next layer.
        let new_index = layer[index].next_node as usize;
        let candidate = Neighbor {
            index: new_index,
            distance,
        };
        // Insert the index of the nearest neighbor into the nearest pool for the next layer.
        searcher.nearest.push(candidate);
        // Insert the index into the candidate pool as well.
        searcher.candidates.push(candidate);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&self, q: &T, searcher: &mut Searcher, cap: usize) {
        // Clear the searcher.
        searcher.clear();
        searcher.nearest.set_cap(cap);
        // Add the entry point.
        let entry_distance = T::distance(q, self.entry_feature());
        let candidate = Neighbor {
            index: 0,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(
            self.layers
                .last()
                .map(|layer| layer[0].zero_node)
                .unwrap_or(0),
        );
    }

    /// Gets the entry point's feature.
    fn entry_feature(&self) -> &T {
        if let Some(last_layer) = self.layers.last() {
            &self.features[last_layer[0].zero_node as usize]
        } else {
            &self.features[0]
        }
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.next_u32() as f64 / core::u32::MAX as f64;
        (-libm::log(uniform) * libm::log(M::to_usize() as f64).recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: &T, nearest: &CandidatesVec, layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors: GenericArray<u32, M0> = core::iter::repeat(!0).collect();
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index as u32;
            }
            let node = ZeroNode { neighbors };
            for neighbor in node.neighbors() {
                self.add_neighbor(q, new_index as u32, neighbor, layer);
            }
            self.zero.push(node);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors: GenericArray<u32, M> = core::iter::repeat(!0).collect();
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index as u32;
            }
            let node = Node {
                zero_node: self.zero.len() as u32,
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                } as u32,
                neighbors,
            };
            for neighbor in node.neighbors() {
                self.add_neighbor(q, new_index as u32, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// Attempts to add a neighbor to a target node.
    fn add_neighbor(&mut self, q: &T, node_ix: u32, target_ix: u32, layer: usize) {
        // Get the feature for the target and get the neighbor slice for the target.
        // This is different for the zero layer.
        let (target_feature, target_neighbors) = if layer == 0 {
            (
                &self.features[target_ix as usize],
                &self.zero[target_ix as usize].neighbors[..],
            )
        } else {
            let target = &self.layers[layer - 1][target_ix as usize];
            (
                &self.features[target.zero_node as usize],
                &target.neighbors[..],
            )
        };

        // Get the worst neighbor of this node currently.
        let (worst_ix, worst_distance) = target_neighbors
            .iter()
            .enumerate()
            .map(|(ix, &n)| {
                // Compute the distance to be higher than possible if the neighbor is not filled yet so its always filled.
                let distance = if n == !0 {
                    core::u32::MAX
                } else {
                    // Compute the distance. The feature is looked up differently for the zero layer.
                    T::distance(
                        target_feature,
                        &self.features[if layer == 0 {
                            n as usize
                        } else {
                            self.layers[layer - 1][n as usize].zero_node as usize
                        }],
                    )
                };
                (ix, distance)
            })
            // This was done instead of max_by_key because min_by_key takes the first equally bad element.
            .min_by_key(|&(_, distance)| !distance)
            .unwrap();

        // If this is better than the worst, insert it in the worst's place.
        // This is also different for the zero layer.
        if T::distance(q, target_feature) < worst_distance {
            if layer == 0 {
                self.zero[target_ix as usize].neighbors[worst_ix] = node_ix;
            } else {
                self.layers[layer - 1][target_ix as usize].neighbors[worst_ix] = node_ix;
            }
        }
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> Default for HNSW<T, M, M0, R>
where
    R: SeedableRng,
{
    fn default() -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params: Params::new(),
        }
    }
}
