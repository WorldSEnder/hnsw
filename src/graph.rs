use super::searcher::Searcher;
use num_traits::Zero;
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64 as Rng;
use space::{Knn, KnnPoints, Neighbor};
use std::marker::PhantomData;
use std::{vec, vec::Vec};

mod sealed {
    pub trait Seal {}
}

pub trait NodeData: sealed::Seal {
    fn feature_ix(&self, own_ix: usize) -> usize;
    fn neighbors(&self) -> NeighborsBuf<&[usize]>;
    fn neighbors_mut(&mut self) -> NeighborsBuf<&mut [usize]>;
}

pub trait HnswComponents {
    type Params;
    type L0Node: NodeData;
    type LayerNode: NodeData;
}

pub trait Metric<Q: ?Sized, T: ?Sized = Q> {
    type Unit: Copy + Ord + Zero;
    fn distance(&self, query: &Q, point: &T) -> Self::Unit;
}
pub trait LayerStrategy<T: ?Sized> {
    fn ef_construction(&self) -> usize;
    fn construction_layer(&mut self, point: &T) -> usize;
}

pub trait QueryNodeData: NodeData {
    // This node's index in the next lower layer
    fn lower_index(&self) -> usize;
}
pub trait HnswQueryComponents<Q: ?Sized, T: ?Sized>:
    HnswComponents<Params = Self::QParams, LayerNode = Self::QLayerNode>
{
    type QLayerNode: QueryNodeData;
    type QParams: Metric<Q, T>;
}
impl<Q: ?Sized, T: ?Sized, C: HnswComponents> HnswQueryComponents<Q, T> for C
where
    C::Params: Metric<Q, T>,
    C::LayerNode: QueryNodeData,
{
    type QLayerNode = C::LayerNode;
    type QParams = C::Params;
}

pub trait ConstructZeroNode<Unit>: NodeData {
    fn new_zero_node(nearest: &[Neighbor<Unit>]) -> Self;
}
pub trait ConstructLayerNode<Unit>: NodeData {
    // `next_layer_node` is this node index in the next lower layer
    fn new_node(zero_node: usize, next_layer_node: usize, nearest: &[Neighbor<Unit>]) -> Self;
}
pub trait HnswInsertComponents<T>:
    HnswQueryComponents<
    T,
    T,
    QParams = Self::IParams,
    L0Node = Self::IL0Node,
    QLayerNode = Self::ILayerNode,
>
{
    type IL0Node: ConstructZeroNode<IUnit<Self, T>>;
    type ILayerNode: QueryNodeData + ConstructLayerNode<IUnit<Self, T>>;
    type IParams: Metric<T, T> + LayerStrategy<T>;
}
impl<T, C: HnswComponents> HnswInsertComponents<T> for C
where
    C: HnswQueryComponents<T, T>,
    C::L0Node: ConstructZeroNode<IUnit<Self, T>>,
    C::LayerNode: ConstructLayerNode<IUnit<Self, T>>,
    C::Params: LayerStrategy<T>,
{
    type IParams = C::Params;
    type IL0Node = C::L0Node;
    type ILayerNode = C::LayerNode;
}
type QUnit<C, Q, T> = <<C as HnswQueryComponents<Q, T>>::QParams as Metric<Q, T>>::Unit;
type IUnit<C, T> = QUnit<C, T, T>;

pub struct DefaultComponents<Met = (), const M: usize = 20, const M0: usize = 40, R = Rng> {
    _params: PhantomData<(Met, R)>,
}
pub struct DefaultParams<Met, R = Rng, const M: usize = 20> {
    metric: Met,
    prng: R,
    ef_construction: usize,
}
impl<Met: Default, R: SeedableRng, const M: usize> Default for DefaultParams<Met, R, M> {
    fn default() -> Self {
        Self::from_metric(Met::default())
    }
}
impl<Met, R, const M: usize> DefaultParams<Met, R, M> {
    pub fn ef_construction(self, ef_construction: usize) -> Self {
        let mut this = self;
        this.ef_construction = ef_construction;
        this
    }
}
impl<Met, R: SeedableRng, const M: usize> DefaultParams<Met, R, M> {
    pub fn from_metric(metric: Met) -> Self {
        Self {
            metric,
            prng: R::from_seed(<R as SeedableRng>::Seed::default()),
            ef_construction: 400,
        }
    }
}
impl<Met, R, const M: usize, T: ?Sized, Q: ?Sized> Metric<Q, T> for DefaultParams<Met, R, M>
where
    Met: Metric<Q, T>,
{
    type Unit = <Met as Metric<Q, T>>::Unit;

    fn distance(&self, query: &Q, point: &T) -> Self::Unit {
        self.metric.distance(query, point)
    }
}
impl<Met, R: RngCore, const M: usize, T: ?Sized> LayerStrategy<T> for DefaultParams<Met, R, M> {
    fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    fn construction_layer(&mut self, _point: &T) -> usize {
        let m_l: f64 = (M as f64).ln().recip();
        // log_(1/m) (X) where X ~ Uniform(0, 1)
        // Generate a random u32 here, since this will fit precisely into a f64 without truncation.
        // We do not want to generate 0, since this would lead to INFINITY after ln().
        // Hence, our approximation of Uniform(0, 1) is drawing a number from [1..=2^32]
        // the division is then a linear bias applied after taking logarithms.
        let ranu: f64 = (1. + self.prng.next_u32() as f64).ln();
        let bias: f64 = (1. + u32::MAX as f64).ln();
        let level = (bias - ranu) * m_l;
        level as usize
    }
}
impl<const M0: usize, const M: usize, Met, R> HnswComponents for DefaultComponents<Met, M, M0, R> {
    type Params = DefaultParams<Met, R, M>;
    type L0Node = NeighborNodes<M0>;
    type LayerNode = Node<M>;
}

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement [`space::Metric`] to get implementations.
#[derive(Clone)]
pub struct Hnsw<T, Comps: HnswComponents = DefaultComponents> {
    /// Contains the features of the zero layer.
    /// These are stored separately to allow SIMD speedup in the future by
    /// grouping small worlds of features together.
    features: Vec<T>,
    /// Contains the zero layer.
    zero: Vec<Comps::L0Node>,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Comps::LayerNode>>,
    params: Comps::Params,
}

impl<T, Comps: HnswComponents> Default for Hnsw<T, Comps>
where
    Comps::Params: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Comps: HnswComponents> Hnsw<T, Comps> {
    /// Creates a new HNSW with defaulted params.
    pub fn new() -> Self
    where
        Comps::Params: Default,
    {
        Self::with_params(Default::default())
    }

    /// Creates a new HNSW with the specified params.
    pub fn with_params(params: Comps::Params) -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            params,
        }
    }
}

const UNINIT_NEIGHBOR: usize = !0;

pub struct NeighborsBuf<Buf> {
    len: usize,
    neighbors: Buf,
}
type NeighborsRef<'a> = NeighborsBuf<&'a [usize]>;
type NeighborsMut<'a> = NeighborsBuf<&'a mut [usize]>;

impl<Buf> NeighborsBuf<Buf>
where
    Buf: AsRef<[usize]>,
{
    fn new(neighbors: Buf) -> Self {
        let len = neighbors
            .as_ref()
            .iter()
            .take_while(|&&n| n != UNINIT_NEIGHBOR)
            .count();
        Self { len, neighbors }
    }
    fn iter(&self) -> impl '_ + Iterator<Item = usize> {
        self.neighbors.as_ref()[..self.len].iter().cloned()
    }
    fn is_full(&self) -> bool {
        self.len == self.neighbors.as_ref().len()
    }
    fn push(&mut self, neighbor: usize)
    where
        Buf: AsMut<[usize]>,
    {
        self.neighbors.as_mut()[self.len] = neighbor;
        self.len += 1;
    }
    fn replace(&mut self, ix: usize, new_neighbor: usize)
    where
        Buf: AsMut<[usize]>,
    {
        let neighbors = &mut self.neighbors.as_mut()[..self.len];
        neighbors[ix] = new_neighbor;
    }
}

/// A node in the zero layer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct NeighborNodes<const N: usize> {
    /// The neighbors of this node.
    pub(crate) neighbors: [usize; N], // pub(crate) for serde impl
}

impl<const N: usize> sealed::Seal for NeighborNodes<N> {}
impl<const N: usize> NodeData for NeighborNodes<N> {
    fn feature_ix(&self, own_ix: usize) -> usize {
        own_ix
    }

    fn neighbors(&self) -> NeighborsRef {
        NeighborsBuf::new(&self.neighbors)
    }

    fn neighbors_mut(&mut self) -> NeighborsMut {
        NeighborsBuf::new(&mut self.neighbors)
    }
}
impl<const N: usize, Unit> ConstructZeroNode<Unit> for NeighborNodes<N> {
    fn new_zero_node(nearest: &[Neighbor<Unit>]) -> Self {
        let mut this = Self {
            neighbors: [UNINIT_NEIGHBOR; N],
        };
        for (neighbor, near) in this.neighbors.iter_mut().zip(nearest) {
            *neighbor = near.index;
        }
        this
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct Node<const N: usize> {
    /// The node in the zero layer this refers to.
    zero_node: usize,
    /// The node in the layer below this one that this node corresponds to.
    next_node: usize,
    /// The neighbors in the graph of this node.
    neighbors: NeighborNodes<N>,
}
impl<const N: usize> sealed::Seal for Node<N> {}
impl<const N: usize> NodeData for Node<N> {
    fn feature_ix(&self, _own_ix: usize) -> usize {
        self.zero_node
    }
    fn neighbors(&self) -> NeighborsRef {
        self.neighbors.neighbors()
    }
    fn neighbors_mut(&mut self) -> NeighborsMut {
        self.neighbors.neighbors_mut()
    }
}
impl<const N: usize> QueryNodeData for Node<N> {
    fn lower_index(&self) -> usize {
        self.next_node
    }
}
impl<const N: usize, Unit> ConstructLayerNode<Unit> for Node<N> {
    fn new_node(zero_node: usize, next_layer_node: usize, nearest: &[Neighbor<Unit>]) -> Self {
        Self {
            zero_node,
            next_node: next_layer_node,
            neighbors: ConstructZeroNode::<Unit>::new_zero_node(nearest),
        }
    }
}

pub struct KnnMetric<T: ?Sized, Comps>(PhantomData<(*const T, Comps)>);
impl<T, Comps> space::Metric<T> for KnnMetric<T, Comps>
where
    Comps: HnswQueryComponents<T, T>,
    <Comps::Params as Metric<T, T>>::Unit: num_traits::Unsigned,
{
    type Unit = <Comps::Params as Metric<T, T>>::Unit;

    fn distance(&self, _a: &T, _b: &T) -> Self::Unit {
        panic!("Dont actually call this!");
    }
}
impl<T, Comps> Knn for Hnsw<T, Comps>
where
    Comps: HnswQueryComponents<T, T>,
    <Comps::Params as Metric<T, T>>::Unit: num_traits::Unsigned,
{
    type Ix = usize;
    type Metric = KnnMetric<T, Comps>;
    type Point = T;
    type KnnIter = Vec<Neighbor<<Comps::Params as Metric<T, T>>::Unit>>;

    fn knn(&self, query: &T, num: usize) -> Self::KnnIter {
        let mut searcher = Searcher::default();
        self.nearest(query, num + 16, &mut searcher);
        searcher.nearest
    }
}

impl<T, Comps: HnswComponents> KnnPoints for Hnsw<T, Comps>
where
    Comps: HnswQueryComponents<T, T>,
    <Comps::Params as Metric<T, T>>::Unit: num_traits::Unsigned,
{
    fn get_point(&self, index: usize) -> &'_ T {
        self.feature(index)
    }
}

impl<T, Comps> Hnsw<T, Comps>
where
    Comps: HnswComponents,
{
    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    /// Extract the feature for a given item returned by [`HNSW::nearest`].
    ///
    /// The `item` must be retrieved from [`HNSW::search_layer`].
    pub fn feature(&self, item: usize) -> &T {
        &self.features[item]
    }

    pub fn layers(&self) -> usize {
        self.layers.len() + 1
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

    pub fn layer_is_empty(&self, level: usize) -> bool {
        self.layer_len(level) == 0
    }

    /// Retrieve the item ID for a given layer item returned by [`HNSW::search_layer`].
    pub fn layer_item_id(&self, level: usize, item: usize) -> usize {
        if level == 0 {
            self.zero[item].feature_ix(item)
        } else {
            self.layers[level - 1][item].feature_ix(item)
        }
    }

    /// Extract the feature from a particular level for a given item returned by [`HNSW::search_layer`].
    pub fn layer_feature(&self, level: usize, item: usize) -> &T {
        &self.features[self.layer_item_id(level, item)]
    }
}

impl<T, Comps> Hnsw<T, Comps>
where
    Comps: HnswInsertComponents<T>,
{
    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher<IUnit<Comps, T>>) -> usize {
        // Get the level of this feature.
        let level = self.params.construction_layer(&q);

        // If this is empty, none of this will work, so just add it manually.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(Comps::IL0Node::new_zero_node(&[]));
            self.features.push(q);

            // Add all the layers its in.
            while self.layers.len() < level {
                // It's always index 0 with no neighbors since its the first feature.
                let node = Comps::ILayerNode::new_node(0, 0, &[]);
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(&q, searcher);

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            let cap = 1;
            self.search_single_layer(&q, searcher, &self.layers[ix], cap);
            self.lower_search(&self.layers[ix], searcher);
        }
        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            // Perform an ANN search on this layer like normal.
            let cap = self.params.ef_construction();
            self.search_single_layer(&q, searcher, &self.layers[ix], cap);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(&q, &searcher.nearest, ix);
            // Then lower the search only after we create the node to preserve nearest data
            self.lower_search(&self.layers[ix], searcher);
        }

        // Also search and connect the node to the zero layer.
        let cap = self.params.ef_construction();
        self.search_zero_layer(&q, searcher, cap);
        let zero_node = self.create_zero_node(q, &searcher.nearest);

        // Add all level vectors needed to be able to add this level.
        while self.layers.len() < level {
            let next_node = self.layers.last().map(|l| l.len() - 1).unwrap_or(zero_node);
            let node = Comps::ILayerNode::new_node(zero_node, next_node, &[]);
            self.layers.push(vec![node]);
        }
        zero_node
    }

    fn create_zero_node(&mut self, q: T, nearest: &[Neighbor<IUnit<Comps, T>>]) -> usize {
        let new_index = self.zero.len();
        let node = Comps::IL0Node::new_zero_node(nearest);
        for neighbor in node.neighbors().iter() {
            self.add_neighbor(&q, new_index, neighbor, 0);
        }
        // Add the feature to the zero layer and features
        self.zero.push(node);
        self.features.push(q);
        new_index
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: &T, nearest: &[Neighbor<IUnit<Comps, T>>], layer_ix: usize) {
        let layer = layer_ix + 1;
        let zero_node = self.zero.len();
        let next_node = if let Some(prev_layer) = layer_ix.checked_sub(1) {
            self.layers[prev_layer].len()
        } else {
            zero_node
        };
        let node = Comps::ILayerNode::new_node(zero_node, next_node, nearest);
        let new_index = self.layers[layer_ix].len();
        for neighbor in node.neighbors().iter() {
            self.add_neighbor(q, new_index, neighbor, layer);
        }
        self.layers[layer_ix].push(node);
    }

    fn add_neighbor_impl(
        q: &T,
        new_neighbor: usize,
        target_ix: usize,
        params: &Comps::Params,
        features: &[T],
        layer: &mut [impl NodeData],
    ) {
        // Get the feature for the target and get the neighbor slice for the target.
        // This is different for the zero layer.
        let target = &layer[target_ix];
        let feature_ix = target.feature_ix(target_ix);
        let target_feature = &features[feature_ix];
        let target_neighbors = target.neighbors();

        // Check if there is a point where the target has empty neighbor slots and add it there in that case.
        if !target_neighbors.is_full() {
            // In this case we did find the first spot where the target was empty within the slice.
            // Now we add the neighbor to this slot.
            layer[target_ix].neighbors_mut().push(new_neighbor);
        } else {
            // Otherwise, we need to find the worst neighbor currently.
            let (worst_ix, worst_distance) = target_neighbors
                .iter()
                .enumerate()
                // This was done instead of max_by_key because min_by_key takes the first equally bad element.
                .map(|(ix, neighbor_ix)| {
                    // Compute the distance. The feature is looked up differently for the zero layer.
                    let zero_node = layer[neighbor_ix].feature_ix(neighbor_ix);
                    let distance = params.distance(target_feature, &features[zero_node]);
                    (ix, distance)
                })
                .min_by_key(|&(_, distance)| core::cmp::Reverse(distance))
                .unwrap();

            // If this is better than the worst, insert it in the worst's place.
            // This is also different for the zero layer.
            if params.distance(q, target_feature) < worst_distance {
                layer[target_ix]
                    .neighbors_mut()
                    .replace(worst_ix, new_neighbor);
            }
        }
    }
    /// Attempts to add a neighbor to a target node.
    fn add_neighbor(&mut self, q: &T, new_neighbor: usize, target_ix: usize, layer: usize) {
        // The two layer types are not the same, hence the branches don't unify
        if layer == 0 {
            Self::add_neighbor_impl(
                q,
                new_neighbor,
                target_ix,
                &self.params,
                &self.features,
                &mut self.zero,
            );
        } else {
            Self::add_neighbor_impl(
                q,
                new_neighbor,
                target_ix,
                &self.params,
                &self.features,
                &mut self.layers[layer - 1],
            );
        }
    }
}

impl<T, Comps> Hnsw<T, Comps>
where
    Comps: HnswComponents,
{
    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    /// `ef` is the candidate pool size. `ef` can be increased to get better recall at the expense of speed.
    /// If `ef` is less than `dest.len()` then `dest` will only be filled with `ef` elements.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'s, Q: ?Sized>(
        &self,
        q: &Q,
        ef: usize,
        searcher: &'s mut Searcher<QUnit<Comps, Q, T>>,
    ) -> &'s mut [Neighbor<QUnit<Comps, Q, T>>]
    where
        Comps: HnswQueryComponents<Q, T>,
    {
        self.search_layer(q, ef, 0, searcher)
    }

    /// Performs the same algorithm as [`HNSW::nearest`], but stops on a particular layer of the network
    /// and returns the unique index on that layer rather than the item index.
    ///
    /// If this is passed a `level` of `0`, then this has the exact same functionality as [`HNSW::nearest`]
    /// since the unique indices at layer `0` are the item indices.
    pub fn search_layer<'s, Q: ?Sized>(
        &self,
        q: &Q,
        ef: usize,
        level: usize,
        searcher: &'s mut Searcher<QUnit<Comps, Q, T>>,
    ) -> &'s mut [Neighbor<QUnit<Comps, Q, T>>]
    where
        Comps: HnswQueryComponents<Q, T>,
    {
        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() || level >= self.layers() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher);
        let cap = 1;

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_single_layer(q, searcher, layer, cap);
            if ix + 1 == level {
                return &mut searcher.nearest[..];
            }
            self.lower_search(layer, searcher);
        }

        let cap = ef;

        self.search_zero_layer(q, searcher, cap);

        &mut searcher.nearest[..]
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_single_layer<Q: ?Sized>(
        &self,
        q: &Q,
        searcher: &mut Searcher<QUnit<Comps, Q, T>>,
        layer: &[impl NodeData],
        cap: usize,
    ) where
        Comps: HnswQueryComponents<Q, T>,
    {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            for neighbor in layer[index].neighbors().iter() {
                let neighbor_node = &layer[neighbor];
                let feature_ix = neighbor_node.feature_ix(neighbor);
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(feature_ix) {
                    // Compute the distance of this neighbor.
                    let distance = self.params.distance(q, &self.features[feature_ix]);
                    // Attempt to insert into nearest queue.
                    let pos = searcher.nearest.partition_point(|n| n.distance <= distance);
                    if pos != cap {
                        // It was successful. Now we need to know if its full.
                        if searcher.nearest.len() == cap {
                            // In this case remove the worst item.
                            searcher.nearest.pop();
                        }
                        // Either way, add the new item.
                        let candidate = Neighbor {
                            index: neighbor,
                            distance,
                        };
                        searcher.nearest.insert(pos, candidate);
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer<Q: ?Sized>(
        &self,
        q: &Q,
        searcher: &mut Searcher<QUnit<Comps, Q, T>>,
        cap: usize,
    ) where
        Comps: HnswQueryComponents<Q, T>,
    {
        self.search_single_layer(q, searcher, &self.zero, cap);
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search<Unit: Copy>(
        &self,
        layer: &[impl QueryNodeData],
        searcher: &mut Searcher<Unit>,
    ) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Only preserve the best candidate. The original paper's algorithm uses `1` every time.
        // See Algorithm 5 line 5 of the paper. The paper makes no further comment on why `1` was chosen.
        searcher.nearest.truncate(1);
        // Update all nodes to the next layer.
        for near in &mut searcher.nearest {
            let lower_index = layer[near.index].lower_index();
            near.index = lower_index;
            // Insert the index into the candidate pool as well.
            searcher.candidates.push(*near);
        }
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher<Q: ?Sized>(&self, q: &Q, searcher: &mut Searcher<QUnit<Comps, Q, T>>)
    where
        Comps: HnswQueryComponents<Q, T>,
    {
        // Clear the searcher.
        searcher.clear();
        let entry_layer = self.layers.len();
        let entry_node = 0; // No layer is empty, hence this node exists
        let feature_ix = self.layer_item_id(entry_layer, entry_node);
        // Add the entry point.
        let entry_distance = self.params.distance(q, &self.features[feature_ix]);
        let candidate = Neighbor {
            index: entry_node,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(feature_ix);
    }
}
