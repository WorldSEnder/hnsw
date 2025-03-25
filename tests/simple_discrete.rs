//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::compat::{Hnsw, Searcher};
use hnsw::details::Neighbor;
use rand_pcg::Pcg64;

#[path = "utils/hamming.rs"]
mod hamming;
use hamming::Hamming;

fn test_hnsw_discrete() -> (Hnsw<Hamming, u8, Pcg64, 12, 24>, Searcher<u8>) {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new();

    let features = [
        0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1100, 0b1001,
    ];

    for &feature in &features {
        hnsw.insert(feature, &mut searcher);
    }

    (hnsw, searcher)
}

#[test]
fn insertion_discrete() {
    test_hnsw_discrete();
}

#[test]
fn nearest_neighbor_discrete() {
    let (hnsw, mut searcher) = test_hnsw_discrete();

    let neighbors = hnsw.nearest(&0b0001, 24, &mut searcher);
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(
        neighbors,
        [
            Neighbor {
                index: 0,
                distance: 0
            },
            Neighbor {
                index: 4,
                distance: 1
            },
            Neighbor {
                index: 7,
                distance: 1
            },
            Neighbor {
                index: 1,
                distance: 2
            },
            Neighbor {
                index: 2,
                distance: 2
            },
            Neighbor {
                index: 3,
                distance: 2
            },
            Neighbor {
                index: 5,
                distance: 3
            },
            Neighbor {
                index: 6,
                distance: 3
            }
        ]
    );
}
