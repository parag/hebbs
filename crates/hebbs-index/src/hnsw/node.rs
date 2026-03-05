use crate::error::{IndexError, Result};

/// In-memory representation of an HNSW graph node.
///
/// Each node stores its vector, layer assignment, neighbor lists per layer,
/// and a tombstone flag for lazy deletion.
///
/// ## Memory layout per node (384-dim)
///
/// | Component | Bytes |
/// |-----------|-------|
/// | vector (384 * 4) | 1,536 |
/// | neighbors layer 0 (M_max=32 * 16B) | 512 |
/// | neighbors upper layers (M=16 * 16B * ~0.36 avg) | ~92 |
/// | metadata (layer, tombstone, id) | ~24 |
/// | **Total** | **~2,164** |
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub memory_id: [u8; 16],
    pub vector: Vec<f32>,
    pub layer: u8,
    /// neighbors[l] contains the neighbor IDs at layer l.
    /// Length: layer + 1 (layers 0 through self.layer).
    pub neighbors: Vec<Vec<[u8; 16]>>,
    pub deleted: bool,
}

/// Compact binary format for persisting HNSW nodes to the vectors CF.
///
/// Format:
/// ```text
/// [layer: u8]
/// [vector: dimensions * 4 bytes, f32 little-endian]
/// [num_layers: u8]
/// for each layer 0..num_layers:
///   [neighbor_count: u16 LE]
///   [neighbor_ids: neighbor_count * 16 bytes]
/// ```
///
/// This format is owned by hebbs-index and versioned internally.
/// If the format changes, the index is rebuilt from scratch.
impl HnswNode {
    /// Serialize this node to compact binary format for the vectors CF.
    ///
    /// Complexity: O(d + total_neighbors * 16) where d = dimensions.
    pub fn serialize(&self) -> Vec<u8> {
        let neighbor_bytes: usize = self
            .neighbors
            .iter()
            .map(|layer| 2 + layer.len() * 16)
            .sum();
        let capacity = 1 + self.vector.len() * 4 + 1 + neighbor_bytes;
        let mut buf = Vec::with_capacity(capacity);

        buf.push(self.layer);

        for &val in &self.vector {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let num_layers = self.neighbors.len() as u8;
        buf.push(num_layers);

        for layer_neighbors in &self.neighbors {
            let count = layer_neighbors.len() as u16;
            buf.extend_from_slice(&count.to_le_bytes());
            for neighbor_id in layer_neighbors {
                buf.extend_from_slice(neighbor_id);
            }
        }

        buf
    }

    /// Deserialize a node from the vectors CF binary format.
    ///
    /// Requires the memory_id (from the CF key) and the expected
    /// dimensionality (from HnswParams) for validation.
    ///
    /// Complexity: O(d + total_neighbors * 16).
    pub fn deserialize(memory_id: [u8; 16], data: &[u8], dimensions: usize) -> Result<Self> {
        let vector_bytes = dimensions * 4;
        let min_size = 1 + vector_bytes + 1;
        if data.len() < min_size {
            return Err(IndexError::Serialization {
                message: format!(
                    "HNSW node data too short: {} bytes, need at least {}",
                    data.len(),
                    min_size
                ),
            });
        }

        let mut pos = 0;

        let layer = data[pos];
        pos += 1;

        let mut vector = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            if pos + 4 > data.len() {
                return Err(IndexError::Serialization {
                    message: "truncated vector data in HNSW node".to_string(),
                });
            }
            let val = f32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            vector.push(val);
            pos += 4;
        }

        if pos >= data.len() {
            return Err(IndexError::Serialization {
                message: "missing num_layers in HNSW node".to_string(),
            });
        }
        let num_layers = data[pos] as usize;
        pos += 1;

        let mut neighbors = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            if pos + 2 > data.len() {
                return Err(IndexError::Serialization {
                    message: format!("truncated neighbor count at layer {}", l),
                });
            }
            let count = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            let needed = count * 16;
            if pos + needed > data.len() {
                return Err(IndexError::Serialization {
                    message: format!(
                        "truncated neighbor IDs at layer {}: need {} bytes, have {}",
                        l,
                        needed,
                        data.len() - pos
                    ),
                });
            }

            let mut layer_neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                let mut id = [0u8; 16];
                id.copy_from_slice(&data[pos..pos + 16]);
                layer_neighbors.push(id);
                pos += 16;
            }
            neighbors.push(layer_neighbors);
        }

        Ok(HnswNode {
            memory_id,
            vector,
            layer,
            neighbors,
            deleted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node(dims: usize) -> HnswNode {
        let mut vector = Vec::with_capacity(dims);
        for i in 0..dims {
            vector.push(i as f32 * 0.001);
        }

        let id_a = [0xAA; 16];
        let id_b = [0xBB; 16];
        let id_c = [0xCC; 16];

        HnswNode {
            memory_id: [0x01; 16],
            vector,
            layer: 2,
            neighbors: vec![
                vec![id_a, id_b, id_c], // layer 0
                vec![id_a, id_b],       // layer 1
                vec![id_a],             // layer 2
            ],
            deleted: false,
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let node = sample_node(384);
        let data = node.serialize();
        let restored = HnswNode::deserialize(node.memory_id, &data, 384).unwrap();

        assert_eq!(restored.memory_id, node.memory_id);
        assert_eq!(restored.layer, node.layer);
        assert_eq!(restored.vector, node.vector);
        assert_eq!(restored.neighbors.len(), node.neighbors.len());
        for (l, (orig, rest)) in node
            .neighbors
            .iter()
            .zip(restored.neighbors.iter())
            .enumerate()
        {
            assert_eq!(orig, rest, "neighbors differ at layer {}", l);
        }
        assert!(!restored.deleted);
    }

    #[test]
    fn serialization_roundtrip_empty_neighbors() {
        let node = HnswNode {
            memory_id: [0x42; 16],
            vector: vec![1.0, 2.0, 3.0],
            layer: 0,
            neighbors: vec![vec![]],
            deleted: false,
        };
        let data = node.serialize();
        let restored = HnswNode::deserialize(node.memory_id, &data, 3).unwrap();
        assert_eq!(restored.neighbors, vec![Vec::<[u8; 16]>::new()]);
    }

    #[test]
    fn serialization_roundtrip_layer_0_only() {
        let neighbor = [0xFF; 16];
        let node = HnswNode {
            memory_id: [0x01; 16],
            vector: vec![0.5; 10],
            layer: 0,
            neighbors: vec![vec![neighbor]],
            deleted: false,
        };
        let data = node.serialize();
        let restored = HnswNode::deserialize(node.memory_id, &data, 10).unwrap();
        assert_eq!(restored.layer, 0);
        assert_eq!(restored.neighbors.len(), 1);
        assert_eq!(restored.neighbors[0], vec![neighbor]);
    }

    #[test]
    fn deserialize_truncated_data_fails() {
        let result = HnswNode::deserialize([0; 16], &[0x00], 384);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_wrong_dimensions_fails() {
        let node = HnswNode {
            memory_id: [0; 16],
            vector: vec![1.0; 10],
            layer: 0,
            neighbors: vec![vec![]],
            deleted: false,
        };
        let data = node.serialize();
        let result = HnswNode::deserialize([0; 16], &data, 384);
        assert!(result.is_err());
    }

    #[test]
    fn serialized_size_384dim() {
        let node = sample_node(384);
        let data = node.serialize();
        // layer(1) + vector(384*4=1536) + num_layers(1) +
        // layer0: count(2) + 3*16=48 + layer1: count(2) + 2*16=32 + layer2: count(2) + 1*16=16
        let expected = 1 + 1536 + 1 + (2 + 48) + (2 + 32) + (2 + 16);
        assert_eq!(data.len(), expected);
    }
}
