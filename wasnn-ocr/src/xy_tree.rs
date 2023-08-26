use std::iter::zip;

use wasnn_imageproc::{BoundingRect, Coord, RectF, RotatedRect};

/// A one-dimensional line or interval.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Line1d<T: Coord> {
    start: T,
    end: T,
}

type Line1dF = Line1d<f32>;

impl<T: Coord> Line1d<T> {
    /// Return a new 1D line from start (inclusive) to end (exclusive).
    fn new(start: T, end: T) -> Line1d<T> {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };

        Line1d { start, end }
    }

    /// Return a new 1D line that spans the X range of a shape.
    fn x_range<BR: BoundingRect<Coord = T>>(shape: &BR) -> Line1d<T> {
        let rect = shape.bounding_rect();
        Line1d::new(rect.left(), rect.right())
    }

    /// Return a new 1D line that spans the Y range of a shape.
    fn y_range<BR: BoundingRect<Coord = T>>(shape: &BR) -> Line1d<T> {
        let rect = shape.bounding_rect();
        Line1d::new(rect.top(), rect.bottom())
    }

    /// Return the length of this line.
    fn len(&self) -> T {
        self.end - self.start
    }

    /// Return true if this line has zero length.
    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Return the intersection of this line with `other`, or `None` if the
    /// lines do not intersect.
    fn intersection(&self, other: Line1d<T>) -> Option<Line1d<T>> {
        let max_start = if self.start >= other.start {
            self.start
        } else {
            other.start
        };
        let min_end = if self.end <= other.end {
            self.end
        } else {
            other.end
        };

        if max_start < min_end {
            Some(Self::new(max_start, min_end))
        } else {
            None
        }
    }

    /// Return an interval which includes this line and `other`.
    fn union(&self, other: Line1d<T>) -> Line1d<T> {
        let min_start = if self.start <= other.start {
            self.start
        } else {
            other.start
        };
        let max_end = if self.end >= other.end {
            self.end
        } else {
            other.end
        };
        Self::new(min_start, max_end)
    }
}

/// Find gaps between items in a slice of sorted intervals. `intervals` must
/// be sorted by their `start` field.
fn find_gaps(intervals: &[Line1dF]) -> Vec<Line1dF> {
    let mut gaps = Vec::new();
    for (curr, next) in zip(intervals.iter(), intervals.iter().skip(1)) {
        if curr.intersection(*next).is_none() {
            gaps.push(Line1d::new(curr.end, next.start));
        }
    }
    gaps
}

/// Returns a sorted sequence of X intervals that do not overlap the bounding
/// rects of any shape in `shapes`.
fn horizontal_gaps<BR: BoundingRect<Coord = f32>>(shapes: &[BR]) -> Vec<Line1dF> {
    let mut hor_lines: Vec<Line1dF> = shapes.iter().map(Line1d::x_range).collect();
    hor_lines.sort_unstable_by(|a, b| a.start.total_cmp(&b.start));
    find_gaps(&hor_lines)
}

/// Returns a sorted sequence of Y intervals that do not overlap the bounding
/// rects of any shape in `shapes`.
fn vertical_gaps<BR: BoundingRect<Coord = f32>>(shapes: &[BR]) -> Vec<Line1dF> {
    let mut ver_lines: Vec<Line1dF> = shapes.iter().map(Line1d::y_range).collect();
    ver_lines.sort_unstable_by(|a, b| a.start.total_cmp(&b.start));
    find_gaps(&ver_lines)
}

/// Thresholds used by [XyNode::partition] when creating an XY-tree.
#[derive(Copy, Clone)]
pub struct PartitionOpts {
    /// Minimum horizontal gap between two objects to create a vertical split.
    /// Must be > 0.
    pub min_horizontal_gap: f32,

    /// Minimum vertical gap between two objects to create a horizontal split.
    /// Must be > 0.
    pub min_vertical_gap: f32,
}

/// A node in an XY-tree. An XY-tree partitions a set of 2D objects using
/// binary horizontal or vertical splits.
#[derive(Clone, Debug, PartialEq)]
pub enum XyNode<T: BoundingRect<Coord = f32> + Clone> {
    /// A leaf node which cannot be partitioned further.
    Leaf(Vec<T>),

    /// A vertical split which partitions objects into those left of an X
    /// coordinate and those to the right of it.
    Vertical {
        left: Box<XyNode<T>>,
        right: Box<XyNode<T>>,
    },

    /// A horizontal split which partitions objects into those above a Y
    /// coordinate and those below it.
    Horizontal {
        top: Box<XyNode<T>>,
        bottom: Box<XyNode<T>>,
    },
}

impl<T: BoundingRect<Coord = f32> + Clone> XyNode<T> {
    /// Create an XY tree which partitions items with 2D bounding boxes.
    ///
    /// At each step, the largest horizontal and vertical gap between all
    /// items is computed. If the gap is larger than the threshold specified
    /// by `opts`, the items are split into left/right or top/bottom groups and
    /// then each group is recursively partitioned. If there is no such gap, the
    /// items are collected into a single leaf node.
    pub fn partition(items: &[T], opts: &PartitionOpts) -> XyNode<T> {
        assert!(opts.min_horizontal_gap > 0.);
        assert!(opts.min_vertical_gap > 0.);

        let h_gaps = horizontal_gaps(items);
        let v_gaps = vertical_gaps(items);

        let max_h_gap = h_gaps.iter().max_by(|a, b| a.len().total_cmp(&b.len()));
        let max_v_gap = v_gaps.iter().max_by(|a, b| a.len().total_cmp(&b.len()));

        // nb. We require min gaps are > 0, so the fallback value here will
        // always be less.
        let max_h_gap_len = max_h_gap.map(|v| v.len()).unwrap_or(0.);
        let max_v_gap_len = max_v_gap.map(|v| v.len()).unwrap_or(0.);

        if max_h_gap_len >= opts.min_horizontal_gap && max_h_gap_len >= max_v_gap_len {
            let (left, right): (Vec<_>, Vec<_>) = items
                .iter()
                .cloned()
                .partition(|r| r.bounding_rect().left() < max_h_gap.unwrap().start);
            let (left_node, right_node) = (
                XyNode::partition(&left, opts),
                XyNode::partition(&right, opts),
            );

            XyNode::Vertical {
                left: Box::new(left_node),
                right: Box::new(right_node),
            }
        } else if max_v_gap_len >= opts.min_vertical_gap {
            let (top, bottom): (Vec<_>, Vec<_>) = items
                .iter()
                .cloned()
                .partition(|r| r.bounding_rect().top() < max_v_gap.unwrap().start);
            let (top_node, bottom_node) = (
                XyNode::partition(&top, opts),
                XyNode::partition(&bottom, opts),
            );

            XyNode::Horizontal {
                top: Box::new(top_node),
                bottom: Box::new(bottom_node),
            }
        } else {
            XyNode::Leaf(items.to_vec())
        }
    }

    /// Traverse the tree in logical order and call `f` with the items from
    /// each leaf node.
    ///
    /// "Logical order" means that the left side of vertical splits is visited
    /// first and the top side of horizontal splits.
    pub fn visit_leaves<F: FnMut(&[T])>(&self, f: &mut F) {
        match self {
            XyNode::Leaf(items) => f(items),
            XyNode::Vertical { left, right } => {
                left.visit_leaves(f);
                right.visit_leaves(f);
            }
            XyNode::Horizontal { top, bottom } => {
                top.visit_leaves(f);
                bottom.visit_leaves(f);
            }
        }
    }

    /// Traverse the tree in logical order and return a concatenation of items
    /// from leaf nodes.
    pub fn items(&self) -> Vec<T> {
        let mut items = Vec::new();
        self.visit_leaves(&mut |leaf_items| {
            items.extend(leaf_items.iter().cloned());
        });
        items
    }
}

#[cfg(test)]
mod tests {
    use wasnn_imageproc::{Rect, RotatedRect};

    use super::{PartitionOpts, XyNode};

    fn rr_tlhw(top: f32, left: f32, height: f32, width: f32) -> RotatedRect {
        RotatedRect::from_rect(Rect::from_tlhw(top, left, height, width))
    }

    fn part_opts() -> PartitionOpts {
        PartitionOpts {
            min_horizontal_gap: 5.,
            min_vertical_gap: 10.,
        }
    }

    fn xy_leaf(rects: &[RotatedRect]) -> XyNode<RotatedRect> {
        XyNode::Leaf(rects.to_vec())
    }

    fn xy_vertical(left: XyNode<RotatedRect>, right: XyNode<RotatedRect>) -> XyNode<RotatedRect> {
        XyNode::Vertical {
            left: left.into(),
            right: right.into(),
        }
    }

    fn xy_horizontal(top: XyNode<RotatedRect>, bottom: XyNode<RotatedRect>) -> XyNode<RotatedRect> {
        XyNode::Horizontal {
            top: top.into(),
            bottom: bottom.into(),
        }
    }

    #[test]
    fn test_xynode_partition() {
        // Single leaf node.
        let rect = rr_tlhw(0., 0., 5., 10.);
        let xy = XyNode::partition(&[rect], &part_opts());
        assert_eq!(xy, xy_leaf(&[rect]));

        // Leaf row (horizontal gaps are < threshold).
        let left = rr_tlhw(0., 0., 5., 10.);
        let right = rr_tlhw(0., 11., 5., 10.);
        let xy = XyNode::partition(&[left, right], &part_opts());
        assert_eq!(xy, xy_leaf(&[left, right]));

        // Leaf column (vertical gaps are < threshold).
        let top = rr_tlhw(0., 0., 5., 10.);
        let bottom = rr_tlhw(8., 0., 5., 10.);
        let xy = XyNode::partition(&[top, bottom], &part_opts());
        assert_eq!(xy, xy_leaf(&[top, bottom]));

        // Single vertical split.
        let left = rr_tlhw(0., 0., 5., 10.);
        let right = rr_tlhw(0., 30., 5., 10.);
        let xy = XyNode::partition(&[left, right], &part_opts());
        assert_eq!(xy, xy_vertical(xy_leaf(&[left]), xy_leaf(&[right])));

        // Single horizontal split
        let top = rr_tlhw(0., 0., 5., 10.);
        let bottom = rr_tlhw(30., 0., 5., 10.);
        let xy = XyNode::partition(&[top, bottom], &part_opts());
        assert_eq!(xy, xy_horizontal(xy_leaf(&[top]), xy_leaf(&[bottom])));

        // Nested split
        let top_left = rr_tlhw(0., 0., 5., 5.);
        let top_right = rr_tlhw(0., 20., 5., 5.);
        let bottom_left = rr_tlhw(20., 0., 5., 5.);
        let bottom_right = rr_tlhw(20., 20., 5., 5.);
        let xy = XyNode::partition(
            &[bottom_right, top_right, bottom_left, top_left],
            &part_opts(),
        );
        assert_eq!(
            xy,
            xy_vertical(
                xy_horizontal(xy_leaf(&[top_left]), xy_leaf(&[bottom_left])),
                xy_horizontal(xy_leaf(&[top_right]), xy_leaf(&[bottom_right]),)
            )
        )
    }

    #[test]
    fn test_xynode_to_ordered() {
        // Single leaf node
        let rect = rr_tlhw(0., 0., 5., 10.);
        let xy = XyNode::partition(&[rect], &part_opts());
        assert_eq!(xy.items(), vec![rect]);

        // Nested split
        let top_left = rr_tlhw(0., 0., 5., 5.);
        let top_right = rr_tlhw(0., 20., 5., 5.);
        let bottom_left = rr_tlhw(20., 0., 5., 5.);
        let bottom_right = rr_tlhw(20., 20., 5., 5.);
        let xy = XyNode::partition(
            &[bottom_right, top_right, bottom_left, top_left],
            &part_opts(),
        );
        assert_eq!(
            xy.items(),
            vec![top_left, bottom_left, top_right, bottom_right]
        )
    }
}
