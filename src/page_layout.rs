use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::zip;

use crate::geometry::{
    find_contours, min_area_rect, simplify_polygon, Line, Point, Rect, RetrievalMode, RotatedRect,
    Vec2,
};
use crate::tensor::NdTensorView;

struct Partition {
    score: f32,
    boundary: Rect,
    obstacles: Vec<Rect>,
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Partition {}

impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Iterator over empty rectangles within a rectangular boundary that contains
/// a set of "obstacles". See [max_empty_rects].
///
/// The order in which rectangles are returned is determined by a scoring
/// function `S`.
pub struct MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    queue: BinaryHeap<Partition>,
    score: S,
    min_width: u32,
    min_height: u32,
}

impl<S> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    fn new(obstacles: &[Rect], boundary: Rect, score: S, min_width: u32, min_height: u32) -> Self {
        let mut queue = BinaryHeap::new();

        // Sort obstacles by X then Y coord. This means that when we choose a pivot
        // from any sub-sequence of `obstacles` we'll also be biased towards picking
        // a central obstacle.
        let mut obstacles = obstacles.to_vec();
        obstacles.sort_by_key(|o| {
            let c = o.center();
            (c.x, c.y)
        });

        if !boundary.is_empty() {
            queue.push(Partition {
                score: score(boundary),
                boundary,
                obstacles: obstacles.to_vec(),
            });
        }

        MaxEmptyRects {
            queue,
            score,
            min_width,
            min_height,
        }
    }
}

impl<S> Iterator for MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        // Assuming the obstacle list is sorted, eg. by X coordinate, choose
        // a pivot that is in the middle.
        let choose_pivot = |r: &[Rect]| r[r.len() / 2];

        while let Some(part) = self.queue.pop() {
            let Partition {
                score: _,
                boundary: b,
                obstacles,
            } = part;

            if obstacles.is_empty() {
                return Some(b);
            }

            let pivot = choose_pivot(&obstacles);
            let right_rect = Rect::from_tlbr(b.top(), pivot.right(), b.bottom(), b.right());
            let left_rect = Rect::from_tlbr(b.top(), b.left(), b.bottom(), pivot.left());
            let top_rect = Rect::from_tlbr(b.top(), b.left(), pivot.top(), b.right());
            let bottom_rect = Rect::from_tlbr(pivot.bottom(), b.left(), b.bottom(), b.right());

            let sub_rects = [top_rect, left_rect, bottom_rect, right_rect];

            for sr in sub_rects {
                if (sr.width().max(0) as u32) < self.min_width
                    || (sr.height().max(0) as u32) < self.min_height
                    || sr.is_empty()
                {
                    continue;
                }

                let sr_obstacles: Vec<_> = obstacles
                    .iter()
                    .filter(|o| o.intersects(sr))
                    .copied()
                    .collect();

                // There should always be fewer obstacles in `sr` since it should
                // not intersect the pivot.
                assert!(sr_obstacles.len() < obstacles.len());

                self.queue.push(Partition {
                    score: (self.score)(sr),
                    obstacles: sr_obstacles,
                    boundary: sr,
                });
            }
        }

        None
    }
}

/// Return an iterator over empty rects in `boundary`, ordered by decreasing
/// value of the `score` function.
///
/// The `score` function must have the property that for any rectangle R and
/// sub-rectangle S that is contained within R, `score(S) <= score(R)`. A
/// typical score function would be the area of the rect, but other functions
/// can be used to favor different aspect ratios.
///
/// `min_width` and `min_height` specify thresholds on the size of rectangles
/// yielded by the iterator.
///
/// The implementation is based on algorithms from [1].
///
/// [1] Breuel, Thomas M. “Two Geometric Algorithms for Layout Analysis.”
///     International Workshop on Document Analysis Systems (2002).
pub fn max_empty_rects<S>(
    obstacles: &[Rect],
    boundary: Rect,
    score: S,
    min_width: u32,
    min_height: u32,
) -> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    MaxEmptyRects::new(obstacles, boundary, score, min_width, min_height)
}

/// Iterator adapter which filters rectangles that overlap rectangles already
/// returned by more than a certain amount.
pub trait FilterOverlapping {
    type Output: Iterator<Item = Rect>;

    /// Create an iterator which filters out rectangles that overlap those
    /// already returned by more than `factor`.
    ///
    /// `factor` is the minimum Intersection-over-Union ratio or Jaccard index [1].
    /// See also [Rect::iou].
    ///
    /// [1] https://en.wikipedia.org/wiki/Jaccard_index
    fn filter_overlapping(self, factor: f32) -> Self::Output;
}

/// Implementation of [FilterOverlapping].
pub struct FilterRectIter<I: Iterator<Item = Rect>> {
    source: I,

    /// Rectangles already found.
    found: Vec<Rect>,

    /// Intersection-over-Union threshold.
    overlap_threshold: f32,
}

impl<I: Iterator<Item = Rect>> FilterRectIter<I> {
    fn new(source: I, overlap_threshold: f32) -> FilterRectIter<I> {
        FilterRectIter {
            source,
            found: Vec::new(),
            overlap_threshold,
        }
    }
}

impl<I: Iterator<Item = Rect>> Iterator for FilterRectIter<I> {
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        while let Some(r) = self.source.next() {
            if self
                .found
                .iter()
                .any(|f| f.iou(r) >= self.overlap_threshold)
            {
                continue;
            }
            self.found.push(r);
            return Some(r);
        }
        None
    }
}

impl<I: Iterator<Item = Rect>> FilterOverlapping for I {
    type Output = FilterRectIter<I>;

    fn filter_overlapping(self, factor: f32) -> Self::Output {
        FilterRectIter::new(self, factor)
    }
}

fn vec_to_point(v: Vec2) -> Point {
    Point::from_yx(v.y as i32, v.x as i32)
}

fn rects_separated_by_line(a: &RotatedRect, b: &RotatedRect, l: Line) -> bool {
    let a_to_b = Line::from_endpoints(vec_to_point(a.center()), vec_to_point(b.center()));
    a_to_b.intersects(l)
}

fn rightmost_edge(r: &RotatedRect) -> Line {
    let mut corners = r.corners();
    corners.sort_by_key(|p| p.x);
    Line::from_endpoints(corners[2], corners[3])
}

fn leftmost_edge(r: &RotatedRect) -> Line {
    let mut corners = r.corners();
    corners.sort_by_key(|p| p.x);
    Line::from_endpoints(corners[0], corners[1])
}

/// Group rects into lines. Each line is a chain of oriented rects ordered
/// left-to-right.
///
/// `separators` is a list of line segments that prevent the formation of
/// lines which cross them. They can be used to specify column boundaries
/// for example.
pub fn group_into_lines(rects: &[RotatedRect], separators: &[Line]) -> Vec<Vec<RotatedRect>> {
    let mut sorted_rects: Vec<_> = rects.to_vec();
    sorted_rects.sort_by_key(|r| r.bounding_rect().left());

    let mut lines: Vec<Vec<_>> = Vec::new();

    // Minimum amount by which two words must overlap vertically to be
    // considered part of the same line.
    let overlap_threshold = 5;

    // Maximum amount by which a candidate word to extend a line from
    // left-to-right may horizontally overlap the current last word in the line.
    //
    // This is necessary when the code that produces the input rects can create
    // overlapping rects. `find_connected_component_rects` pads the rects it
    // produces for example.
    let max_h_overlap = 5;

    while !sorted_rects.is_empty() {
        let mut line = Vec::new();
        line.push(sorted_rects.remove(0));

        // Find the best candidate to extend the current line by one word to the
        // right, and keep going as long as we can find such a candidate.
        loop {
            let last = line.last().unwrap();
            let last_edge = rightmost_edge(last);

            if let Some((i, next_item)) = sorted_rects
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    let edge = leftmost_edge(r);
                    r.center().x > last.center().x
                        && edge.center().x - last_edge.center().x >= -max_h_overlap
                        && last_edge.vertical_overlap(edge) >= overlap_threshold
                        && !separators
                            .iter()
                            .any(|&s| rects_separated_by_line(last, r, s))
                })
                .min_by_key(|(_, r)| r.center().x as i32)
            {
                line.push(*next_item);
                sorted_rects.remove(i);
            } else {
                break;
            }
        }
        lines.push(line);
    }

    lines
}

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
pub fn find_connected_component_rects(
    mask: NdTensorView<i32, 2>,
    expand_dist: f32,
) -> Vec<RotatedRect> {
    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let simplified = simplify_polygon(poly, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .collect()
}

/// Group text words into lines.
pub fn find_text_lines(words: &[RotatedRect], page: Rect) -> Vec<Vec<RotatedRect>> {
    // Estimate spacing statistics
    let mut lines = group_into_lines(&words, &[]);
    lines.sort_by_key(|l| l.first().unwrap().bounding_rect().top());

    let mut all_word_spacings = Vec::new();
    for line in lines.iter() {
        if line.len() > 1 {
            let mut spacings: Vec<_> = zip(line.iter(), line.iter().skip(1))
                .map(|(cur, next)| next.bounding_rect().left() - cur.bounding_rect().right())
                .collect();
            spacings.sort();
            all_word_spacings.extend_from_slice(&spacings);
        }
    }
    all_word_spacings.sort();

    let median_word_spacing = all_word_spacings
        .get(all_word_spacings.len() / 2)
        .copied()
        .unwrap_or(10);
    let median_height = words
        .get(words.len() / 2)
        .map(|r| r.height())
        .unwrap_or(10.)
        .round() as i32;

    // Scoring function for empty rectangles. Taken from Section 3.D in [1].
    // This favors tall rectangles.
    //
    // [1] F. Shafait, D. Keysers and T. Breuel, "Performance Evaluation and
    //     Benchmarking of Six-Page Segmentation Algorithms".
    //     10.1109/TPAMI.2007.70837.
    let score = |r: Rect| {
        let aspect_ratio = (r.height() as f32) / (r.width() as f32);
        let aspect_ratio_weight = match aspect_ratio.log2().abs() {
            r if r < 3. => 0.5,
            r if r < 5. => 1.5,
            r => r,
        };
        ((r.area() as f32) * aspect_ratio_weight).sqrt()
    };

    // Find separators between columns and articles.
    let object_bboxes: Vec<_> = words.iter().map(|r| r.bounding_rect()).collect();
    let min_width = (median_word_spacing * 3) / 2;
    let min_height = (3 * median_height.max(0)) as u32;
    let mut separator_rects = Vec::new();
    for er in max_empty_rects(
        &object_bboxes,
        page,
        score,
        min_width.try_into().unwrap(),
        min_height,
    )
    .filter_overlapping(0.5)
    .take(80)
    {
        separator_rects.push(er);
    }

    // Find lines that do not cross separators
    let separator_lines: Vec<_> = separator_rects
        .iter()
        .map(|r| {
            let center = r.center();
            if r.height() > r.width() {
                Line::from_endpoints(
                    Point::from_yx(r.top(), center.x),
                    Point::from_yx(r.bottom(), center.x),
                )
            } else {
                Line::from_endpoints(
                    Point::from_yx(center.y, r.left()),
                    Point::from_yx(center.y, r.right()),
                )
            }
        })
        .collect();
    group_into_lines(&words, &separator_lines)
}

/// Normalize a line so that it's endpoints are sorted from left to right.
fn normalize_line(l: Line) -> Line {
    if l.start.x <= l.end.x {
        l
    } else {
        Line::from_endpoints(l.end, l.start)
    }
}

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
pub fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut line_points = Vec::new();

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let top_edge = normalize_line(
            word_rect
                .edges()
                .iter()
                .copied()
                .min_by_key(|e| e.center().y)
                .unwrap(),
        );
        line_points.push(top_edge.start);
        line_points.push(top_edge.end);
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let bottom_edge = normalize_line(
            word_rect
                .edges()
                .iter()
                .copied()
                .max_by_key(|e| e.center().y)
                .unwrap(),
        );
        line_points.push(bottom_edge.end);
        line_points.push(bottom_edge.start);
    }

    line_points
}

#[cfg(test)]
mod tests {
    use super::max_empty_rects;
    use crate::geometry::{Point, Rect};

    /// Generate a grid of uniformly sized and spaced rects.
    ///
    /// `grid_shape` is a (rows, columns) tuple. `rect_size` and `gap_size` are
    /// (height, width) tuples.
    fn gen_rect_grid(
        top_left: Point,
        grid_shape: (i32, i32),
        rect_size: (i32, i32),
        gap_size: (i32, i32),
    ) -> Vec<Rect> {
        let mut rects = Vec::new();

        let (rows, cols) = grid_shape;
        let (rect_h, rect_w) = rect_size;
        let (gap_h, gap_w) = gap_size;

        for r in 0..rows {
            for c in 0..cols {
                let top = top_left.y + r * (rect_h + gap_h);
                let left = top_left.x + c * (rect_w + gap_w);
                rects.push(Rect::from_tlbr(top, left, top + rect_h, left + rect_w))
            }
        }

        rects
    }

    /// Return the union of `rects` or `None` if rects is empty.
    fn union_rects(rects: &[Rect]) -> Option<Rect> {
        rects
            .iter()
            .fold(None, |union, r| union.map(|u| u.union(*r)).or(Some(*r)))
    }

    #[test]
    fn test_max_empty_rects() {
        // Create a collection of obstacles that are laid out roughly like
        // words in a two-column document.
        let page = Rect::from_tlbr(0, 0, 80, 90);

        let left_col = gen_rect_grid(
            Point::from_yx(0, 0),
            /* grid_shape */ (10, 5),
            /* rect_size */ (5, 5),
            /* gap_size */ (3, 2),
        );
        let left_col_boundary = union_rects(&left_col).unwrap();
        assert!(page.contains(left_col_boundary));

        let right_col = gen_rect_grid(
            Point::from_yx(0, left_col_boundary.right() + 20),
            /* grid_shape */ (10, 5),
            /* rect_size */ (5, 5),
            /* gap_size */ (3, 2),
        );

        let right_col_boundary = union_rects(&right_col).unwrap();
        assert!(page.contains(right_col_boundary));

        let mut all_cols = left_col.clone();
        all_cols.extend_from_slice(&right_col);

        let max_area_rect = max_empty_rects(&all_cols, page, |r| r.area() as f32, 0, 0).next();

        assert_eq!(
            max_area_rect,
            Some(Rect::from_tlbr(
                page.top(),
                left_col_boundary.right(),
                page.bottom(),
                right_col_boundary.left()
            ))
        );
    }

    #[test]
    fn test_max_empty_rects_if_none() {
        // Case with no empty space within the boundary
        let boundary = Rect::from_tlbr(0, 0, 5, 5);
        assert_eq!(
            max_empty_rects(&[boundary], boundary, |r| r.area() as f32, 0, 0).next(),
            None
        );

        // Case where boundary is empty
        let boundary = Rect::from_hw(0, 0);
        assert_eq!(
            max_empty_rects(&[], boundary, |r| r.area() as f32, 0, 0).next(),
            None
        );
    }
}
