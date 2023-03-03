///! Geometry functions for pre and post-processing images.
///!
///! TODO: Move these out of Wasnn and into a separate crate.
use std::fmt::Display;
use std::ops::Range;
use std::slice::Iter;

use crate::tensor::{MatrixLayout, NdTensor, NdTensorView, NdTensorViewMut};

pub type Coord = i32;

/// A point defined by integer X and Y coordinates.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point {
    pub x: Coord,
    pub y: Coord,
}

impl Point {
    pub fn from_yx(y: Coord, x: Coord) -> Point {
        Point { y, x }
    }

    /// Return self as a [y, x] index.
    pub fn coord(self) -> [usize; 2] {
        [self.y as usize, self.x as usize]
    }

    pub fn translate(self, y: Coord, x: Coord) -> Point {
        Point {
            y: self.y + y,
            x: self.x + x,
        }
    }

    /// Return the neighbors of the current point in clockwise order, starting
    /// from the point directly above `self`.
    pub fn neighbors(self) -> [Point; 8] {
        [
            self.translate(-1, 0),  // N
            self.translate(-1, 1),  // NE
            self.translate(0, 1),   // E
            self.translate(1, 1),   // SE
            self.translate(1, 0),   // S
            self.translate(1, -1),  // SW
            self.translate(0, -1),  // W
            self.translate(-1, -1), // NW
        ]
    }
}

/// Rectangle defined by left, top, right and bottom integer coordinates.
///
/// The left and top coordinates are inclusive. The right and bottom coordinates
/// are exclusive.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect {
    top_left: Point,
    bottom_right: Point,
}

impl Rect {
    /// Return a rect with top-left corner at 0, 0 and the given height/width.
    pub fn from_hw(height: Coord, width: Coord) -> Rect {
        Self::new(Point::default(), Point::from_yx(height, width))
    }

    /// Return a rect with the given top, left, bottom and right coordinates.
    pub fn from_tlbr(top: Coord, left: Coord, bottom: Coord, right: Coord) -> Rect {
        Self::new(Point::from_yx(top, left), Point::from_yx(bottom, right))
    }

    pub fn new(top_left: Point, bottom_right: Point) -> Rect {
        Rect {
            top_left,
            bottom_right,
        }
    }

    pub fn top(&self) -> Coord {
        self.top_left.y
    }

    pub fn left(&self) -> Coord {
        self.top_left.x
    }

    pub fn right(&self) -> Coord {
        self.bottom_right.x
    }

    pub fn bottom(&self) -> Coord {
        self.bottom_right.y
    }

    /// Return the top, left, bottom and right coordinates as an array.
    pub fn tlbr(&self) -> [Coord; 4] {
        [
            self.top_left.y,
            self.top_left.x,
            self.bottom_right.y,
            self.bottom_right.x,
        ]
    }

    /// Return a new Rect with each coordinate adjusted by an offset.
    pub fn adjust_tlbr(&self, top: Coord, left: Coord, bottom: Coord, right: Coord) -> Rect {
        Rect {
            top_left: self.top_left.translate(top, left),
            bottom_right: self.bottom_right.translate(bottom, right),
        }
    }

    /// Return a new with each side adjusted so that the result lies inside
    /// `rect`.
    pub fn clamp(&self, rect: Rect) -> Rect {
        let top = self.top().max(rect.top());
        let left = self.left().max(rect.left());
        let bottom = self.bottom().min(rect.bottom());
        let right = self.right().min(rect.right());
        Rect {
            top_left: Point::from_yx(top, left),
            bottom_right: Point::from_yx(bottom, right),
        }
    }
}

/// A collection of polygons, where each polygon is defined by a slice of points.
///
/// `Polygons` is primarily useful when building up collections of many polygons
/// as it stores all points in a single Vec, which is more efficient than
/// allocating a separate Vec for each polygon's points.
pub struct Polygons {
    points: Vec<Point>,

    // Offsets within `points` where each polygon starts and ends.
    polygons: Vec<Range<usize>>,
}

impl Polygons {
    /// Construct an empty polygon collection.
    pub fn new() -> Polygons {
        Polygons {
            points: Vec::new(),
            polygons: Vec::new(),
        }
    }

    /// Add a new polygon to the list, defined by the given points.
    pub fn push(&mut self, points: &[Point]) {
        let range = self.points.len()..self.points.len() + points.len();
        self.polygons.push(range);
        self.points.extend_from_slice(points);
    }

    /// Return the number of polygons in the collection.
    pub fn len(&self) -> usize {
        self.polygons.len()
    }

    /// Return true if this collection has no polygons.
    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// Return an iterator over individual polygons in the sequence.
    pub fn iter(&self) -> PolygonsIter {
        PolygonsIter {
            points: &self.points,
            polygons: self.polygons.iter(),
        }
    }
}

impl Default for Polygons {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over polygons in a [Polygons] collection.
pub struct PolygonsIter<'a> {
    points: &'a [Point],
    polygons: Iter<'a, Range<usize>>,
}

impl<'a> Iterator for PolygonsIter<'a> {
    type Item = &'a [Point];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(range) = self.polygons.next() {
            Some(&self.points[range.clone()])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.polygons.size_hint()
    }
}

impl<'a> ExactSizeIterator for PolygonsIter<'a> {}

enum Direction {
    Clockwise,
    CounterClockwise,
}

/// Search the neighborhood of the pixel `center` in `mask` for a pixel with
/// a non-zero value, starting from `start` and in the order given by `dir`.
///
/// If `skip_first` is true, start the search from the next neighbor of `start`
/// in the order given by `dir`.
fn find_nonzero_neighbor(
    mask: &NdTensorView<i32, 2>,
    center: Point,
    start: Point,
    dir: Direction,
    skip_first: bool,
) -> Option<Point> {
    let neighbors = center.neighbors();
    let next_neighbor = |idx| match dir {
        Direction::Clockwise => (idx + 1) % neighbors.len(),
        Direction::CounterClockwise => {
            if idx == 0 {
                neighbors.len() - 1
            } else {
                idx - 1
            }
        }
    };

    let start_idx = neighbors
        .iter()
        .position(|&p| p == start)
        .map(|index| {
            if skip_first {
                next_neighbor(index)
            } else {
                index
            }
        })
        .unwrap();

    let mut idx = start_idx;
    loop {
        if mask[neighbors[idx].coord()] != 0 {
            return Some(neighbors[idx]);
        }
        idx = next_neighbor(idx);
        if idx == start_idx {
            break;
        }
    }

    None
}

/// Specifies which contours to extract from a mask in [find_contours].
pub enum RetrievalMode {
    /// Get only the outer-most contours.
    External,

    /// Retrieve all contours as a flat list without hierarchy information.
    List,
}

/// Find the contours of connected components in the binary image `mask`.
///
/// Returns a collection of the polygons of each component. The algorithm follows
/// the border of each component in counter-clockwise order.
///
/// This uses the algorithm from [1] (see Appendix 1), which is also the same
/// algorithm used in OpenCV's `findContours` function [2]. This function does
/// not currently implement the parts of the algorithm that discover the
/// hierarchical relationships between contours.
///
/// [1] Suzuki, Satoshi and Keiichi Abe. “Topological structural analysis of digitized binary
///     images by border following.” Comput. Vis. Graph. Image Process. 30 (1985): 32-46.
/// [2] https://docs.opencv.org/4.7.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
pub fn find_contours(mask: NdTensorView<i32, 2>, mode: RetrievalMode) -> Polygons {
    // Create a copy of the mask with zero-padding around the border. The
    // padding enables the algorithm to handle objects that touch the edge of
    // the mask.
    let padding = 1;
    let mut padded_mask = NdTensor::zeros([mask.rows() + 2 * padding, mask.cols() + 2 * padding]);
    for y in 0..mask.rows() {
        for x in 0..mask.cols() {
            // Clamp values in the copied mask to { 0, 1 } so the algorithm
            // below can use other values as part of its working.
            let value = mask[[y, x]].clamp(0, 1);
            padded_mask[[y + padding, x + padding]] = value;
        }
    }
    let mut mask = padded_mask;

    let mut contours = Polygons::new();

    // Points of current border.
    let mut border = Vec::new();

    // Sequential number of next border. Called `NBD` in the paper.
    let mut border_num = 1;

    // Value of last non-zero pixel visited on current row. See Algorithm 2 in
    // paper. This value is zero if we've not passed through any borders on the
    // current row yet, +ve if we're inside an outer border and -ve after
    // exiting the outer border.
    let mut last_nonzero_pixel;

    let outer_only = matches!(mode, RetrievalMode::External);

    for y in padding..mask.rows() - padding {
        let y = y as i32;
        last_nonzero_pixel = 0;

        for x in padding..mask.cols() - padding {
            let x = x as i32;

            let start_point = Point { y, x };
            let current = mask[start_point.coord()];
            if current == 0 {
                continue;
            }

            // Neighbor of current point to start searching for next pixel
            // along the border that begins at the current point.
            let mut start_neighbor = None;

            let prev_point = start_point.translate(0, -1);
            let next_point = start_point.translate(0, 1);

            // Test whether we are at the starting point of an unvisited border.
            if outer_only {
                if last_nonzero_pixel <= 0 && mask[prev_point.coord()] == 0 && current == 1 {
                    start_neighbor = Some(prev_point);
                }
            } else if mask[prev_point.coord()] == 0 && current == 1 {
                // This is a new outer border.
                start_neighbor = Some(prev_point);
            } else if current >= 1 && mask[next_point.coord()] == 0 {
                // This is a new hole border.
                start_neighbor = Some(next_point);
            }

            // Follow the border if we found a start point.
            if let Some(start_neighbor) = start_neighbor {
                border_num += 1;
                border.clear();

                let nonzero_start_neighbor = find_nonzero_neighbor(
                    &mask.view(),
                    start_point,
                    start_neighbor,
                    Direction::Clockwise,
                    false, // skip_first
                );

                if let Some(start_neighbor) = nonzero_start_neighbor {
                    let mut current_point = start_point;
                    let mut prev_neighbor = start_neighbor;

                    loop {
                        let next_point = find_nonzero_neighbor(
                            &mask.view(),
                            current_point,
                            prev_neighbor,
                            Direction::CounterClockwise,
                            true, // skip_first
                        );

                        // Determine if this is the right or left side of the
                        // border and set current pixel to -ve / +ve.
                        if mask[current_point.translate(0, 1).coord()] == 0 {
                            border.push(current_point);
                            mask[current_point.coord()] = -border_num;
                        } else if mask[current_point.coord()] == 1 {
                            border.push(current_point);
                            mask[current_point.coord()] = border_num;
                        }

                        if next_point == Some(start_point) && current_point == start_neighbor {
                            // We are back to the starting point of the border.
                            break;
                        }

                        // Go to the next pixel along the border.
                        prev_neighbor = current_point;
                        current_point = next_point.unwrap();
                    }
                } else {
                    // The current border consists of a single point. Mark it
                    // as being the right edge of a border.
                    border.push(start_point);
                    mask[start_point.coord()] = -border_num;
                }

                // Adjust coordinates to remove padding.
                for point in border.iter_mut() {
                    point.x -= padding as i32;
                    point.y -= padding as i32;
                }
                contours.push(&border);
            }

            last_nonzero_pixel = mask[start_point.coord()];
        }
    }

    contours
}

/// Print out elements of a 2D grid for debugging.
#[allow(dead_code)]
fn print_grid<T: Display>(grid: NdTensorView<T, 2>) {
    for y in 0..grid.rows() {
        for x in 0..grid.cols() {
            print!("{:2} ", grid[[y, x]]);
        }
        println!();
    }
    println!();
}

/// Return the bounding box containing a set of points.
///
/// Panics if the point list is empty.
pub fn bounding_box(points: &[Point]) -> Rect {
    assert!(!points.is_empty(), "Point list must be non-empty");

    let mut left = points[0].x;
    let mut top = points[0].y;
    let mut right = left + 1;
    let mut bottom = top + 1;

    for point in points {
        left = left.min(point.x);
        right = right.max(point.x + 1);
        top = top.min(point.y);
        bottom = bottom.max(point.y + 1);
    }

    Rect::from_tlbr(top, left, bottom, right)
}

// Draw the outline of a rectangle `rect` with border width `width`.
//
// The outline is drawn such that the bounding box of the outermost pixels
// will be `rect`.
pub fn stroke_rect<T: Copy>(mut mask: NdTensorViewMut<T, 2>, rect: Rect, value: T, width: u32) {
    let width = width as i32;

    // Left edge
    fill_rect(
        mask.view_mut(),
        Rect::from_tlbr(rect.top(), rect.left(), rect.bottom(), rect.left() + width),
        value,
    );

    // Top edge (minus ends)
    fill_rect(
        mask.view_mut(),
        Rect::from_tlbr(
            rect.top(),
            rect.left() + width,
            rect.top() + width,
            rect.right() - width,
        ),
        value,
    );

    // Right edge
    fill_rect(
        mask.view_mut(),
        Rect::from_tlbr(
            rect.top(),
            rect.right() - width,
            rect.bottom(),
            rect.right(),
        ),
        value,
    );

    // Bottom edge (minus ends)
    fill_rect(
        mask.view_mut(),
        Rect::from_tlbr(
            rect.bottom() - width,
            rect.left() + width,
            rect.bottom(),
            rect.right() - width,
        ),
        value,
    );
}

/// Fill all points inside `rect` with the value `value`.
pub fn fill_rect<T: Copy>(mut mask: NdTensorViewMut<T, 2>, rect: Rect, value: T) {
    for y in rect.top()..rect.bottom() {
        for x in rect.left()..rect.right() {
            mask[[y as usize, x as usize]] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::{bounding_box, fill_rect, find_contours, stroke_rect, Point, Rect, RetrievalMode};
    use crate::tensor::{MatrixLayout, NdTensor, NdTensorView, NdTensorViewMut};

    /// Return a list of the points on the border of `rect`, in counter-clockwise
    /// order starting from the top-left corner.
    ///
    /// If `omit_corners` is true, the corner points of the rect are not
    /// included.
    fn border_points(rect: Rect, omit_corners: bool) -> Vec<Point> {
        let mut points = Vec::new();

        let left_range = if omit_corners {
            rect.top() + 1..rect.bottom() - 1
        } else {
            rect.top()..rect.bottom()
        };

        // Left edge
        for y in left_range.clone() {
            points.push(Point::from_yx(y, rect.left()));
        }

        // Bottom edge
        for x in rect.left() + 1..rect.right() - 1 {
            points.push(Point::from_yx(rect.bottom() - 1, x));
        }

        // Right edge
        for y in left_range.rev() {
            points.push(Point::from_yx(y, rect.right() - 1));
        }

        // Top edge
        for x in (rect.left() + 1..rect.right() - 1).rev() {
            points.push(Point::from_yx(rect.top(), x));
        }

        points
    }

    /// Set the elements of a grid listed in `points` to `value`.
    #[allow(dead_code)]
    fn plot_points<T: Copy>(mut grid: NdTensorViewMut<T, 2>, points: &[Point], value: T) {
        for point in points {
            grid[point.coord()] = value;
        }
    }

    /// Plot the 1-based indices of points in `points` on a grid. `step` is the
    /// increment value for each plotted point.
    #[allow(dead_code)]
    fn plot_point_indices<T: std::ops::AddAssign + Copy + Default>(
        mut grid: NdTensorViewMut<T, 2>,
        points: &[Point],
        step: T,
    ) {
        let mut value = T::default();
        value += step;
        for point in points {
            grid[point.coord()] = value;
            value += step;
        }
    }

    /// Return coordinates of all points in `grid` with a non-zero value.
    fn nonzero_points<T: Default + PartialEq>(grid: NdTensorView<T, 2>) -> Vec<Point> {
        let mut points = Vec::new();
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                if grid[[y, x]] != T::default() {
                    points.push(Point::from_yx(y as i32, x as i32))
                }
            }
        }
        points
    }

    #[test]
    fn test_bounding_box() {
        let rect = Rect::from_tlbr(5, 5, 10, 10);
        let border = border_points(rect, false /* omit_corners */);
        assert_eq!(bounding_box(&border), rect);
    }

    #[test]
    fn test_find_contours_in_empty_mask() {
        struct Case {
            size: [usize; 2],
        }

        let cases = [
            Case { size: [0, 0] },
            Case { size: [1, 1] },
            Case { size: [10, 10] },
        ];

        for case in cases {
            let mask = NdTensor::zeros(case.size);
            let contours = find_contours(mask.view(), RetrievalMode::List);
            assert_eq!(contours.len(), 0);
        }
    }

    #[test]
    fn test_find_contours_single_rect() {
        struct Case {
            rect: Rect,
            value: i32,
        }

        let cases = [
            Case {
                rect: Rect::from_tlbr(5, 5, 10, 10),
                value: 1,
            },
            // Values > 1 in the mask are clamped to 1, so they don't affect
            // the contours found.
            Case {
                rect: Rect::from_tlbr(5, 5, 10, 10),
                value: 2,
            },
            // Values < 0 are clamped to 0 and ignored.
            Case {
                rect: Rect::from_tlbr(5, 5, 10, 10),
                value: -2,
            },
        ];

        for case in cases {
            let mut mask = NdTensor::zeros([20, 20]);
            fill_rect(mask.view_mut(), case.rect, case.value);

            let contours = find_contours(mask.view(), RetrievalMode::List);

            if case.value > 0 {
                assert_eq!(contours.len(), 1);
                let border = contours.iter().next().unwrap();
                assert_eq!(border, border_points(case.rect, false /* omit_corners */));
            } else {
                assert!(contours.is_empty());
            }
        }
    }

    #[test]
    fn test_find_contours_rect_touching_frame() {
        let mut mask = NdTensor::zeros([5, 5]);
        let rect = Rect::from_tlbr(0, 0, 5, 5);
        fill_rect(mask.view_mut(), rect, 1);

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, border_points(rect, false /* omit_corners */));
    }

    #[test]
    fn test_find_contours_hollow_rect() {
        let mut mask = NdTensor::zeros([20, 20]);
        let rect = Rect::from_tlbr(5, 5, 12, 12);
        stroke_rect(mask.view_mut(), rect, 1, 2);

        let contours = find_contours(mask.view(), RetrievalMode::List);

        // There should be two contours: one for the outer border of the rect
        // and one for the inner "hole" border.
        assert_eq!(contours.len(), 2);

        // Check outer border.
        let mut contours_iter = contours.iter();
        let outer_border = contours_iter.next().unwrap();
        let inner_border = contours_iter.next().unwrap();
        assert_eq!(outer_border, border_points(rect, false /* omit_corners */));

        // Check hole border.
        let inner_rect = rect.adjust_tlbr(1, 1, -1, -1);
        let mut expected_inner_border = border_points(inner_rect, true /* omit_corners */);

        // Due to the way the algorithm works, hole border points are returned
        // in the opposite order (clockwise instead of counter-clockwise) to
        // outer border points, and the start position is shifted by one.
        expected_inner_border.reverse(); // CCW => CW
        expected_inner_border.rotate_right(1);

        assert_eq!(inner_border, expected_inner_border);
    }

    #[test]
    fn test_find_contours_external() {
        let mut mask = NdTensor::zeros([20, 20]);
        let rect = Rect::from_tlbr(5, 5, 12, 12);
        stroke_rect(mask.view_mut(), rect, 1, 2);

        let contours = find_contours(mask.view(), RetrievalMode::External);

        // There should only be one, outermost contour.
        assert_eq!(contours.len(), 1);
        let outer_border = contours.iter().next().unwrap();
        assert_eq!(outer_border, border_points(rect, false /* omit_corners */));
    }

    #[test]
    fn test_find_contours_single_point() {
        let mut mask = NdTensor::zeros([20, 20]);
        mask[[5, 5]] = 1;

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, [Point::from_yx(5, 5)]);
    }

    #[test]
    fn test_find_contours_many_rects() {
        let mut mask = NdTensor::zeros([20, 20]);

        let rects = [
            Rect::from_tlbr(5, 5, 10, 10),
            Rect::from_tlbr(15, 15, 18, 18),
        ];
        for rect in rects {
            fill_rect(mask.view_mut(), rect, 1);
        }

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in zip(contours.iter(), rects.iter()) {
            assert_eq!(border, border_points(*rect, false /* omit_corners */));
        }
    }

    #[test]
    fn test_find_contours_nested_rects() {
        let mut mask = NdTensor::zeros([15, 15]);

        let rects = [Rect::from_tlbr(5, 5, 11, 11), Rect::from_tlbr(7, 7, 9, 9)];
        for rect in rects {
            stroke_rect(mask.view_mut(), rect, 1, 1);
        }

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in zip(contours.iter(), rects.iter()) {
            assert_eq!(border, border_points(*rect, false /* omit_corners */));
        }
    }

    #[test]
    fn test_rect_clamp() {
        struct Case {
            rect: Rect,
            boundary: Rect,
            expected: Rect,
        }

        let cases = [
            Case {
                rect: Rect::from_tlbr(-5, -10, 100, 200),
                boundary: Rect::from_tlbr(0, 0, 50, 100),
                expected: Rect::from_tlbr(0, 0, 50, 100),
            },
            Case {
                rect: Rect::from_tlbr(5, 10, 40, 80),
                boundary: Rect::from_tlbr(0, 0, 50, 100),
                expected: Rect::from_tlbr(5, 10, 40, 80),
            },
        ];

        for case in cases {
            assert_eq!(case.rect.clamp(case.boundary), case.expected);
        }
    }

    #[test]
    fn test_stroke_rect() {
        let mut mask = NdTensor::zeros([10, 10]);
        let rect = Rect::from_tlbr(4, 4, 9, 9);

        stroke_rect(mask.view_mut(), rect, 1, 1);
        let points = nonzero_points(mask.view());

        assert_eq!(bounding_box(&points), rect);
    }
}
