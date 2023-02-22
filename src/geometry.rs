///! Geometry functions for pre and post-processing images.
///!
///! TODO: Move these out of Wasnn and into a separate crate.
use std::ops::Range;
use std::slice::Iter;

use crate::tensor::{MatrixLayout, NdTensor, NdTensorView};

pub type Coord = i32;

/// A point defined by integer X and Y coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
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

    /// Return an iterator over individual polygons in the sequence.
    pub fn iter(&self) -> PolygonsIter {
        PolygonsIter {
            points: &self.points,
            polygons: self.polygons.iter(),
        }
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
pub fn find_contours(mask: NdTensorView<i32, 2>) -> Polygons {
    // Create a copy of the mask with zero-padding around the border. The
    // padding enables the algorithm to handle objects that touch the edge of
    // the mask.
    let padding = 1;
    let mut padded_mask = NdTensor::zeros([mask.rows() + 2 * padding, mask.cols() + 2 * padding]);
    for y in 0..mask.rows() {
        for x in 0..mask.cols() {
            padded_mask[[y + padding, x + padding]] = mask[[y, x]];
        }
    }
    let mut mask = padded_mask;

    let mut contours = Polygons::new();

    // Points of current border.
    let mut border = Vec::new();

    // Sequential number of next border. Called `NBD` in the paper.
    let mut border_num = 1;

    for y in padding..mask.rows() - padding {
        let y = y as i32;

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

            // Test whether we are at the starting point of an unvisited border.
            if mask[start_point.translate(0, -1).coord()] == 0 && current == 1 {
                // This is a new outer border.
                border_num += 1;
                start_neighbor = Some(Point { y, x: x - 1 });
            } else if current >= 1 && mask[start_point.translate(0, 1).coord()] == 0 {
                // This is a new hole border.
                border_num += 1;
                start_neighbor = Some(Point { y, x: x + 1 });
            }

            // Follow the border if we found a start point.
            if let Some(start_neighbor) = start_neighbor {
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
        }
    }

    contours
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

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::{bounding_box, find_contours, Point, Rect};
    use crate::tensor::{NdTensor, NdTensorViewMut};

    /// Fill all points inside `rect` with the value `value`.
    fn fill_rect(mut mask: NdTensorViewMut<i32, 2>, rect: Rect, value: i32) {
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                mask[[y as usize, x as usize]] = value;
            }
        }
    }

    // Draw the outline of a rectangle `rect` with border width `width`.
    fn stroke_rect(mut mask: NdTensorViewMut<i32, 2>, rect: Rect, value: i32, width: u32) {
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

    /// Return a list of the points on the border of `rect`, in clockwise
    /// order starting from the top-left corner.
    fn border_points(rect: Rect) -> Vec<Point> {
        let mut points = Vec::new();

        // Left edge
        for y in rect.top()..rect.bottom() {
            points.push(Point::from_yx(y, rect.left()));
        }

        // Bottom edge
        for x in rect.left() + 1..rect.right() {
            points.push(Point::from_yx(rect.bottom() - 1, x));
        }

        // Right edge
        for y in (rect.top()..rect.bottom() - 1).rev() {
            points.push(Point::from_yx(y, rect.right() - 1));
        }

        // Top edge
        for x in (rect.left() + 1..rect.right() - 1).rev() {
            points.push(Point::from_yx(rect.top(), x));
        }

        points
    }

    #[test]
    fn test_bounding_box() {
        let rect = Rect::from_tlbr(5, 5, 10, 10);
        let border = border_points(rect);
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
            let contours = find_contours(mask.view());
            assert_eq!(contours.len(), 0);
        }
    }

    #[test]
    fn test_find_contours_single_rect() {
        let mut mask = NdTensor::zeros([20, 20]);
        let rect = Rect::from_tlbr(5, 5, 10, 10);
        fill_rect(mask.view_mut(), rect, 1);

        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, border_points(rect));
    }

    #[test]
    fn test_find_contours_rect_touching_frame() {
        let mut mask = NdTensor::zeros([5, 5]);
        let rect = Rect::from_tlbr(0, 0, 5, 5);
        fill_rect(mask.view_mut(), rect, 1);

        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, border_points(rect));
    }

    #[test]
    fn test_find_contours_hollow_rect() {
        let mut mask = NdTensor::zeros([20, 20]);
        let rect = Rect::from_tlbr(5, 5, 10, 10);
        stroke_rect(mask.view_mut(), rect, 1, 3);

        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, border_points(rect));
    }

    #[test]
    fn test_find_contours_single_point() {
        let mut mask = NdTensor::zeros([20, 20]);
        mask[[5, 5]] = 1;

        let contours = find_contours(mask.view());
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

        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in zip(contours.iter(), rects.iter()) {
            assert_eq!(border, border_points(*rect));
        }
    }

    #[test]
    fn test_find_contours_nested_rects() {
        let mut mask = NdTensor::zeros([15, 15]);

        let rects = [Rect::from_tlbr(5, 5, 11, 11), Rect::from_tlbr(7, 7, 9, 9)];
        for rect in rects {
            stroke_rect(mask.view_mut(), rect, 1, 1);
        }

        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in zip(contours.iter(), rects.iter()) {
            assert_eq!(border, border_points(*rect));
        }
    }
}
