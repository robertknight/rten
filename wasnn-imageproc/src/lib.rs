//! Functions for pre and post-processing images.

use std::fmt::Display;
use std::iter::zip;

use wasnn_tensor::{MatrixLayout, NdTensor, NdTensorView, NdTensorViewMut};

mod math;
mod shapes;

pub use math::Vec2;
pub use shapes::{bounding_rect, BoundingRect, Line, Point, Polygon, Polygons, Rect, RotatedRect};

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
/// This uses the algorithm from [^1] (see Appendix 1), which is also the same
/// algorithm used in OpenCV's `findContours` function [^2]. This function does
/// not currently implement the parts of the algorithm that discover the
/// hierarchical relationships between contours.
///
/// [^1]: Suzuki, Satoshi and Keiichi Abe. “Topological structural analysis of digitized binary
///       images by border following.” Comput. Vis. Graph. Image Process. 30 (1985): 32-46.
/// [^2]: <https://docs.opencv.org/4.7.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0>
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

fn simplify_polyline_internal(
    points: &[Point],
    epsilon: f32,
    out_points: &mut Vec<Point>,
    keep_last: bool,
) {
    if points.len() <= 1 {
        if let Some(&point) = points.first() {
            out_points.push(point);
        }
        return;
    }

    // Find point furthest from the line segment through the first and last
    // points.
    let line_segment = Line::from_endpoints(*points.first().unwrap(), *points.last().unwrap());
    let inner_points = &points[1..points.len() - 1];
    let (max_index, max_dist) =
        inner_points
            .iter()
            .enumerate()
            .fold((0, 0.), |(max_i, max_dist), (i, &point)| {
                let dist = line_segment.distance(point);
                if dist >= max_dist {
                    (i + 1, dist)
                } else {
                    (max_i, max_dist)
                }
            });

    if max_dist > epsilon {
        // Recursively simplify polyline segments before and after pivot.
        simplify_polyline_internal(
            &points[..max_index + 1],
            epsilon,
            out_points,
            false, /* keep_last */
        );
        simplify_polyline_internal(&points[max_index..], epsilon, out_points, keep_last);
    } else {
        // Simplify current polyline to start and end points.
        out_points.push(line_segment.start);
        if keep_last {
            out_points.push(line_segment.end);
        }
    }
}

/// Return a simplified version of the polyline defined by `points`.
///
/// The result will be a subset of points from the input, which always includes
/// the first and last points.
///
/// `epsilon` specifies the maximum distance that any removed point may be from
/// the closest point on the simplified polygon.
///
/// This uses the Douglas-Peucker algorithm [^1].
///
/// [^1]: <https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm>
pub fn simplify_polyline(points: &[Point], epsilon: f32) -> Vec<Point> {
    assert!(epsilon >= 0.);
    let mut result = Vec::new();
    simplify_polyline_internal(points, epsilon, &mut result, true /* keep_last */);
    result
}

/// Return a simplified version of the polygon defined by `points`.
///
/// This is very similar to [simplify_polyline] except that the input is
/// treated as a polygon where the last point implicitly connects to the first
/// point to close the shape.
pub fn simplify_polygon(points: &[Point], epsilon: f32) -> Vec<Point> {
    // Convert polygon to polyline.
    let mut polyline = points.to_vec();
    polyline.push(points[0]);

    // Simplify and convert polyline back to polygon.
    let mut simplified = simplify_polyline(&polyline, epsilon);
    simplified.truncate(simplified.len() - 1);

    simplified
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

/// Return a copy of `p` with X and Y coordinates clamped to `[0, width)` and
/// `[0, height)` respectively.
fn clamp_to_bounds(p: Point, height: i32, width: i32) -> Point {
    Point::from_yx(
        p.y.clamp(0, height.saturating_sub(1).max(0)),
        p.x.clamp(0, width.saturating_sub(1).max(0)),
    )
}

/// Iterator over points that lie on a line, as determined by the Bresham
/// algorithm.
///
/// The implementation in [Pillow](https://pillow.readthedocs.io/en/stable/) was
/// used as a reference.
struct BreshamPoints {
    /// Next point to return
    current: Point,

    /// Remaining points to return
    remaining_steps: u32,

    /// Twice total change in X along line
    dx: i32,

    /// Twice total change in Y along line
    dy: i32,

    /// Tracks error between integer points yielded by this iterator and the
    /// "true" coordinate.
    error: i32,

    /// Increment to X coordinate of `current`.
    x_step: i32,

    /// Increment to Y coordinate of `current`.
    y_step: i32,
}

impl BreshamPoints {
    fn new(l: Line) -> BreshamPoints {
        let dx = (l.end.x - l.start.x).abs();
        let dy = (l.end.y - l.start.y).abs();

        BreshamPoints {
            current: l.start,
            remaining_steps: dx.max(dy) as u32,

            // dx and dy are doubled here as it makes stepping simpler.
            dx: dx * 2,
            dy: dy * 2,

            error: if dx >= dy { dy * 2 - dx } else { dx * 2 - dy },
            x_step: (l.end.x - l.start.x).signum(),
            y_step: (l.end.y - l.start.y).signum(),
        }
    }
}

impl Iterator for BreshamPoints {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        if self.remaining_steps == 0 {
            return None;
        }

        let current = self.current;
        self.remaining_steps -= 1;

        if self.x_step == 0 {
            // Vertical line
            self.current.y += self.y_step;
        } else if self.y_step == 0 {
            // Horizontal line
            self.current.x += self.x_step;
        } else if self.dx >= self.dy {
            // X-major line (width >= height). Advances X on each step and
            // advances Y on some steps.
            if self.error >= 0 {
                self.current.y += self.y_step;
                self.error -= self.dx;
            }
            self.error += self.dy;
            self.current.x += self.x_step;
        } else {
            // Y-major line (height > width). Advances Y on each step and
            // advances X on some steps.
            if self.error >= 0 {
                self.current.x += self.x_step;
                self.error -= self.dy
            }
            self.error += self.dx;
            self.current.y += self.y_step;
        }

        Some(current)
    }
}

/// Draw a non-antialiased line in an image.
pub fn draw_line<T: Copy>(mut image: NdTensorViewMut<T, 2>, line: Line, value: T) {
    // This function uses Bresham's line algorithm, with the implementation
    // in Pillow (https://pillow.readthedocs.io/en/stable/) used as a reference.
    let height: i32 = image.rows().try_into().unwrap();
    let width: i32 = image.cols().try_into().unwrap();

    let start = clamp_to_bounds(line.start, height, width);
    let end = clamp_to_bounds(line.end, height, width);
    let clamped = Line::from_endpoints(start, end);

    for p in BreshamPoints::new(clamped) {
        image[p.coord()] = value;
    }
}

/// Draw the outline of a non anti-aliased polygon in an image.
pub fn draw_polygon<T: Copy>(mut image: NdTensorViewMut<T, 2>, poly: &[Point], value: T) {
    for edge in Polygon::new(poly).edges() {
        draw_line(image.view_mut(), edge, value);
    }
}

/// Return the sorted subset of points from `poly` that form a convex hull
/// containing `poly`.
pub fn convex_hull(poly: &[Point]) -> Vec<Point> {
    // See https://en.wikipedia.org/wiki/Graham_scan

    let mut hull = Vec::new();

    // Find lowest and left-most point.
    let min_point = match poly.iter().min_by_key(|p| (-p.y, p.x)) {
        Some(p) => p,
        None => {
            return hull;
        }
    };

    // FIXME - Should `min_point` be removed from the list? It leads to NaN
    // outputs from `angle` when angle is called with `p == min_point`.
    //
    // TODO - Break ties if multiple points form the same angle, by preferring
    // the furthest point.

    // Compute cosine of angle between the vector `p - min_point` and the X axis.
    let angle = |p: Point| {
        let dy = p.y - min_point.y;
        let dx = p.x - min_point.x;
        let x_axis = Vec2::from_yx(0., 1.);
        Vec2::from_yx(dy as f32, dx as f32).normalized().dot(x_axis)
    };

    // Sort points by angle between `point - min_point` and X axis.
    let mut sorted_points = poly.to_vec();
    sorted_points.sort_by(|&a, &b| f32::total_cmp(&angle(a), &angle(b)));

    // Visit sorted points and keep the sequence that can be followed without
    // making any clockwise turns.
    for &p in sorted_points.iter() {
        while hull.len() >= 2 {
            let [prev2, prev] = [hull[hull.len() - 2], hull[hull.len() - 1]];
            let ac = Vec2::from_points(prev2, p);
            let bc = Vec2::from_points(prev, p);
            let turn_dir = ac.cross_product_norm(bc);
            if turn_dir > 0. {
                // Last three points form a counter-clockwise turn.
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }

    hull
}

/// Return the rotated rectangle with minimum area which contains `points`.
///
/// Returns `None` if `points` contains fewer than 2 points.
pub fn min_area_rect(points: &[Point]) -> Option<RotatedRect> {
    // See "Exhaustive Search Algorithm" in
    // https://www.geometrictools.com/Documentation/MinimumAreaRectangle.pdf.

    let hull = convex_hull(points);

    // Iterate over each edge of the polygon and find the smallest bounding
    // rect where one of the rect's edges aligns with the polygon edge. Keep
    // the rect that has the smallest area over all edges.
    let mut min_rect: Option<RotatedRect> = None;
    for (&edge_start, &edge_end) in zip(hull.iter(), hull.iter().cycle().skip(1)) {
        // Project polygon points onto axes that are parallel and perpendicular
        // to the current edge. The maximum distance between the projected
        // points gives the width and height of the bounding rect.

        let par_axis = Vec2::from_points(edge_start, edge_end).normalized();

        // nb. Perpendicular axis points into the hull.
        let perp_axis = -par_axis.perpendicular();

        let (min_par, max_par, max_perp): (f32, f32, f32) = hull.iter().fold(
            (f32::MAX, f32::MIN, f32::MIN),
            |(min_par, max_par, max_perp), point| {
                let d = Vec2::from_points(edge_start, *point);
                let par_proj = par_axis.dot(d);
                let perp_proj = perp_axis.dot(d);
                (
                    min_par.min(par_proj),
                    max_par.max(par_proj),
                    max_perp.max(perp_proj),
                )
            },
        );

        let height = max_perp;
        let width = max_par - min_par;
        let area = height * width;

        if area < min_rect.map(|r| r.area()).unwrap_or(f32::MAX) {
            let center = Vec2::from_yx(edge_start.y as f32, edge_start.x as f32)
                + (par_axis * ((min_par + max_par) / 2.))
                + (perp_axis * (height / 2.));
            min_rect = Some(RotatedRect::new(
                center, /* up_axis */ perp_axis, width, height,
            ))
        }
    }

    min_rect
}

/// Tracks data about an edge in a polygon being traversed by [FillIter].
#[derive(Clone, Copy, Debug)]
struct Edge {
    /// Y coordinate where this edge starts
    start_y: i32,

    /// Number of scanlines remaining for this edge
    y_steps: u32,

    /// X coordinate where this edge intersects the current scanline
    x: i32,

    /// Error term indicating difference between true X coordinate for current
    /// scanline and `x`.
    error: i32,

    /// Amount to increment `error` for every scanline.
    error_incr: i32,

    /// Amount to decrement `error` when it becomes positive.
    error_decr: i32,

    /// Amount to increment `x` for every scanline.
    x_step: i32,

    /// Amount to increment `x` when `error` becomes positive.
    extra_x_step: i32,
}

/// Iterator over coordinates of pixels that fill a polygon. See
/// [Polygon::fill_iter] for notes on how this iterator determines which
/// pixels are inside the polygon.
///
/// The implementation follows <https://www.jagregory.com/abrash-black-book/#filling-arbitrary-polygons>.
pub struct FillIter {
    /// Edges in the polygon, sorted by Y coordinate.
    edges: Vec<Edge>,

    /// Edges in the polygon which intersect the horizontal line at `cursor.y`.
    ///
    /// Sorted by X coordinate.
    active_edges: Vec<Edge>,

    /// Bounding rect that contains the polygon.
    bounds: Rect,

    /// Coordinates of next pixel to return.
    cursor: Point,
}

impl FillIter {
    fn new(poly: Polygon<&[Point]>) -> FillIter {
        let mut edges: Vec<_> = poly
            .edges()
            // Ignore horizontal edges
            .filter(|e| e.start.y != e.end.y)
            .map(|e| {
                // Normalize edge so that `delta_y` is +ve
                let (start, end) = if e.start.y <= e.end.y {
                    (e.start, e.end)
                } else {
                    (e.end, e.start)
                };

                let delta_x = end.x - start.x;
                let delta_y = end.y - start.y;

                Edge {
                    start_y: start.y,
                    y_steps: delta_y as u32,

                    x: start.x,

                    // `x_step` is the integer part of `1/slope`.
                    x_step: delta_x / delta_y,

                    // The error term tracks when `x` needs an adjustment due
                    // to accumulation of the fractional part of `1/slope`.
                    error: if delta_x >= 0 {
                        0
                    } else {
                        // TODO - Clarify where this comes from.
                        -delta_y + 1
                    },
                    error_incr: delta_x.abs() % delta_y,
                    error_decr: delta_y,
                    extra_x_step: delta_x.signum(),
                }
            })
            .collect();
        edges.sort_by_key(|e| -e.start_y);

        let active_edges = Vec::with_capacity(edges.len());

        let bounds = poly.bounding_rect();
        let mut iter = FillIter {
            edges,
            active_edges,
            bounds,
            cursor: if bounds.is_empty() {
                // If the polygon is empty, the cursor starts at the bottom right
                // so that the iterator immediately yields `None`, rather than
                // having to loop over all the empty rows.
                bounds.bottom_right()
            } else {
                bounds.top_left()
            },
        };
        iter.update_active_edges();

        iter
    }

    /// Update the `active_edges` list after moving to a new line.
    fn update_active_edges(&mut self) {
        // Remove entries that end at this line and update X coord of other entries.
        self.active_edges.retain_mut(|mut e| {
            e.y_steps -= 1;
            if e.y_steps > 0 {
                // Advance X coordinate for current line and error term that
                // tracks difference between `e.x` and true X coord.
                e.x += e.x_step;
                e.error += e.error_incr;
                if e.error > 0 {
                    e.error -= e.error_decr;
                    e.x += e.extra_x_step;
                }
                true
            } else {
                false
            }
        });

        // Add edges that begin at this line.
        while let Some(edge) = self.edges.last().copied() {
            if edge.start_y > self.cursor.y {
                // `self.edges` is sorted on Y coordinate, so remaining entries
                // start on lines with higher Y coordinate than cursor.
                break;
            }
            self.edges.pop();
            self.active_edges.push(edge);
        }

        // Sort edges by X coordinate of intersection with scanline. We only
        // need to sort by `e.x`, but including other elements in the sort key
        // provides more predictable ordering for debugging.
        self.active_edges
            .sort_by_key(|e| (e.x, e.x_step, e.extra_x_step));
    }
}

impl Iterator for FillIter {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
        while !self.active_edges.is_empty() {
            let current = self.cursor;
            let intersections =
                self.active_edges
                    .iter()
                    .fold(0, |i, e| if e.x <= current.x { i + 1 } else { i });

            self.cursor.move_by(0, 1);
            if self.cursor.x == self.bounds.right() {
                self.cursor.move_to(current.y + 1, self.bounds.left());
                self.update_active_edges();
            }

            if intersections % 2 == 1 {
                return Some(current);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use wasnn_tensor::{Layout, MatrixLayout, NdTensor, NdTensorView, NdTensorViewMut};

    use super::{
        convex_hull, draw_polygon, fill_rect, find_contours, min_area_rect, print_grid,
        simplify_polygon, simplify_polyline, stroke_rect, BoundingRect, Point, Polygon, Rect,
        RetrievalMode,
    };

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

    /// Create a 2D NdTensor from an MxN nested array.
    fn image_from_2d_array<const M: usize, const N: usize>(xs: [[i32; N]; M]) -> NdTensor<i32, 2> {
        let mut image = NdTensor::zeros([M, N]);
        for y in 0..M {
            for x in 0..N {
                image[[y, x]] = xs[y][x];
            }
        }
        image
    }

    /// Compare two single-channel images with i32 pixel values.
    fn compare_images(a: NdTensorView<i32, 2>, b: NdTensorView<i32, 2>) {
        assert_eq!(a.rows(), b.rows());
        assert_eq!(a.cols(), b.cols());

        for y in 0..a.rows() {
            for x in 0..a.cols() {
                if a[[y, x]] != b[[y, x]] {
                    print_grid(a);
                    panic!("mismatch at coord [{}, {}]", y, x);
                }
            }
        }
    }

    /// Convert a slice of `[y, x]` coordinates to `Point`s
    pub fn points_from_coords(coords: &[[i32; 2]]) -> Vec<Point> {
        coords.iter().map(|[y, x]| Point::from_yx(*y, *x)).collect()
    }

    /// Convery an array of `[y, x]` coordinates to `Point`s
    pub fn points_from_n_coords<const N: usize>(coords: [[i32; 2]; N]) -> [Point; N] {
        coords.map(|[y, x]| Point::from_yx(y, x))
    }

    #[test]
    fn test_convex_hull() {
        struct Case {
            points: &'static [[i32; 2]],
            hull: &'static [[i32; 2]],
        }

        let cases = [
            // Empty polygon
            Case {
                points: &[],
                hull: &[],
            },
            // Simple square. The hull is a re-ordering of the input.
            Case {
                points: &[[0, 0], [0, 4], [4, 4], [4, 0]],
                hull: &[[4, 0], [0, 0], [0, 4], [4, 4]],
            },
            // Square with an indent on each edge. The hull is just a rect.
            Case {
                points: &[
                    // Top
                    [0, 0],
                    [1, 2],
                    [0, 4],
                    // Right
                    [2, 3],
                    [4, 4],
                    // Bottom
                    [3, 2],
                    [4, 0],
                    // Left
                    [2, 1],
                ],

                // Hull starts with lowest, left-most corner then proceeds
                // clockwise.
                hull: &[[4, 0], [0, 0], [0, 4], [4, 4]],
            },
        ];

        for case in cases {
            let points = points_from_coords(case.points);
            let expected_hull = points_from_coords(case.hull);

            let hull = convex_hull(&points);

            assert_eq!(hull, expected_hull);
        }
    }

    #[test]
    fn test_draw_polygon() {
        struct Case {
            points: &'static [[i32; 2]],
            expected: NdTensor<i32, 2>,
        }

        let cases = [
            // A simple rect: Straight lines in each direction
            Case {
                points: &[[0, 0], [0, 4], [4, 4], [4, 0]],
                expected: image_from_2d_array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ]),
            },
            // Slopes in each direction.
            Case {
                points: &[[0, 2], [2, 0], [4, 2], [2, 4]],
                expected: image_from_2d_array([
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                ]),
            },
            // Steep slopes in each direction.
            Case {
                points: &[[0, 2], [2, 1], [4, 2], [2, 3]],
                expected: image_from_2d_array([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ]),
            },
        ];

        for case in cases {
            let points: Vec<_> = case
                .points
                .iter()
                .map(|[y, x]| Point::from_yx(*y, *x))
                .collect();

            let mut image = NdTensor::zeros(case.expected.shape());
            draw_polygon(image.view_mut(), &points, 1);
            compare_images(image.view(), case.expected.view());
        }
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
    fn test_min_area_rect() {
        struct Case {
            points: Vec<Point>,
            expected: [Point; 4],
        }

        let cases = [
            // Axis-aligned rect
            Case {
                points: points_from_coords(&[[0, 0], [0, 4], [4, 4], [4, 0]]),
                expected: points_from_n_coords([[4, 0], [0, 0], [0, 4], [4, 4]]),
            },
        ];

        for case in cases {
            let min_rect = min_area_rect(&case.points).unwrap();
            assert_eq!(min_rect.corners(), case.expected);
        }
    }

    #[test]
    fn test_stroke_rect() {
        let mut mask = NdTensor::zeros([10, 10]);
        let rect = Rect::from_tlbr(4, 4, 9, 9);

        stroke_rect(mask.view_mut(), rect, 1, 1);
        let points = nonzero_points(mask.view());

        assert_eq!(
            Polygon::new(&points).bounding_rect(),
            rect.adjust_tlbr(0, 0, -1, -1)
        );
    }

    #[test]
    fn test_simplify_polyline() {
        struct Case {
            poly: Vec<Point>,
            epsilon: f32,
            simplified: Vec<Point>,
        }

        let cases = [
            // Single point
            Case {
                poly: vec![Point::from_yx(0, 0)],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(0, 0)],
            },
            // Line of 2 points
            Case {
                poly: vec![Point::from_yx(5, 2), Point::from_yx(3, 5)],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5, 2), Point::from_yx(3, 5)],
            },
            // Line of 3 points
            Case {
                poly: vec![
                    Point::from_yx(5, 2),
                    Point::from_yx(5, 3),
                    Point::from_yx(5, 4),
                ],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5, 2), Point::from_yx(5, 4)],
            },
            // Line of 4 points
            Case {
                poly: vec![
                    Point::from_yx(5, 2),
                    Point::from_yx(5, 3),
                    Point::from_yx(5, 4),
                    Point::from_yx(5, 5),
                ],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5, 2), Point::from_yx(5, 5)],
            },
            // Boundary points of rect
            Case {
                poly: border_points(Rect::from_tlbr(4, 4, 9, 9), false /* omit_corners */),
                epsilon: 0.1,
                simplified: [[4, 4], [8, 4], [8, 8], [4, 8], [4, 5]]
                    .map(|[y, x]| Point::from_yx(y, x))
                    .into_iter()
                    .collect(),
            },
        ];

        for case in cases {
            let simplified = simplify_polyline(&case.poly, case.epsilon);
            assert_eq!(&simplified, &case.simplified);
        }
    }

    #[test]
    fn test_simplify_polygon() {
        struct Case {
            poly: Vec<Point>,
            epsilon: f32,
            simplified: Vec<Point>,
        }

        // Since `simplify_polygon` is a thin wrapper around `simplify_polyline`,
        // so we only have a few cases to cover the differences here.
        let cases = [Case {
            poly: border_points(Rect::from_tlbr(4, 4, 9, 9), false /* omit_corners */),
            epsilon: 0.1,
            simplified: [[4, 4], [8, 4], [8, 8], [4, 8]]
                .map(|[y, x]| Point::from_yx(y, x))
                .into_iter()
                .collect(),
        }];

        for case in cases {
            let simplified = simplify_polygon(&case.poly, case.epsilon);
            assert_eq!(&simplified, &case.simplified);
        }
    }
}
