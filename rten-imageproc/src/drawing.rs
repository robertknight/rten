use crate::{BoundingRect, Line, Point, Polygon, Rect, RotatedRect, Vec2};

use rten_tensor::{MatrixLayout, NdTensorViewMut};

/// Return a copy of `p` with X and Y coordinates clamped to `[0, width)` and
/// `[0, height)` respectively.
fn clamp_to_bounds(p: Point, height: i32, width: i32) -> Point {
    Point::from_yx(
        p.y.clamp(0, height.saturating_sub(1).max(0)),
        p.x.clamp(0, width.saturating_sub(1).max(0)),
    )
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
pub fn draw_line<T: Copy>(mut image: NdTensorViewMut<T, 2>, line: Line, value: T, width: u32) {
    if width == 0 {
        return;
    }

    if width == 1 {
        // This function uses Bresham's line algorithm, with the implementation
        // in Pillow (https://pillow.readthedocs.io/en/stable/) used as a reference.
        let img_height: i32 = image.rows().try_into().unwrap();
        let img_width: i32 = image.cols().try_into().unwrap();

        let start = clamp_to_bounds(line.start, img_height, img_width);
        let end = clamp_to_bounds(line.end, img_height, img_width);
        let clamped = Line::from_endpoints(start, end);
        for p in BreshamPoints::new(clamped) {
            image[p.coord()] = value;
        }
    } else {
        // Convert this wide line into a polygon and fill it.
        let line = line.to_f32();
        let line_vec = Vec2::from_xy(line.width(), line.height());
        let rrect = RotatedRect::new(
            line.center(),
            line_vec.perpendicular(),
            line_vec.length(),
            width as f32,
        );

        let corners: [Point<i32>; 4] = rrect
            .corners()
            .map(|c| Point::from_yx(c.y as i32, c.x as i32));

        for p in Polygon::new(corners).fill_iter() {
            if let Some(img_val) = image.get_mut(p.coord()) {
                *img_val = value;
            }
        }
    }
}

/// Draw the outline of a non anti-aliased polygon in an image.
pub fn draw_polygon<T: Copy>(
    mut image: NdTensorViewMut<T, 2>,
    poly: &[Point],
    value: T,
    width: u32,
) {
    for edge in Polygon::new(poly).edges() {
        draw_line(image.view_mut(), edge, value, width);
    }
}

/// Tracks data about an edge in a polygon being traversed by [`FillIter`].
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
/// [`Polygon::fill_iter`] for notes on how this iterator determines which
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
    pub(crate) fn new(poly: Polygon<i32, &[Point]>) -> FillIter {
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
        self.active_edges.retain_mut(|e| {
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

pub type Rgb<T = u8> = [T; 3];

/// Drawing style properties.
#[derive(Copy, Clone)]
struct PainterState<T> {
    stroke: Rgb<T>,
    stroke_width: u32,
}

/// Painter is a context for drawing into an image tensor.
///
/// Painter uses a mutable tensor with `[channel, height, width]` dimensions as
/// the drawing surface. It has a current drawing state with properties such as
/// stroke color and width and allows states to be saved to and restored from
/// a stack.
///
/// Drawing currently operates in the first 3 channels of the surface, which are
/// assumed to represent red, green and blue colors.
pub struct Painter<'a, T> {
    /// CHW image tensor.
    surface: NdTensorViewMut<'a, T, 3>,
    saved_states: Vec<PainterState<T>>,
    state: PainterState<T>,
}

impl<'a, T: Copy + Default> Painter<'a, T> {
    /// Create a Painter which draws into the CHW tensor `surface`.
    pub fn new(surface: NdTensorViewMut<'a, T, 3>) -> Painter<'a, T> {
        Painter {
            surface,
            state: PainterState {
                stroke: [T::default(); 3],
                stroke_width: 1,
            },
            saved_states: Vec::new(),
        }
    }

    /// Save the current drawing style on a stack.
    pub fn save(&mut self) {
        self.saved_states.push(self.state);
    }

    /// Pop and apply a drawing style from the stack created with [`Painter::save`].
    pub fn restore(&mut self) {
        if let Some(state) = self.saved_states.pop() {
            self.state = state;
        }
    }

    /// Save the current drawing style, run `f(self)` and then restore the saved
    /// style.
    ///
    /// This avoids the need to manually save and restore state with
    /// [`Painter::save`] and [`Painter::restore`].
    pub fn with_save<F: Fn(&mut Self)>(&mut self, f: F) {
        self.save();
        f(self);
        self.restore();
    }

    /// Set the RGB color values used by the `draw_*` methods.
    pub fn set_stroke(&mut self, stroke: Rgb<T>) {
        self.state.stroke = stroke;
    }

    pub fn set_stroke_width(&mut self, width: u32) {
        self.state.stroke_width = width;
    }

    /// Draw a polygon into the surface.
    pub fn draw_polygon(&mut self, points: &[Point]) {
        for i in 0..3 {
            draw_polygon(
                self.surface.slice_mut([i]),
                points,
                self.state.stroke[i],
                self.state.stroke_width,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{MatrixLayout, NdTensor, NdTensorView};

    use crate::tests::print_grid;
    use crate::{BoundingRect, Painter, Point, Polygon, Rect};

    use super::{draw_polygon, stroke_rect};

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
            draw_polygon(image.view_mut(), &points, 1, 1 /* width */);
            compare_images(image.view(), case.expected.view());
        }
    }

    #[test]
    fn test_painter_draw_polygon() {
        let [width, height] = [6, 6];
        let mut img = NdTensor::zeros([3, height, width]);
        let mut painter = Painter::new(img.view_mut());
        let [r, g, b] = [255, 100, 50];
        painter.set_stroke([r, g, b]);

        painter.draw_polygon(&Rect::from_tlbr(2, 2, 5, 5).corners());

        let expected_r = image_from_2d_array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, r, r, r, r],
            [0, 0, r, 0, 0, r],
            [0, 0, r, 0, 0, r],
            [0, 0, r, r, r, r],
        ]);
        let expected_g = expected_r.map(|&x| if x == r { g } else { 0 });
        let expected_b = expected_r.map(|&x| if x == r { b } else { 0 });

        compare_images(img.slice([0]), expected_r.view());
        compare_images(img.slice([1]), expected_g.view());
        compare_images(img.slice([2]), expected_b.view());
    }

    #[test]
    fn test_painter_save_restore() {
        let [width, height] = [6, 6];
        let mut img = NdTensor::zeros([3, height, width]);
        let mut painter = Painter::new(img.view_mut());

        let r1 = 255;
        let r2 = 50;

        // Set custom state to save.
        painter.set_stroke([r1, 0, 0]);

        painter.with_save(|painter| {
            painter.set_stroke([r2, 0, 0]);
            painter.draw_polygon(&Rect::from_tlbr(3, 3, 4, 4).corners());
        });

        // Draw outer rect with earlier saved state.
        painter.draw_polygon(&Rect::from_tlbr(2, 2, 5, 5).corners());

        let expected = image_from_2d_array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, r1, r1, r1, r1],
            [0, 0, r1, r2, r2, r1],
            [0, 0, r1, r2, r2, r1],
            [0, 0, r1, r1, r1, r1],
        ]);
        compare_images(img.slice([0]), expected.view());
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
}
