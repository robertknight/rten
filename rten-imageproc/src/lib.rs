//! Provides 2D geometry and image processing functions.
//!
//! This includes:
//!
//! - 2D vectors and related math
//! - 2D shapes and associated algorithms: [Point], [Line], [Rect],
//!   [RotatedRect], [Polygon]
//! - Rudimentary drawing functions
//! - Algorithms for finding the contours of connected components in an image
//!   ([find_contours])
//! - Algorithms for simplifying polygons and finding various kinds of shape
//!   that contain a polygon: [simplify_polygon], [min_area_rect], [convex_hull]

mod contours;
mod drawing;
mod math;
mod normalize;
mod poly_algos;
mod shapes;

pub use contours::{RetrievalMode, find_contours};
pub use drawing::{FillIter, Painter, Rgb, draw_line, draw_polygon, fill_rect, stroke_rect};
pub use math::Vec2;
pub use normalize::{IMAGENET_MEAN, IMAGENET_STD_DEV, normalize_image};
pub use poly_algos::{convex_hull, min_area_rect, simplify_polygon, simplify_polyline};
pub use shapes::{
    BoundingRect, Coord, Line, LineF, Point, PointF, Polygon, PolygonF, Polygons, Rect, RectF,
    RotatedRect, bounding_rect,
};

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use rten_tensor::{MatrixLayout, NdTensorView, NdTensorViewMut};

    use super::{Coord, Point, Rect};

    /// Return a list of the points on the border of `rect`, in counter-clockwise
    /// order starting from the top-left corner.
    ///
    /// If `omit_corners` is true, the corner points of the rect are not
    /// included.
    pub fn border_points(rect: Rect, omit_corners: bool) -> Vec<Point> {
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
    pub fn plot_points<T: Copy>(mut grid: NdTensorViewMut<T, 2>, points: &[Point], value: T) {
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

    /// Convert a slice of `[y, x]` coordinates to `Point`s
    pub fn points_from_coords<T: Coord>(coords: &[[T; 2]]) -> Vec<Point<T>> {
        coords.iter().map(|[y, x]| Point::from_yx(*y, *x)).collect()
    }

    /// Convery an array of `[y, x]` coordinates to `Point`s
    pub fn points_from_n_coords<T: Coord, const N: usize>(coords: [[T; 2]; N]) -> [Point<T>; N] {
        coords.map(|[y, x]| Point::from_yx(y, x))
    }

    /// Print out elements of a 2D grid for debugging.
    #[allow(dead_code)]
    pub fn print_grid<T: Display>(grid: NdTensorView<T, 2>) {
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                print!("{:2} ", grid[[y, x]]);
            }
            println!();
        }
        println!();
    }
}
