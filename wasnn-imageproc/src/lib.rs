//! Functions for pre and post-processing images.

mod contours;
mod drawing;
mod math;
mod poly_algos;
mod shapes;

pub use contours::{find_contours, RetrievalMode};
pub use drawing::{draw_line, draw_polygon, fill_rect, stroke_rect, FillIter};
pub use math::Vec2;
pub use poly_algos::{convex_hull, min_area_rect, simplify_polygon, simplify_polyline};
pub use shapes::{bounding_rect, BoundingRect, Line, Point, Polygon, Polygons, Rect, RotatedRect};

#[cfg(test)]
mod tests {
    use wasnn_tensor::NdTensorViewMut;

    use super::{Point, Rect};

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

    /// Convert a slice of `[y, x]` coordinates to `Point`s
    pub fn points_from_coords(coords: &[[i32; 2]]) -> Vec<Point> {
        coords.iter().map(|[y, x]| Point::from_yx(*y, *x)).collect()
    }

    /// Convery an array of `[y, x]` coordinates to `Point`s
    pub fn points_from_n_coords<const N: usize>(coords: [[i32; 2]; N]) -> [Point; N] {
        coords.map(|[y, x]| Point::from_yx(y, x))
    }
}
