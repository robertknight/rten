use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;
use std::slice::Iter;

use crate::{FillIter, Vec2};

/// Trait for shapes which have a well-defined bounding rectangle.
pub trait BoundingRect {
    /// Coordinate type of bounding rect.
    type Coord: Coord;

    /// Return the smallest axis-aligned bounding rect which contains this
    /// shape.
    fn bounding_rect(&self) -> Rect<Self::Coord>;
}

/// Trait for types which can be used as coordinates of shapes.
///
/// This trait captures the most common requirements of integral and float
/// coordinate types for various shape methods.
pub trait Coord:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + std::fmt::Display
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
{
    /// Return true if this coordinate is a float NaN value.
    fn is_nan(self) -> bool;
}

impl Coord for f32 {
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl Coord for i32 {
    fn is_nan(self) -> bool {
        false
    }
}

/// Return the minimum of `a` and `b`, or `a` if `a` and `b` are unordered.
fn min_or_lhs<T: PartialOrd>(a: T, b: T) -> T {
    if b < a {
        b
    } else {
        a
    }
}

/// Return the maximum of `a` and `b`, or `a` if `a` and `b` are unordered.
fn max_or_lhs<T: PartialOrd>(a: T, b: T) -> T {
    if b > a {
        b
    } else {
        a
    }
}

/// A point defined by X and Y coordinates.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
pub struct Point<T: Coord = i32> {
    pub x: T,
    pub y: T,
}

pub type PointF = Point<f32>;

impl<T: Coord> Point<T> {
    /// Construct a point from X and Y coordinates.
    pub fn from_yx(y: T, x: T) -> Self {
        Point { y, x }
    }

    /// Set the coordinates of this point.
    pub fn move_to(&mut self, y: T, x: T) {
        self.y = y;
        self.x = x;
    }

    pub fn translate(self, y: T, x: T) -> Self {
        Point {
            y: self.y + y,
            x: self.x + x,
        }
    }

    pub fn move_by(&mut self, y: T, x: T) {
        *self = self.translate(y, x);
    }
}

impl Point<f32> {
    pub fn distance(self, other: Self) -> f32 {
        self.vec_to(other).length()
    }

    /// Return the vector from this point to another point.
    pub fn vec_to(self, other: Self) -> Vec2 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        Vec2::from_xy(dx, dy)
    }

    /// Return the vector from the origin to this point.
    pub fn to_vec(self) -> Vec2 {
        Vec2::from_xy(self.x, self.y)
    }
}

impl Point<i32> {
    /// Return self as a [y, x] array. This is useful for indexing into an
    /// image or matrix.
    ///
    /// Panics if the X or Y coordinates of the point are negative.
    pub fn coord(self) -> [usize; 2] {
        assert!(self.y >= 0 && self.x >= 0, "Coordinates are negative");
        [self.y as usize, self.x as usize]
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

    pub fn to_f32(self) -> Point<f32> {
        Point {
            x: self.x as f32,
            y: self.y as f32,
        }
    }

    pub fn distance(self, other: Point<i32>) -> f32 {
        self.to_f32().distance(other.to_f32())
    }
}

impl<T: Coord> fmt::Debug for Point<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.y, self.x)
    }
}

/// Compute the overlap between two 1D lines `a` and `b`, where each line is
/// given as (start, end) coords.
///
/// Returns NaN if any input coordinate is NaN.
fn overlap<T: Coord>(a: (T, T), b: (T, T)) -> T {
    let a = sort_pair(a);
    let b = sort_pair(b);
    let ((_a_start, a_end), (b_start, b_end)) = sort_pair((a, b));

    let min_overlap = T::default();
    let max_overlap = b_end - b_start;
    let overlap = a_end - b_start;

    // This check handles NaN for two of the inputs. The checks below will
    // return NaN if `overlap` is NaN, which handles the other two inputs.
    if max_overlap.is_nan() {
        return max_overlap;
    }

    if overlap < min_overlap {
        min_overlap
    } else if overlap > max_overlap {
        max_overlap
    } else {
        overlap
    }
}

/// Sort the elements of a tuple. If the ordering of the elements is undefined,
/// return the input unchanged.
fn sort_pair<T: PartialOrd>(pair: (T, T)) -> (T, T) {
    if pair.0 > pair.1 {
        (pair.1, pair.0)
    } else {
        pair
    }
}

/// A bounded line segment defined by a start and end point.
#[derive(Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
pub struct Line<T: Coord = i32> {
    pub start: Point<T>,
    pub end: Point<T>,
}

pub type LineF = Line<f32>;

impl<T: Coord> fmt::Debug for Line<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} -> {:?}", self.start, self.end)
    }
}

impl<T: Coord> Line<T> {
    pub fn from_endpoints(start: Point<T>, end: Point<T>) -> Line<T> {
        Line { start, end }
    }

    /// Return true if this line has zero length.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Return the difference between the starting and ending X coordinates of
    /// the line.
    pub fn width(&self) -> T {
        self.end.x - self.start.x
    }

    /// Return the difference between the starting and ending Y coordinates of
    /// the line.
    pub fn height(&self) -> T {
        self.end.y - self.start.y
    }

    /// Return true if the Y coordinate of the line's start and end points are
    /// the same.
    pub fn is_horizontal(&self) -> bool {
        self.start.y == self.end.y
    }

    /// Return a copy of this line with the start and end points swapped.
    pub fn reverse(&self) -> Line<T> {
        Line::from_endpoints(self.end, self.start)
    }

    /// Return a copy of this line with the same endpoints but swapped if
    /// needed so that `end.y >= start.y`.
    pub fn downwards(&self) -> Line<T> {
        if self.start.y <= self.end.y {
            *self
        } else {
            self.reverse()
        }
    }

    /// Return a copy of this line with the same endpoints but swapped if
    /// needed so that `end.x >= start.x`.
    pub fn rightwards(&self) -> Line<T> {
        if self.start.x <= self.end.x {
            *self
        } else {
            self.reverse()
        }
    }

    /// Return the number of pixels by which this line overlaps `other` in the
    /// vertical direction.
    pub fn vertical_overlap(&self, other: Line<T>) -> T {
        overlap((self.start.y, self.end.y), (other.start.y, other.end.y))
    }

    /// Return the number of pixels by which this line overlaps `other` in the
    /// horizontal direction.
    pub fn horizontal_overlap(&self, other: Line<T>) -> T {
        overlap((self.start.x, self.end.x), (other.start.x, other.end.x))
    }
}

impl Line<f32> {
    /// Return the euclidean distance between a point and the closest coordinate
    /// that lies on the line.
    pub fn distance(&self, p: PointF) -> f32 {
        if self.is_empty() {
            return self.start.distance(p);
        }

        // Method taken from http://www.faqs.org/faqs/graphics/algorithms-faq/,
        // "Subject 1.02: How do I find the distance from a point to a line?".

        // Compute normalized scalar projection of line from `start` to `p` onto
        // self. This indicates how far along the `self` line the nearest point
        // to `p` is.
        let ab = self.start.vec_to(self.end);
        let ac = self.start.vec_to(p);
        let scalar_proj = ac.dot(ab) / (ab.length() * ab.length());

        if scalar_proj <= 0. {
            // Nearest point is start of line.
            self.start.distance(p)
        } else if scalar_proj >= 1. {
            // Nearest point is end of line.
            self.end.distance(p)
        } else {
            let intercept_x = self.start.x + ab.x * scalar_proj;
            let intercept_y = self.start.y + ab.y * scalar_proj;
            let proj_line = Vec2::from_yx(intercept_y - p.y, intercept_x - p.x);
            proj_line.length()
        }
    }

    /// Test whether this line segment intersects `other` at a single point.
    ///
    /// Returns false if the line segments do not intersect, or are coincident
    /// (ie. overlap for part of their lengths).
    pub fn intersects(&self, other: Line<f32>) -> bool {
        // See https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments

        let (x1, x2) = (self.start.x, self.end.x);
        let (y1, y2) = (self.start.y, self.end.y);
        let (x3, x4) = (other.start.x, other.end.x);
        let (y3, y4) = (other.start.y, other.end.y);

        // To find the intersection, we first represent the lines as functions
        // parametrized by `s` and `t`:
        //
        // x(s), y(s) = x1 + s(x2 - x1), y1 + s(y2 - y1)
        // x(t), y(t) = x3 + t(x4 - x3), y3 + t(y4 - y3)
        //
        // Then the coordinates of the intersection s0 and t0 are the solutions
        // of:
        //
        // s(x2 - x1) - t(x4 - x3) = x3 - x1
        // s(y2 - y1) - t(y4 - y3) = y3 - y1
        //
        // These equations are solved using Cramer's rule. The lines intersect
        // if s0 and t0 are in [0, 1].

        let a = x2 - x1;
        let b = -(x4 - x3);
        let c = y2 - y1;
        let d = -(y4 - y3);

        let b0 = x3 - x1;
        let b1 = y3 - y1;

        let det_a = a * d - b * c;
        if det_a == 0. {
            // Lines are either parallel or coincident.
            return false;
        }
        let det_a0 = b0 * d - b * b1;
        let det_a1 = a * b1 - b0 * c;

        // We could calculate `s0` as `det_a0 / det_a` and `t0` as `det_a1 / det_a`
        // (using float division). We only need to test whether s0 and t0 are
        // in [0, 1] though, so this can be done without division.
        let s_ok = (det_a0 >= 0.) == (det_a > 0.) && det_a0.abs() <= det_a.abs();
        let t_ok = (det_a1 >= 0.) == (det_a > 0.) && det_a1.abs() <= det_a.abs();

        s_ok && t_ok
    }
}

impl Line<f32> {
    /// Return the midpoint between the start and end points of the line.
    pub fn center(&self) -> PointF {
        let cy = (self.start.y + self.end.y) / 2.;
        let cx = (self.start.x + self.end.x) / 2.;
        Point::from_yx(cy, cx)
    }

    /// Return `(slope, intercept)` tuple for line or None if the line is
    /// vertical.
    fn slope_intercept(&self) -> Option<(f32, f32)> {
        let dx = self.end.x - self.start.x;
        if dx == 0. {
            return None;
        }
        let slope = (self.end.y - self.start.y) / dx;
        let intercept = self.start.y - slope * self.start.x;
        Some((slope, intercept))
    }

    /// Return the X coordinate that corresponds to a given Y coordinate on
    /// the line.
    ///
    /// Returns `None` if the Y coordinate is not on the line or the line is
    /// horizontal.
    pub fn x_for_y(&self, y: f32) -> Option<f32> {
        let (min_y, max_y) = sort_pair((self.start.y, self.end.y));
        if y < min_y || y > max_y || min_y == max_y {
            return None;
        }
        self.slope_intercept()
            .map(|(slope, intercept)| (y - intercept) / slope)
            .or(Some(self.start.x))
    }

    /// Return the Y coordinate that corresponds to a given X coordinate on
    /// the line.
    ///
    /// Returns `None` if the X coordinate is not on the line or the line
    /// is vertical.
    pub fn y_for_x(&self, x: f32) -> Option<f32> {
        let (min_x, max_x) = sort_pair((self.start.x, self.end.x));
        if x < min_x || x > max_x {
            return None;
        }
        self.slope_intercept()
            .map(|(slope, intercept)| slope * x + intercept)
    }
}

impl Line<i32> {
    pub fn to_f32(&self) -> LineF {
        Line::from_endpoints(self.start.to_f32(), self.end.to_f32())
    }

    /// Return the euclidean distance between a point and the closest coordinate
    /// that lies on the line.
    pub fn distance(&self, p: Point) -> f32 {
        self.to_f32().distance(p.to_f32())
    }

    /// Test whether this line segment intersects `other` at a single point.
    ///
    /// Returns false if the line segments do not intersect, or are coincident
    /// (ie. overlap for part of their lengths).
    pub fn intersects(&self, other: Line) -> bool {
        self.to_f32().intersects(other.to_f32())
    }
}

impl<T: Coord> BoundingRect for Line<T> {
    type Coord = T;

    fn bounding_rect(&self) -> Rect<T> {
        let d = self.downwards();
        let r = self.rightwards();
        Rect::from_tlbr(d.start.y, r.start.x, d.end.y, r.end.x)
    }
}

/// Rectangle defined by left, top, right and bottom coordinates.
///
/// The left and top coordinates are inclusive. The right and bottom coordinates
/// are exclusive.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect<T: Coord = i32> {
    top_left: Point<T>,
    bottom_right: Point<T>,
}

pub type RectF = Rect<f32>;

impl<T: Coord> Rect<T> {
    pub fn new(top_left: Point<T>, bottom_right: Point<T>) -> Rect<T> {
        Rect {
            top_left,
            bottom_right,
        }
    }

    pub fn width(&self) -> T {
        // TODO - Handle inverted rects here
        self.bottom_right.x - self.top_left.x
    }

    pub fn height(&self) -> T {
        // TODO - Handle inverted rects here
        self.bottom_right.y - self.top_left.y
    }

    pub fn top(&self) -> T {
        self.top_left.y
    }

    pub fn left(&self) -> T {
        self.top_left.x
    }

    pub fn right(&self) -> T {
        self.bottom_right.x
    }

    pub fn bottom(&self) -> T {
        self.bottom_right.y
    }

    /// Return the corners of the rect in clockwise order, starting from the
    /// top left.
    pub fn corners(&self) -> [Point<T>; 4] {
        [
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
    }

    /// Return the coordinate of the top-left corner of the rect.
    pub fn top_left(&self) -> Point<T> {
        self.top_left
    }

    /// Return the coordinate of the top-right corner of the rect.
    pub fn top_right(&self) -> Point<T> {
        Point::from_yx(self.top_left.y, self.bottom_right.x)
    }

    /// Return the coordinate of the bottom-left corner of the rect.
    pub fn bottom_left(&self) -> Point<T> {
        Point::from_yx(self.bottom_right.y, self.top_left.x)
    }

    /// Return the coordinate of the bottom-right corner of the rect.
    pub fn bottom_right(&self) -> Point<T> {
        self.bottom_right
    }

    /// Return the line segment of the left edge of the rect.
    pub fn left_edge(&self) -> Line<T> {
        Line::from_endpoints(self.top_left(), self.bottom_left())
    }

    /// Return the line segment of the top edge of the rect.
    pub fn top_edge(&self) -> Line<T> {
        Line::from_endpoints(self.top_left(), self.top_right())
    }

    /// Return the line segment of the right edge of the rect.
    pub fn right_edge(&self) -> Line<T> {
        Line::from_endpoints(self.top_right(), self.bottom_right())
    }

    /// Return the line segment of the bottom edge of the rect.
    pub fn bottom_edge(&self) -> Line<T> {
        Line::from_endpoints(self.bottom_left(), self.bottom_right())
    }

    /// Return the top, left, bottom and right coordinates as an array.
    pub fn tlbr(&self) -> [T; 4] {
        [
            self.top_left.y,
            self.top_left.x,
            self.bottom_right.y,
            self.bottom_right.x,
        ]
    }

    /// Return a rect with top-left corner at 0, 0 and the given height/width.
    pub fn from_hw(height: T, width: T) -> Rect<T> {
        Self::new(Point::default(), Point::from_yx(height, width))
    }

    /// Return a rect with the given top, left, bottom and right coordinates.
    pub fn from_tlbr(top: T, left: T, bottom: T, right: T) -> Rect<T> {
        Self::new(Point::from_yx(top, left), Point::from_yx(bottom, right))
    }

    /// Return a rect with the given top, left, height and width.
    pub fn from_tlhw(top: T, left: T, height: T, width: T) -> Rect<T> {
        Self::from_tlbr(top, left, top + height, left + width)
    }

    /// Return the signed area of this rect.
    pub fn area(&self) -> T
    where
        T: std::ops::Mul<Output = T>,
    {
        self.width() * self.height()
    }

    /// Return the top, left, height and width as an array.
    pub fn tlhw(&self) -> [T; 4] {
        [
            self.top_left.y,
            self.top_left.x,
            self.height(),
            self.width(),
        ]
    }

    /// Return true if `other` lies on the boundary or interior of this rect.
    pub fn contains_point(&self, other: Point<T>) -> bool {
        self.top() <= other.y
            && self.bottom() >= other.y
            && self.left() <= other.x
            && self.right() >= other.x
    }

    /// Return true if the width or height of this rect are <= 0.
    pub fn is_empty(&self) -> bool {
        self.right() <= self.left() || self.bottom() <= self.top()
    }

    /// Return a new Rect with each coordinate adjusted by an offset.
    pub fn adjust_tlbr(&self, top: T, left: T, bottom: T, right: T) -> Rect<T> {
        Rect {
            top_left: self.top_left.translate(top, left),
            bottom_right: self.bottom_right.translate(bottom, right),
        }
    }

    /// Return true if the intersection of this rect and `other` is non-empty.
    pub fn intersects(&self, other: Rect<T>) -> bool {
        self.left_edge().vertical_overlap(other.left_edge()) > T::default()
            && self.top_edge().horizontal_overlap(other.top_edge()) > T::default()
    }

    /// Return the smallest rect that contains both this rect and `other`.
    pub fn union(&self, other: Rect<T>) -> Rect<T> {
        let t = min_or_lhs(self.top(), other.top());
        let l = min_or_lhs(self.left(), other.left());
        let b = max_or_lhs(self.bottom(), other.bottom());
        let r = max_or_lhs(self.right(), other.right());
        Rect::from_tlbr(t, l, b, r)
    }

    /// Return the largest rect that is contained within this rect and `other`.
    pub fn intersection(&self, other: Rect<T>) -> Rect<T> {
        let t = max_or_lhs(self.top(), other.top());
        let l = max_or_lhs(self.left(), other.left());
        let b = min_or_lhs(self.bottom(), other.bottom());
        let r = min_or_lhs(self.right(), other.right());
        Rect::from_tlbr(t, l, b, r)
    }

    /// Return true if `other` lies entirely within this rect.
    pub fn contains(&self, other: Rect<T>) -> bool {
        self.union(other) == *self
    }

    /// Return a new with each side adjusted so that the result lies inside
    /// `rect`.
    pub fn clamp(&self, rect: Rect<T>) -> Rect<T> {
        self.intersection(rect)
    }

    pub fn to_polygon(&self) -> Polygon<T, [Point<T>; 4]> {
        Polygon::new(self.corners())
    }
}

impl Rect<i32> {
    /// Return the center point of the rect.
    pub fn center(&self) -> Point {
        let y = (self.top_left.y + self.bottom_right.y) / 2;
        let x = (self.top_left.x + self.bottom_right.x) / 2;
        Point::from_yx(y, x)
    }

    /// Return the Intersection over Union ratio for this rect and `other`.
    ///
    /// See <https://en.wikipedia.org/wiki/Jaccard_index>.
    pub fn iou(&self, other: Rect) -> f32 {
        self.intersection(other).area() as f32 / self.union(other).area() as f32
    }

    pub fn to_f32(&self) -> RectF {
        Rect::from_tlbr(
            self.top_left.y as f32,
            self.top_left.x as f32,
            self.bottom_right.y as f32,
            self.bottom_right.x as f32,
        )
    }
}

impl Rect<f32> {
    /// Return the center point of the rect.
    pub fn center(&self) -> PointF {
        let y = (self.top_left.y + self.bottom_right.y) / 2.;
        let x = (self.top_left.x + self.bottom_right.x) / 2.;
        Point::from_yx(y, x)
    }

    /// Return the Intersection over Union ratio for this rect and `other`.
    ///
    /// See <https://en.wikipedia.org/wiki/Jaccard_index>.
    pub fn iou(&self, other: RectF) -> f32 {
        self.intersection(other).area() / self.union(other).area()
    }

    /// Return the smallest rect with integral coordinates that contains this
    /// rect.
    pub fn integral_bounding_rect(&self) -> Rect<i32> {
        Rect::from_tlbr(
            self.top() as i32,
            self.left() as i32,
            self.bottom().ceil() as i32,
            self.right().ceil() as i32,
        )
    }
}

impl<T: Coord> BoundingRect for Rect<T> {
    type Coord = T;

    fn bounding_rect(&self) -> Rect<T> {
        *self
    }
}

/// An oriented rectangle.
///
/// This is characterized by a center point, an "up" direction indicating the
/// orientation, width (extent along axis perpendicular to the up axis) and
/// height (extent along up axis).
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
pub struct RotatedRect {
    // Centroid of the rect.
    center: PointF,

    // Unit-length vector indicating the "up" direction for this rect.
    up: Vec2,

    // Extent of the rect along the axis perpendicular to `up`.
    width: f32,

    // Extent of the rect along the `up` axis.
    height: f32,
}

impl RotatedRect {
    /// Construct a new RotatedRect with a given `center`, up direction and
    /// dimensions.
    pub fn new(center: PointF, up_axis: Vec2, width: f32, height: f32) -> RotatedRect {
        RotatedRect {
            center,
            up: up_axis.normalized(),
            width,
            height,
        }
    }

    /// Return true if a point lies within this rotated rect.
    pub fn contains(&self, point: PointF) -> bool {
        // Treat zero width/height rectangles as being very thin rectangles
        // rather than lines.
        let height = self.height.max(1e-6);
        let width = self.width.max(1e-6);

        // Project line from center to `p` onto the up and cross axis. The
        // results will be in the range [-1, 1] if the point is within the
        // rect.
        //
        // See notes in `Line::distance` about distance from point to a line.
        let ac = point.to_vec() - self.center.to_vec();
        let ab = self.up * (height / 2.);
        let up_proj = ac.dot(ab) / (height / 2.).powi(2);

        let ad = self.up.perpendicular() * (width / 2.);
        let cross_proj = ac.dot(ad) / (width / 2.).powi(2);

        up_proj.abs() <= 1. && cross_proj.abs() <= 1.
    }

    /// Return a copy of this rect with width increased by `dw` and height
    /// increased by `dh`.
    pub fn expanded(&self, dw: f32, dh: f32) -> RotatedRect {
        RotatedRect {
            width: self.width + dw,
            height: self.height + dh,
            ..*self
        }
    }

    /// Return the coordinates of the rect's corners.
    ///
    /// The corners are returned in clockwise order starting from the corner
    /// that is the top-left when the "up" axis has XY coordinates [0, 1], or
    /// equivalently, bottom-right when the "up" axis has XY coords [0, -1].
    pub fn corners(&self) -> [PointF; 4] {
        let par_offset = self.up.perpendicular() * (self.width / 2.);
        let perp_offset = self.up * (self.height / 2.);

        let center = self.center.to_vec();
        let coords: [Vec2; 4] = [
            center - perp_offset - par_offset,
            center - perp_offset + par_offset,
            center + perp_offset + par_offset,
            center + perp_offset - par_offset,
        ];

        coords.map(|c| Point::from_yx(c.y, c.x))
    }

    /// Return the edges of this rect, in clockwise order starting from the
    /// edge that is the top edge if the rect has no rotation.
    pub fn edges(&self) -> [LineF; 4] {
        let corners = self.corners();
        [
            Line::from_endpoints(corners[0], corners[1]),
            Line::from_endpoints(corners[1], corners[2]),
            Line::from_endpoints(corners[2], corners[3]),
            Line::from_endpoints(corners[3], corners[0]),
        ]
    }

    /// Return the centroid of the rect.
    pub fn center(&self) -> PointF {
        self.center
    }

    /// Return the normalized vector that indicates the "up" direction for
    /// this rect.
    pub fn up_axis(&self) -> Vec2 {
        self.up
    }

    /// Return the extent of the rect along the axis perpendicular to `self.up_axis()`.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Return the extent of the rect along `self.up_axis()`.
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Return the signed area of this rect.
    pub fn area(&self) -> f32 {
        self.height * self.width
    }

    /// Set the extents of this rect. `width` and `height` must be >= 0.
    pub fn resize(&mut self, width: f32, height: f32) {
        assert!(width >= 0. && height >= 0.);
        self.width = width;
        self.height = height;
    }

    /// Return true if the intersection of this rect and `other` is non-empty.
    pub fn intersects(&self, other: &RotatedRect) -> bool {
        if !self.bounding_rect().intersects(other.bounding_rect()) {
            return false;
        }
        let other_edges = other.edges();
        self.edges()
            .iter()
            .any(|e| other_edges.iter().any(|oe| e.intersects(*oe)))
    }

    /// Return a new axis-aligned RotatedRect whose bounding rectangle matches
    /// `r`.
    pub fn from_rect(r: RectF) -> RotatedRect {
        RotatedRect::new(r.center(), Vec2::from_yx(1., 0.), r.width(), r.height())
    }

    /// Return the rectangle with the same corner points as `self`, but with
    /// an up axis that has a direction as close to `up` as possible.
    pub fn orient_towards(&self, up: Vec2) -> RotatedRect {
        let target_up = up.normalized();

        let rot_90 = Vec2::from_xy(self.up.y, -self.up.x);
        let rot_180 = Vec2::from_xy(-self.up.x, -self.up.y);
        let rot_270 = Vec2::from_xy(-self.up.y, self.up.x);

        let (rotation, _dotp) = [self.up, rot_90, rot_180, rot_270]
            .map(|new_up| new_up.dot(target_up))
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, 0.));

        match rotation {
            0 => *self,
            1 => RotatedRect::new(self.center, rot_90, self.height, self.width),
            2 => RotatedRect::new(self.center, rot_180, self.width, self.height),
            3 => RotatedRect::new(self.center, rot_270, self.height, self.width),
            _ => unreachable!(),
        }
    }
}

impl BoundingRect for RotatedRect {
    type Coord = f32;

    fn bounding_rect(&self) -> RectF {
        let corners = self.corners();

        let mut xs = corners.map(|p| p.x);
        xs.sort_unstable_by(f32::total_cmp);

        let mut ys = corners.map(|p| p.y);
        ys.sort_unstable_by(f32::total_cmp);

        Rect::from_tlbr(ys[0], xs[0], ys[3], xs[3])
    }
}

/// Return the bounding rectangle of a collection of shapes.
///
/// Returns `None` if the collection is empty.
pub fn bounding_rect<'a, Shape: 'a + BoundingRect, I: Iterator<Item = &'a Shape>>(
    objects: I,
) -> Option<Rect<Shape::Coord>>
where
    Shape::Coord: Coord,
{
    objects.fold(None, |bounding_rect, shape| {
        let sbr = shape.bounding_rect();
        bounding_rect.map(|br| br.union(sbr)).or(Some(sbr))
    })
}

/// Polygon shape defined by a list of vertices.
///
/// Depending on the storage type `S`, a Polygon can own its vertices
/// (eg. `Vec<Point>`) or they can borrowed (eg. `&[Point]`).
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
pub struct Polygon<T: Coord = i32, S: AsRef<[Point<T>]> = Vec<Point<T>>> {
    points: S,

    /// Avoids compiler complaining `T` is unused.
    element_type: PhantomData<T>,
}

pub type PolygonF<S = Vec<PointF>> = Polygon<f32, S>;

impl<T: Coord, S: AsRef<[Point<T>]>> Polygon<T, S> {
    /// Create a view of a set of points as a polygon.
    pub fn new(points: S) -> Polygon<T, S> {
        Polygon {
            points,
            element_type: PhantomData,
        }
    }

    /// Return a polygon which borrows its points from this polygon.
    pub fn borrow(&self) -> Polygon<T, &[Point<T>]> {
        Polygon::new(self.points.as_ref())
    }

    /// Return an iterator over the edges of this polygon.
    pub fn edges(&self) -> impl Iterator<Item = Line<T>> + '_ {
        self.points
            .as_ref()
            .iter()
            .zip(self.points.as_ref().iter().cycle().skip(1))
            .map(|(p0, p1)| Line::from_endpoints(*p0, *p1))
    }

    /// Return a slice of the endpoints of the polygon's edges.
    pub fn vertices(&self) -> &[Point<T>] {
        self.points.as_ref()
    }

    /// Return a clone of this polygon which owns its vertices.
    pub fn to_owned(&self) -> Polygon<T> {
        Polygon::new(self.vertices().to_vec())
    }
}

impl<S: AsRef<[Point]>> Polygon<i32, S> {
    /// Return an iterator over coordinates of pixels that fill the polygon.
    ///
    /// Polygon filling treats the polygon's vertices as being located at the
    /// center of pixels with the corresponding coordinates. Pixels are deemed
    /// inside the polygon if a ray from -infinity to the pixel's center crosses
    /// an odd number of polygon edges, aka. the even-odd rule [^1]. Pixel
    /// centers which lie exactly on a polygon edge are treated as inside
    /// the polygon for top/left edges and outside the polygon for bottom/right
    /// edges. This follows conventions in various graphics libraries (eg. [^2]).
    ///
    /// This treatment of polygon vertices differs from graphics libraries like
    /// Skia or Qt which use float coordinates for paths. In those libraries
    /// polygon filling is still based on the relationship between polygon edges
    /// and pixel centers, but integer polygon vertex coordinates refer to the
    /// corners of pixels.
    ///
    /// [^1]: <https://en.wikipedia.org/wiki/Even–odd_rule>
    /// [^2]: <https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules#triangle-rasterization-rules-without-multisampling>
    pub fn fill_iter(&self) -> FillIter {
        FillIter::new(self.borrow())
    }

    /// Return true if the pixel with coordinates `p` lies inside the polygon.
    ///
    /// The intent of this function is to align with [Polygon::fill_iter] such
    /// that `polygon.contains_pixel(p)` is equivalent to
    /// `polygon.fill_iter().any(|fp| fp == p)` but faster because it doesn't
    /// iterate over every pixel inside the polygon. See [Polygon::fill_iter]
    /// for notes on how the inside/outisde status of a pixel is determined.
    pub fn contains_pixel(&self, p: Point) -> bool {
        let mut edge_crossings = 0;

        for edge in self.edges() {
            let (min_y, max_y) = sort_pair((edge.start.y, edge.end.y));

            // Ignore horizontal edges.
            if min_y == max_y {
                continue;
            }

            // Skip edges that don't overlap this point vertically.
            if p.y < min_y || p.y >= max_y {
                continue;
            }

            // Check which side of the edge this point is on.
            let edge_down = edge.downwards();
            let start_to_end = Vec2::from_yx(
                (edge_down.end.y - edge_down.start.y) as f32,
                (edge_down.end.x - edge_down.start.x) as f32,
            );
            let start_to_point = Vec2::from_yx(
                (p.y - edge_down.start.y) as f32,
                (p.x - edge_down.start.x) as f32,
            );

            let cross = start_to_end.cross_product_norm(start_to_point);
            if cross > 0. {
                // Edge lies to the left of the pixel.
                edge_crossings += 1;
            }
        }

        edge_crossings % 2 == 1
    }

    /// Return true if this polygon has no self-intersections and no holes.
    pub fn is_simple(&self) -> bool {
        // Test for self intersections. We don't need to test for holes
        // because this struct can't model a polygon with holes.
        for (i, e1) in self.edges().enumerate() {
            for (j, e2) in self.edges().enumerate() {
                if i != j && e1.intersects(e2) {
                    let intersection_at_endpoints = e1.start == e2.start
                        || e1.start == e2.end
                        || e1.end == e2.start
                        || e1.end == e2.end;
                    if !intersection_at_endpoints {
                        return false;
                    }
                }
            }
        }
        true
    }
}

macro_rules! impl_bounding_rect_for_polygon {
    ($coord:ty) => {
        impl<S: AsRef<[Point<$coord>]>> BoundingRect for Polygon<$coord, S> {
            type Coord = $coord;

            fn bounding_rect(&self) -> Rect<$coord> {
                let mut min_x = <$coord>::MAX;
                let mut max_x = <$coord>::MIN;
                let mut min_y = <$coord>::MAX;
                let mut max_y = <$coord>::MIN;

                for p in self.points.as_ref() {
                    min_x = min_x.min(p.x);
                    max_x = max_x.max(p.x);
                    min_y = min_y.min(p.y);
                    max_y = max_y.max(p.y);
                }

                Rect::from_tlbr(min_y, min_x, max_y, max_x)
            }
        }
    };
}

impl_bounding_rect_for_polygon!(i32);
impl_bounding_rect_for_polygon!(f32);

/// A collection of polygons, where each polygon is defined by a slice of points.
///
/// `Polygons` is primarily useful when building up collections of many polygons
/// as it stores all points in a single Vec, which is more efficient than
/// allocating a separate Vec for each polygon's points.
#[cfg_attr(feature = "serde_traits", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg(test)]
mod tests {
    use rten_tensor::test_util::ApproxEq;
    use rten_tensor::{MatrixLayout, NdTensor};

    use crate::tests::{points_from_coords, points_from_n_coords};
    use crate::Vec2;

    use super::{bounding_rect, BoundingRect, Line, Point, PointF, Polygon, Rect, RotatedRect};

    #[test]
    fn test_bounding_rect() {
        let rects = [Rect::from_tlbr(0, 0, 5, 5), Rect::from_tlbr(10, 10, 15, 18)];
        assert_eq!(
            bounding_rect(rects.iter()),
            Some(Rect::from_tlbr(0, 0, 15, 18))
        );

        let rects: &[Rect] = &[];
        assert_eq!(bounding_rect(rects.iter()), None);
    }

    #[test]
    fn test_line_distance() {
        struct Case {
            start: Point,
            end: Point,
            point: Point,
            dist: f32,
        }

        // TODO - Test cases where intercept is beyond start/end of line.
        let cases = [
            // Single point
            Case {
                start: Point::default(),
                end: Point::default(),
                point: Point::from_yx(3, 4),
                dist: 5.,
            },
            // Horizontal line
            Case {
                start: Point::from_yx(5, 2),
                end: Point::from_yx(5, 10),
                point: Point::from_yx(8, 5),
                dist: 3.,
            },
            // Vertical line
            Case {
                start: Point::from_yx(5, 3),
                end: Point::from_yx(10, 3),
                point: Point::from_yx(8, 5),
                dist: 2.,
            },
            // Line with +ve gradient
            Case {
                start: Point::default(),
                end: Point::from_yx(5, 5),
                point: Point::from_yx(4, 0),
                dist: (8f32).sqrt(), // Closest point is at (2, 2)
            },
            // Line with -ve gradient
            Case {
                start: Point::default(),
                end: Point::from_yx(5, -5),
                point: Point::from_yx(4, 0),
                dist: (8f32).sqrt(), // Closest point is at (2, -2)
            },
            // Point below line
            Case {
                start: Point::default(),
                end: Point::from_yx(5, 5),
                point: Point::from_yx(0, 4),
                dist: (8f32).sqrt(), // Closest point is at (2, 2)
            },
            // Point beyond end of horizontal line
            Case {
                start: Point::from_yx(5, 2),
                end: Point::from_yx(5, 5),
                point: Point::from_yx(5, 10),
                dist: 5.,
            },
        ];

        for case in cases {
            let line = Line::from_endpoints(case.start, case.end);
            let dist = line.distance(case.point);
            assert!(
                dist.approx_eq(&case.dist),
                "line {:?}, {:?} point {:?} actual {} expected {}",
                line.start,
                line.end,
                case.point,
                dist,
                case.dist
            );
        }
    }

    #[test]
    fn test_line_downwards() {
        struct Case {
            input: Line,
            down: Line,
        }
        let cases = [
            Case {
                input: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
                down: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
            },
            Case {
                input: Line::from_endpoints(Point::from_yx(5, 5), Point::from_yx(0, 0)),
                down: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
            },
        ];
        for case in cases {
            assert_eq!(case.input.downwards(), case.down);
        }
    }

    #[test]
    fn test_line_rightwards() {
        struct Case {
            input: Line,
            right: Line,
        }
        let cases = [
            Case {
                input: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
                right: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
            },
            Case {
                input: Line::from_endpoints(Point::from_yx(5, 5), Point::from_yx(0, 0)),
                right: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 5)),
            },
        ];
        for case in cases {
            assert_eq!(case.input.rightwards(), case.right);
        }
    }

    /// Create a line from [y1, x1, y2, x2] coordinates.
    fn line_from_coords(coords: [i32; 4]) -> Line {
        Line::from_endpoints(
            Point::from_yx(coords[0], coords[1]),
            Point::from_yx(coords[2], coords[3]),
        )
    }

    #[test]
    fn test_line_intersects() {
        struct Case {
            a: Line,
            b: Line,
            expected: bool,
        }

        let cases = [
            // Horizontal and vertical lines that intersect
            Case {
                a: line_from_coords([0, 5, 10, 5]),
                b: line_from_coords([5, 0, 5, 10]),
                expected: true,
            },
            // Diagonal lines that intersect
            Case {
                a: line_from_coords([0, 0, 10, 10]),
                b: line_from_coords([10, 0, 0, 10]),
                expected: true,
            },
            // Horizontal and vertical lines that do not intersect
            Case {
                a: line_from_coords([0, 5, 10, 5]),
                b: line_from_coords([5, 6, 5, 10]),
                expected: false,
            },
            Case {
                a: line_from_coords([0, 5, 10, 5]),
                b: line_from_coords([5, 10, 5, 6]),
                expected: false,
            },
            // Horizontal and vertical lines that touch
            Case {
                a: line_from_coords([0, 5, 5, 5]),
                b: line_from_coords([5, 0, 5, 10]),
                expected: true,
            },
            // Test case from https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments
            Case {
                a: line_from_coords([1, 1, 2, 3]),
                b: line_from_coords([4, 1, -1, 2]),
                expected: true,
            },
            // Parallel lines that do not touch
            Case {
                a: line_from_coords([0, 5, 0, 10]),
                b: line_from_coords([2, 5, 2, 10]),
                expected: false,
            },
            // Coincident lines
            Case {
                a: line_from_coords([0, 5, 0, 10]),
                b: line_from_coords([0, 5, 0, 10]),
                expected: false,
            },
        ];

        for case in cases {
            assert_eq!(case.a.intersects(case.b), case.expected);

            // `intersects` should be commutative.
            assert_eq!(case.b.intersects(case.a), case.expected);
        }
    }

    #[test]
    fn test_line_is_horizontal() {
        assert_eq!(
            Line::from_endpoints(Point::from_yx(5, 0), Point::from_yx(5, 10)).is_horizontal(),
            true
        );
        assert_eq!(
            Line::from_endpoints(Point::from_yx(5, 0), Point::from_yx(6, 10)).is_horizontal(),
            false
        );
    }

    #[test]
    fn test_line_overlap() {
        struct Case {
            a: (i32, i32),
            b: (i32, i32),
            overlap: i32,
        }

        let cases = [
            // No overlap
            Case {
                a: (0, 10),
                b: (15, 20),
                overlap: 0,
            },
            // End of `a` overlaps start of `b`
            Case {
                a: (0, 10),
                b: (5, 15),
                overlap: 5,
            },
            // `a` overlaps all of `b`
            Case {
                a: (0, 10),
                b: (2, 8),
                overlap: 6,
            },
            // `a` and `b` start together, but `a` is shorter
            Case {
                a: (0, 5),
                b: (0, 10),
                overlap: 5,
            },
        ];

        for case in cases {
            // Horizontal lines
            let a = Line::from_endpoints(Point::from_yx(0, case.a.0), Point::from_yx(0, case.a.1));
            let b = Line::from_endpoints(Point::from_yx(0, case.b.0), Point::from_yx(0, case.b.1));
            assert_eq!(a.horizontal_overlap(b), case.overlap);
            assert_eq!(b.horizontal_overlap(a), case.overlap);

            // Vertical lines
            let a = Line::from_endpoints(Point::from_yx(case.a.0, 0), Point::from_yx(case.a.1, 0));
            let b = Line::from_endpoints(Point::from_yx(case.b.0, 0), Point::from_yx(case.b.1, 0));
            assert_eq!(a.vertical_overlap(b), case.overlap);
            assert_eq!(b.vertical_overlap(a), case.overlap);
        }
    }

    #[test]
    fn test_line_width_height() {
        struct Case {
            line: Line,
            width: i32,
            height: i32,
        }

        let cases = [
            Case {
                line: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(5, 3)),
                width: 3,
                height: 5,
            },
            Case {
                line: Line::from_endpoints(Point::from_yx(5, 3), Point::from_yx(0, 0)),
                width: -3,
                height: -5,
            },
        ];

        for case in cases {
            assert_eq!(case.line.width(), case.width);
            assert_eq!(case.line.height(), case.height);
        }
    }

    #[test]
    fn test_line_y_for_x_and_x_for_y() {
        struct Case {
            line: Line,

            // (X, expected Y) coordinate pairs.
            points: Vec<(f32, Option<f32>)>,
        }

        let cases = [
            // Slope 1, intercept 0
            Case {
                line: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(1, 1)),
                points: vec![
                    (-1., None),
                    (0., Some(0.)),
                    (0.5, Some(0.5)),
                    (1., Some(1.)),
                    (1.2, None),
                ],
            },
            // Slope 1, intercept -1
            Case {
                line: Line::from_endpoints(Point::from_yx(0, 1), Point::from_yx(1, 2)),
                points: vec![
                    (-1., None),
                    (1., Some(0.)),
                    (1.5, Some(0.5)),
                    (2., Some(1.)),
                    (2.2, None),
                ],
            },
            // Horizontal line
            Case {
                line: Line::from_endpoints(Point::from_yx(0, 1), Point::from_yx(0, 2)),
                points: vec![(-1., None), (1., Some(0.)), (2., Some(0.)), (3., None)],
            },
            // Vertical line
            Case {
                line: Line::from_endpoints(Point::from_yx(0, 0), Point::from_yx(2, 0)),
                points: vec![(-1., None), (0., None), (1., None)],
            },
        ];

        for case in cases {
            for (x, expected_y) in case.points {
                assert_eq!(case.line.to_f32().y_for_x(x), expected_y);
                if let Some(y) = expected_y {
                    assert_eq!(
                        case.line.to_f32().x_for_y(y),
                        if case.line.is_horizontal() {
                            None
                        } else {
                            Some(x)
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn test_point_coord() {
        assert_eq!(Point::from_yx(3, 5).coord(), [3, 5]);
    }

    #[test]
    #[should_panic(expected = "Coordinates are negative")]
    fn test_point_coord_negative() {
        Point::from_yx(-1, -1).coord();
    }

    #[test]
    fn test_polygon_contains_pixel() {
        struct Case {
            poly: Polygon,
        }

        let cases = [
            // Empty polygon
            Case {
                poly: Polygon::new(Vec::new()),
            },
            // Zero-height polygon
            Case {
                poly: Rect::from_tlbr(0, 0, 0, 5).to_polygon().to_owned(),
            },
            // Rects
            Case {
                poly: Rect::from_tlbr(2, 2, 5, 5).to_polygon().to_owned(),
            },
            Case {
                poly: Rect::from_tlbr(0, 0, 1, 1).to_polygon().to_owned(),
            },
            // Inverted rect
            Case {
                poly: Rect::from_tlbr(5, 5, 2, 2).to_polygon().to_owned(),
            },
            // Triangles
            Case {
                poly: Polygon::new(points_from_coords(&[[0, 2], [3, 0], [3, 4]])),
            },
            Case {
                poly: Polygon::new(points_from_coords(&[[1, 1], [4, 3], [6, 9]])),
            },
        ];

        for case in cases {
            // Create two grids that are slightly larger than the max X + Y
            // coordinates.
            let grid_size = case
                .poly
                .vertices()
                .iter()
                .fold([0, 0], |[h, w], point| {
                    [h.max(point.y) + 2, w.max(point.x) + 2]
                })
                .map(|x| x as usize);

            let mut fill_grid = NdTensor::zeros(grid_size);
            let mut contains_pixel_grid = NdTensor::zeros(grid_size);

            // Fill one grid using `fill_iter` and the other using
            // `contains_pixel` tests, then verify that the same pixels get
            // filled.
            for p in case.poly.fill_iter() {
                fill_grid[p.coord()] = 1;
            }
            for y in 0..contains_pixel_grid.rows() {
                for x in 0..contains_pixel_grid.cols() {
                    if case.poly.contains_pixel(Point::from_yx(y as i32, x as i32)) {
                        contains_pixel_grid[[y, x]] = 1;
                    }
                }
            }

            for y in 0..fill_grid.rows() {
                for x in 0..fill_grid.cols() {
                    assert_eq!(fill_grid[[y, x]], contains_pixel_grid[[y, x]]);
                }
            }
        }
    }

    #[test]
    fn test_polygon_is_simple() {
        struct Case {
            poly: Polygon,
            simple: bool,
        }

        let cases = [
            // Simple rect
            Case {
                poly: Rect::from_tlbr(0, 0, 10, 10).to_polygon().to_owned(),
                simple: true,
            },
            // 4-vertex poly with intersection
            Case {
                poly: Polygon::new(points_from_coords(&[[0, 0], [0, 10], [10, 10], [-2, 2]])),
                simple: false,
            },
        ];

        for case in cases {
            assert_eq!(case.poly.is_simple(), case.simple)
        }
    }

    #[test]
    fn test_polygon_fill_iter() {
        struct Case {
            vertices: Vec<Point>,
            filled: Vec<Point>,
        }

        let cases = [
            // Empty polygon
            Case {
                vertices: Vec::new(),
                filled: Vec::new(),
            },
            // Single line
            Case {
                vertices: points_from_coords(&[[0, 0], [5, 5]]),
                filled: Vec::new(),
            },
            // Rect
            Case {
                vertices: Rect::from_tlbr(0, 0, 3, 3).to_polygon().vertices().to_vec(),
                filled: points_from_coords(&[
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [1, 0],
                    [1, 1],
                    [1, 2],
                    [2, 0],
                    [2, 1],
                    [2, 2],
                ]),
            },
            // Triangle
            Case {
                vertices: points_from_coords(&[[0, 0], [0, 4], [3, 4]]),
                filled: points_from_coords(&[
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 2],
                    [1, 3],
                    [2, 3],
                ]),
            },
        ];

        for case in cases {
            let poly = Polygon::new(&case.vertices);
            let filled: Vec<_> = poly.fill_iter().collect();
            assert_eq!(filled, case.filled);
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
    fn test_rect_contains_point() {
        let r = Rect::from_tlbr(5, 5, 10, 10);

        // Points outside rect
        assert_eq!(r.contains_point(Point::from_yx(0, 0)), false);
        assert_eq!(r.contains_point(Point::from_yx(12, 12)), false);

        // Points inside rect
        assert_eq!(r.contains_point(Point::from_yx(8, 8)), true);

        // Points on boundary
        assert_eq!(r.contains_point(Point::from_yx(5, 5)), true);
        assert_eq!(r.contains_point(Point::from_yx(10, 10)), true);
    }

    #[test]
    fn test_rect_tlbr() {
        let r = Rect::from_tlbr(0, 1, 2, 3);
        assert_eq!(r.tlbr(), [0, 1, 2, 3]);
    }

    #[test]
    fn test_rotated_rect_contains() {
        struct Case {
            rrect: RotatedRect,
        }

        let cases = [
            // Axis-aligned
            Case {
                rrect: RotatedRect::new(PointF::from_yx(0., 0.), Vec2::from_yx(1., 0.), 10., 5.),
            },
            // Axis-aligned, inverted.
            Case {
                rrect: RotatedRect::new(PointF::from_yx(0., 0.), Vec2::from_yx(-1., 0.), 10., 5.),
            },
            // Rotated
            Case {
                rrect: RotatedRect::new(PointF::from_yx(0., 0.), Vec2::from_yx(0.5, 0.5), 10., 5.),
            },
        ];

        for Case { rrect: r } in cases {
            assert!(r.contains(r.center()));

            // Test points slightly inside.
            for c in r.expanded(-1e-5, -1e-5).corners() {
                assert!(r.contains(c));
            }

            // Test points slightly outside.
            for c in r.expanded(1e-5, 1e-5).corners() {
                assert!(!r.contains(c));
            }
        }
    }

    #[test]
    fn test_rotated_rect_corners() {
        let r = RotatedRect::new(PointF::from_yx(5., 5.), Vec2::from_yx(1., 0.), 5., 5.);
        let expected = points_from_n_coords([[2.5, 2.5], [2.5, 7.5], [7.5, 7.5], [7.5, 2.5]]);
        assert_eq!(r.corners(), expected);
    }

    #[test]
    fn test_rotated_rect_expanded() {
        let r = RotatedRect::new(PointF::from_yx(0., 0.), Vec2::from_yx(1., 0.), 10., 5.);
        let r = r.expanded(2., 3.);
        assert_eq!(r.width(), 12.);
        assert_eq!(r.height(), 8.);
    }

    #[test]
    fn test_rotated_rect_from_rect() {
        let r = Rect::from_tlbr(5., 10., 50., 40.);
        let rr = RotatedRect::from_rect(r);
        assert_eq!(rr.width(), r.width());
        assert_eq!(rr.height(), r.height());
        assert_eq!(rr.bounding_rect(), r);
    }

    #[test]
    fn test_rotated_rect_intersects() {
        struct Case {
            a: RotatedRect,
            b: RotatedRect,
            bounding_rect_intersects: bool,
            intersects: bool,
        }

        let up_vec = Vec2::from_yx(-1., 0.);
        let up_left_vec = Vec2::from_yx(-1., -1.);

        let cases = [
            // Identical rects
            Case {
                a: RotatedRect::new(PointF::from_yx(5., 5.), up_vec, 5., 5.),
                b: RotatedRect::new(PointF::from_yx(5., 5.), up_vec, 5., 5.),
                bounding_rect_intersects: true,
                intersects: true,
            },
            // Separated rects
            Case {
                a: RotatedRect::new(PointF::from_yx(5., 5.), up_vec, 5., 5.),
                b: RotatedRect::new(PointF::from_yx(5., 11.), up_vec, 5., 5.),
                bounding_rect_intersects: false,
                intersects: false,
            },
            // Case where bounding rectangles intersect but rotated rects do
            // not.
            Case {
                a: RotatedRect::new(PointF::from_yx(5., 5.), up_left_vec, 12., 1.),
                b: RotatedRect::new(PointF::from_yx(9., 9.), up_vec, 1., 1.),
                bounding_rect_intersects: true,
                intersects: false,
            },
        ];

        for case in cases {
            assert_eq!(
                case.a.bounding_rect().intersects(case.b.bounding_rect()),
                case.bounding_rect_intersects
            );
            assert_eq!(case.a.intersects(&case.b), case.intersects);
            // `intersects` should be transitive
            assert_eq!(case.b.intersects(&case.a), case.intersects);
        }
    }

    #[test]
    fn test_rotated_rect_normalizes_up_vector() {
        // Create rotated rect with non-normalized "up" vector.
        let up_axis = Vec2::from_yx(1., 2.);
        let center = PointF::from_yx(0., 0.);
        let rect = RotatedRect::new(center, up_axis, 2., 3.);
        assert!(rect.up_axis().length().approx_eq(&1.));
    }

    #[test]
    fn test_rotated_rect_orient_towards() {
        let up_axis = Vec2::from_yx(-1., 0.);
        let center = PointF::from_yx(0., 0.);
        let rect = RotatedRect::new(center, up_axis, 2., 3.);

        let sorted_corners = |rect: RotatedRect| {
            let mut corners = rect
                .corners()
                .map(|c| Point::from_yx(c.y.round() as i32, c.x.round() as i32));
            corners.sort_by_key(|p| (p.y, p.x));
            corners
        };

        let targets = [
            Vec2::from_yx(-1., 0.),
            Vec2::from_yx(0., 1.),
            Vec2::from_yx(1., 0.),
            Vec2::from_yx(0., -1.),
        ];
        for target in targets {
            let oriented = rect.orient_towards(target);
            assert_eq!(sorted_corners(oriented), sorted_corners(rect));
            if target != up_axis {
                assert_ne!(oriented.corners(), rect.corners());
            }
            assert_eq!(oriented.up_axis(), target);
        }
    }

    #[test]
    fn test_rotated_rect_resize() {
        let mut r = RotatedRect::new(PointF::from_yx(5., 5.), Vec2::from_yx(1., 0.), 5., 5.);
        assert_eq!(r.area(), 25.);

        r.resize(3., 7.);

        assert_eq!(r.width(), 3.);
        assert_eq!(r.height(), 7.);
        assert_eq!(r.area(), 21.);
    }
}
