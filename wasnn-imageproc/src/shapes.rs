use std::fmt;
use std::iter::zip;
use std::ops::Range;
use std::slice::Iter;

use crate::{FillIter, Vec2};

pub type Coord = i32;

/// Trait for shapes which have a well-defined bounding rectangle.
pub trait BoundingRect {
    /// Return the smallest axis-aligned bounding rect which contains this
    /// shape.
    fn bounding_rect(&self) -> Rect;
}

/// A point defined by integer X and Y coordinates.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Point {
    pub x: Coord,
    pub y: Coord,
}

impl Point {
    pub fn from_yx(y: Coord, x: Coord) -> Point {
        Point { y, x }
    }

    /// Return self as a [y, x] array. This is useful for indexing into an
    /// image or matrix.
    ///
    /// Panics if the X or Y coordinates of the point are negative.
    pub fn coord(self) -> [usize; 2] {
        assert!(self.y >= 0 && self.x >= 0, "Coordinates are negative");
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

    /// Return the euclidean distance between this point and another point.
    pub fn distance(self, other: Point) -> f32 {
        Vec2::from_points(self, other).length()
    }

    pub fn move_by(&mut self, y: Coord, x: Coord) {
        *self = self.translate(y, x);
    }

    /// Set the coordinates of this point.
    pub fn move_to(&mut self, y: Coord, x: Coord) {
        self.y = y;
        self.x = x;
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.y, self.x)
    }
}

/// Compute the overlap between two 1D lines `a` and `b`, where each line is
/// given as (start, end) coords.
fn overlap(a: (i32, i32), b: (i32, i32)) -> i32 {
    let a = sort_pair(a);
    let b = sort_pair(b);
    let ((_a_start, a_end), (b_start, b_end)) = sort_pair((a, b));
    (a_end - b_start).clamp(0, b_end - b_start)
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
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Line {
    pub start: Point,
    pub end: Point,
}

impl Line {
    pub fn from_endpoints(start: Point, end: Point) -> Line {
        Line { start, end }
    }

    /// Return true if this line has zero length.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn width(&self) -> Coord {
        self.end.x - self.start.x
    }

    pub fn height(&self) -> Coord {
        self.end.y - self.start.y
    }

    pub fn center(&self) -> Point {
        let cy = (self.start.y + self.end.y) / 2;
        let cx = (self.start.x + self.end.x) / 2;
        Point::from_yx(cy, cx)
    }

    /// Return the euclidean distance between a point and the closest coordinate
    /// that lies on the line.
    pub fn distance(&self, p: Point) -> f32 {
        if self.is_empty() {
            return self.start.distance(p);
        }

        // Method taken from http://www.faqs.org/faqs/graphics/algorithms-faq/,
        // "Subject 1.02: How do I find the distance from a point to a line?".

        // Compute normalized scalar projection of line from `start` to `p` onto
        // self. This indicates how far along the `self` line the nearest point
        // to `p` is.
        let ab = Vec2::from_points(self.start, self.end);
        let ac = Vec2::from_points(self.start, p);
        let scalar_proj = ac.dot(ab) / (ab.length() * ab.length());

        if scalar_proj <= 0. {
            // Nearest point is start of line.
            self.start.distance(p)
        } else if scalar_proj >= 1. {
            // Nearest point is end of line.
            self.end.distance(p)
        } else {
            let start_x = self.start.x as f32;
            let start_y = self.start.y as f32;
            let intercept_x = start_x + ab.x * scalar_proj;
            let intercept_y = start_y + ab.y * scalar_proj;
            let proj_line = Vec2::from_yx(intercept_y - p.y as f32, intercept_x - p.x as f32);
            proj_line.length()
        }
    }

    /// Return the number of pixels by which this line overlaps `other` in the
    /// vertical direction.
    pub fn vertical_overlap(&self, other: Line) -> i32 {
        overlap((self.start.y, self.end.y), (other.start.y, other.end.y))
    }

    /// Return the number of pixels by which this line overlaps `other` in the
    /// horizontal direction.
    pub fn horizontal_overlap(&self, other: Line) -> i32 {
        overlap((self.start.x, self.end.x), (other.start.x, other.end.x))
    }

    /// Test whether this line segment intersects `other` at a single point.
    ///
    /// Returns false if the line segments do not intersect, or are coincident
    /// (ie. overlap for part of their lengths).
    pub fn intersects(&self, other: Line) -> bool {
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
        if det_a == 0 {
            // Lines are either parallel or coincident.
            return false;
        }
        let det_a0 = b0 * d - b * b1;
        let det_a1 = a * b1 - b0 * c;

        // We could calculate `s0` as `det_a0 / det_a` and `t0` as `det_a1 / det_a`
        // (using float division). We only need to test whether s0 and t0 are
        // in [0, 1] though, so this can be done without division.
        let s_ok = (det_a0 >= 0) == (det_a > 0) && det_a0.abs() <= det_a.abs();
        let t_ok = (det_a1 >= 0) == (det_a > 0) && det_a1.abs() <= det_a.abs();

        s_ok && t_ok
    }

    pub fn is_horizontal(&self) -> bool {
        self.start.y == self.end.y
    }

    /// Return a copy of this line with the start and end points swapped.
    pub fn reverse(&self) -> Line {
        Line::from_endpoints(self.end, self.start)
    }

    /// Return a copy of this line with the same endpoints but swapped if
    /// needed so that `end.y >= start.y`.
    pub fn downwards(&self) -> Line {
        if self.start.y <= self.end.y {
            *self
        } else {
            self.reverse()
        }
    }

    /// Return a copy of this line with the same endpoints but swapped if
    /// needed so that `end.x >= start.x`.
    pub fn rightwards(&self) -> Line {
        if self.start.x <= self.end.x {
            *self
        } else {
            self.reverse()
        }
    }

    /// Return `(slope, intercept)` tuple for line or None if the line is
    /// vertical.
    fn slope_intercept(&self) -> Option<(f32, f32)> {
        let dx = self.end.x - self.start.x;
        if dx == 0 {
            return None;
        }
        let slope = (self.end.y - self.start.y) as f32 / dx as f32;
        let intercept = self.start.y as f32 - slope * self.start.x as f32;
        Some((slope, intercept))
    }

    /// Return the X coordinate that corresponds to a given Y coordinate on
    /// the line.
    ///
    /// Returns `None` if the Y coordinate is not on the line or the line is
    /// horizontal.
    pub fn x_for_y(&self, y: f32) -> Option<f32> {
        let (min_y, max_y) = sort_pair((self.start.y, self.end.y));
        if y < min_y as f32 || y > max_y as f32 || min_y == max_y {
            return None;
        }
        self.slope_intercept()
            .map(|(slope, intercept)| (y - intercept) / slope)
            .or(Some(self.start.x as f32))
    }

    /// Return the Y coordinate that corresponds to a given X coordinate on
    /// the line.
    ///
    /// Returns `None` if the X coordinate is not on the line or the line
    /// is vertical.
    pub fn y_for_x(&self, x: f32) -> Option<f32> {
        let (min_x, max_x) = sort_pair((self.start.x, self.end.x));
        if x < min_x as f32 || x > max_x as f32 {
            return None;
        }
        self.slope_intercept()
            .map(|(slope, intercept)| slope * x + intercept)
    }
}

impl BoundingRect for Line {
    fn bounding_rect(&self) -> Rect {
        let d = self.downwards();
        let r = self.rightwards();
        Rect::from_tlbr(d.start.y, r.start.x, d.end.y, r.end.x)
    }
}

/// Rectangle defined by left, top, right and bottom integer coordinates.
///
/// The left and top coordinates are inclusive. The right and bottom coordinates
/// are exclusive.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

    /// Return a rect with the given top, left, height and width.
    pub fn from_tlhw(top: Coord, left: Coord, height: Coord, width: Coord) -> Rect {
        Self::from_tlbr(top, left, top + height, left + width)
    }

    pub fn new(top_left: Point, bottom_right: Point) -> Rect {
        Rect {
            top_left,
            bottom_right,
        }
    }

    pub fn area(&self) -> Coord {
        self.width() * self.height()
    }

    pub fn width(&self) -> Coord {
        // TODO - Handle inverted rects here
        self.bottom_right.x - self.top_left.x
    }

    pub fn height(&self) -> Coord {
        // TODO - Handle inverted rects here
        self.bottom_right.y - self.top_left.y
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

    /// Return true if the width or height of this rect are <= 0.
    pub fn is_empty(&self) -> bool {
        self.right() <= self.left() || self.bottom() <= self.top()
    }

    /// Return the center point of the rect.
    pub fn center(&self) -> Point {
        let y = (self.top_left.y + self.bottom_right.y) / 2;
        let x = (self.top_left.x + self.bottom_right.x) / 2;
        Point::from_yx(y, x)
    }

    /// Return the corners of the rect in clockwise order, starting from the
    /// top left.
    pub fn corners(&self) -> [Point; 4] {
        [
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
    }

    /// Return the coordinate of the top-left corner of the rect.
    pub fn top_left(&self) -> Point {
        self.top_left
    }

    /// Return the coordinate of the top-right corner of the rect.
    pub fn top_right(&self) -> Point {
        Point::from_yx(self.top_left.y, self.bottom_right.x)
    }

    /// Return the coordinate of the bottom-left corner of the rect.
    pub fn bottom_left(&self) -> Point {
        Point::from_yx(self.bottom_right.y, self.top_left.x)
    }

    /// Return the coordinate of the bottom-right corner of the rect.
    pub fn bottom_right(&self) -> Point {
        self.bottom_right
    }

    /// Return the line segment of the left edge of the rect.
    pub fn left_edge(&self) -> Line {
        Line::from_endpoints(self.top_left(), self.bottom_left())
    }

    /// Return the line segment of the top edge of the rect.
    pub fn top_edge(&self) -> Line {
        Line::from_endpoints(self.top_left(), self.top_right())
    }

    /// Return the line segment of the right edge of the rect.
    pub fn right_edge(&self) -> Line {
        Line::from_endpoints(self.top_right(), self.bottom_right())
    }

    /// Return the line segment of the bottom edge of the rect.
    pub fn bottom_edge(&self) -> Line {
        Line::from_endpoints(self.bottom_left(), self.bottom_right())
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

    /// Return the top, left, height and width as an array.
    pub fn tlhw(&self) -> [Coord; 4] {
        [
            self.top_left.y,
            self.top_left.x,
            self.height(),
            self.width(),
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

    /// Return true if the intersection of this rect and `other` is non-empty.
    pub fn intersects(&self, other: Rect) -> bool {
        self.left_edge().vertical_overlap(other.left_edge()) > 0
            && self.top_edge().horizontal_overlap(other.top_edge()) > 0
    }

    /// Return the smallest rect that contains both this rect and `other`.
    pub fn union(&self, other: Rect) -> Rect {
        let t = self.top().min(other.top());
        let l = self.left().min(other.left());
        let b = self.bottom().max(other.bottom());
        let r = self.right().max(other.right());
        Rect::from_tlbr(t, l, b, r)
    }

    /// Return the largest rect that is contained within this rect and `other`.
    pub fn intersection(&self, other: Rect) -> Rect {
        let t = self.top().max(other.top());
        let l = self.left().max(other.left());
        let b = self.bottom().min(other.bottom());
        let r = self.right().min(other.right());
        Rect::from_tlbr(t, l, b, r)
    }

    /// Return the Intersection over Union ratio for this rect and `other`.
    ///
    /// See <https://en.wikipedia.org/wiki/Jaccard_index>.
    pub fn iou(&self, other: Rect) -> f32 {
        self.intersection(other).area() as f32 / self.union(other).area() as f32
    }

    /// Return true if `other` lies entirely within this rect.
    pub fn contains(&self, other: Rect) -> bool {
        self.union(other) == *self
    }

    /// Return true if `other` lies on the boundary or interior of this rect.
    pub fn contains_point(&self, other: Point) -> bool {
        self.top() <= other.y
            && self.bottom() >= other.y
            && self.left() <= other.x
            && self.right() >= other.x
    }

    pub fn to_polygon(&self) -> Polygon<[Point; 4]> {
        Polygon::new(self.corners())
    }
}

impl BoundingRect for Rect {
    fn bounding_rect(&self) -> Rect {
        *self
    }
}

/// An oriented rectangle.
#[derive(Copy, Clone, Debug)]
pub struct RotatedRect {
    // Centroid of the rect.
    center: Vec2,

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
    pub fn new(center: Vec2, up_axis: Vec2, width: f32, height: f32) -> RotatedRect {
        RotatedRect {
            center,
            up: up_axis.normalized(),
            width,
            height,
        }
    }

    /// Return the coordinates of the rect's corners.
    ///
    /// The corners are returned in clockwise order starting from the corner
    /// that is the top-left when the "up" axis has XY coordinates [0, 1], or
    /// equivalently, bottom-right when the "up" axis has XY coords [0, -1].
    pub fn corners(&self) -> [Point; 4] {
        let par_offset = self.up.perpendicular() * (self.width / 2.);
        let perp_offset = self.up * (self.height / 2.);

        let coords: [Vec2; 4] = [
            self.center - perp_offset - par_offset,
            self.center - perp_offset + par_offset,
            self.center + perp_offset + par_offset,
            self.center + perp_offset - par_offset,
        ];

        coords.map(|v| Point::from_yx(v.y.round() as i32, v.x.round() as i32))
    }

    /// Return the edges of this rect, in clockwise order starting from the
    /// edge that is the top edge if the rect has no rotation.
    pub fn edges(&self) -> [Line; 4] {
        let corners = self.corners();
        [
            Line::from_endpoints(corners[0], corners[1]),
            Line::from_endpoints(corners[1], corners[2]),
            Line::from_endpoints(corners[2], corners[3]),
            Line::from_endpoints(corners[3], corners[0]),
        ]
    }

    /// Return the centroid of the rect.
    pub fn center(&self) -> Vec2 {
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
    pub fn from_rect(r: Rect) -> RotatedRect {
        let center = Vec2::from_yx(r.center().y as f32, r.center().x as f32);
        RotatedRect::new(
            center,
            Vec2::from_yx(1., 0.),
            r.width() as f32,
            r.height() as f32,
        )
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
    fn bounding_rect(&self) -> Rect {
        let corners = self.corners();

        let mut xs = corners.map(|p| p.x);
        xs.sort();

        let mut ys = corners.map(|p| p.y);
        ys.sort();

        Rect::from_tlbr(ys[0], xs[0], ys[3], xs[3])
    }
}

/// Return the bounding rectangle of a collection of shapes.
///
/// Returns `None` if the collection is empty.
pub fn bounding_rect<'a, Shape: 'a + BoundingRect, I: Iterator<Item = &'a Shape>>(
    objects: I,
) -> Option<Rect> {
    if let Some((min_x, max_x, min_y, max_y)) =
        objects.fold(None as Option<(i32, i32, i32, i32)>, |min_max, shape| {
            let br = shape.bounding_rect();
            min_max
                .map(|(min_x, max_x, min_y, max_y)| {
                    (
                        min_x.min(br.left()),
                        max_x.max(br.right()),
                        min_y.min(br.top()),
                        max_y.max(br.bottom()),
                    )
                })
                .or(Some((br.left(), br.right(), br.top(), br.bottom())))
        })
    {
        Some(Rect::from_tlbr(min_y, min_x, max_y, max_x))
    } else {
        None
    }
}

/// Polygon shape defined by a list of vertices.
///
/// Depending on the storage type `S`, a Polygon can own its vertices
/// (eg. `Vec<Point>`) or they can borrowed (eg. `&[Point]`).
#[derive(Copy, Clone, Debug)]
pub struct Polygon<S: AsRef<[Point]> = Vec<Point>> {
    points: S,
}

impl<S: AsRef<[Point]>> Polygon<S> {
    /// Create a view of a set of points as a polygon.
    pub fn new(points: S) -> Polygon<S> {
        Polygon { points }
    }

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

    /// Return a polygon which borrows its points from this polygon.
    pub fn borrow(&self) -> Polygon<&[Point]> {
        Polygon::new(self.points.as_ref())
    }

    /// Return an iterator over the edges of this polygon.
    pub fn edges(&self) -> impl Iterator<Item = Line> + '_ {
        zip(
            self.points.as_ref().iter(),
            self.points.as_ref().iter().cycle().skip(1),
        )
        .map(|(p0, p1)| Line::from_endpoints(*p0, *p1))
    }

    /// Return a slice of the endpoints of the polygon's edges.
    pub fn vertices(&self) -> &[Point] {
        self.points.as_ref()
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

    /// Return a clone of this polygon which owns its vertices.
    pub fn to_owned(&self) -> Polygon {
        Polygon::new(self.vertices().to_vec())
    }
}

impl<S: AsRef<[Point]>> BoundingRect for Polygon<S> {
    fn bounding_rect(&self) -> Rect {
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;

        for p in self.points.as_ref() {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }

        Rect::from_tlbr(min_y, min_x, max_y, max_x)
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

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::ApproxEq;
    use wasnn_tensor::{MatrixLayout, NdTensor};

    use crate::tests::{points_from_coords, points_from_n_coords};
    use crate::Vec2;

    use super::{bounding_rect, BoundingRect, Line, Point, Polygon, Rect, RotatedRect};

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
                assert_eq!(case.line.y_for_x(x), expected_y);
                if let Some(y) = expected_y {
                    assert_eq!(
                        case.line.x_for_y(y),
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
    fn test_rotated_rect_corners() {
        let r = RotatedRect::new(Vec2::from_yx(5., 5.), Vec2::from_yx(1., 0.), 5., 5.);
        let expected = points_from_n_coords([[3, 3], [3, 8], [8, 8], [8, 3]]);
        assert_eq!(r.corners(), expected);
    }

    #[test]
    fn test_rotated_rect_from_rect() {
        let r = Rect::from_tlbr(5, 10, 50, 40);
        let rr = RotatedRect::from_rect(r);
        assert_eq!(rr.width() as i32, r.width());
        assert_eq!(rr.height() as i32, r.height());
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
                a: RotatedRect::new(Vec2::from_yx(5., 5.), up_vec, 5., 5.),
                b: RotatedRect::new(Vec2::from_yx(5., 5.), up_vec, 5., 5.),
                bounding_rect_intersects: true,
                intersects: true,
            },
            // Separated rects
            Case {
                a: RotatedRect::new(Vec2::from_yx(5., 5.), up_vec, 5., 5.),
                b: RotatedRect::new(Vec2::from_yx(5., 11.), up_vec, 5., 5.),
                bounding_rect_intersects: false,
                intersects: false,
            },
            // Case where bounding rectangles intersect but rotated rects do
            // not.
            Case {
                a: RotatedRect::new(Vec2::from_yx(5., 5.), up_left_vec, 12., 1.),
                b: RotatedRect::new(Vec2::from_yx(9., 9.), up_vec, 1., 1.),
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
        let center = Vec2::from_yx(0., 0.);
        let rect = RotatedRect::new(center, up_axis, 2., 3.);
        assert!(rect.up_axis().length().approx_eq(&1.));
    }

    #[test]
    fn test_rotated_rect_orient_towards() {
        let up_axis = Vec2::from_yx(-1., 0.);
        let center = Vec2::from_yx(0., 0.);
        let rect = RotatedRect::new(center, up_axis, 2., 3.);

        let sorted_corners = |rect: RotatedRect| {
            let mut corners = rect.corners();
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
        let mut r = RotatedRect::new(Vec2::from_yx(5., 5.), Vec2::from_yx(1., 0.), 5., 5.);
        assert_eq!(r.area(), 25.);

        r.resize(3., 7.);

        assert_eq!(r.width(), 3.);
        assert_eq!(r.height(), 7.);
        assert_eq!(r.area(), 21.);
    }
}
