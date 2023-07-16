use std::iter::zip;

use crate::{Line, Point, RotatedRect, Vec2};

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

#[cfg(test)]
mod tests {
    use crate::{Point, Rect};

    use crate::tests::{border_points, points_from_coords, points_from_n_coords};

    use super::{convex_hull, min_area_rect, simplify_polygon, simplify_polyline};

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
