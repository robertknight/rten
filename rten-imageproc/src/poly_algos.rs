use crate::{BoundingRect, Line, PointF, Polygon, RotatedRect, Vec2};

/// Return the sorted subset of points from `poly` that form a convex hull
/// containing `poly`.
pub fn convex_hull(poly: &[PointF]) -> Vec<PointF> {
    // See https://en.wikipedia.org/wiki/Graham_scan

    let mut hull = Vec::new();

    // Find lowest and left-most point, assuming a coordinate system where Y
    // increases going down.
    let min_point = match poly.iter().min_by(|a, b| {
        if a.y != b.y {
            (-a.y).total_cmp(&-b.y)
        } else {
            a.x.total_cmp(&b.x)
        }
    }) {
        Some(p) => *p,
        None => {
            return hull;
        }
    };

    // Compute cosine of angle between the vector `p - min_point` and the X axis.
    let angle = |p: PointF| {
        if p == min_point {
            // Ensure `min_point` appears first in the `sorted_points` list.
            f32::MIN
        } else {
            let x_axis = Vec2::from_yx(0., 1.);
            min_point.vec_to(p).normalized().dot(x_axis)
        }
    };

    // Sort points by angle between `point - min_point` and X axis. When
    // multiple points form the same angle, keep only one furthest from
    // `min_point`.
    let mut sorted_points: Vec<(PointF, f32)> = poly.iter().map(|&p| (p, angle(p))).collect();
    sorted_points.sort_by(|(a_pt, a_angle), (b_pt, b_angle)| {
        if a_angle == b_angle {
            let a_dist = min_point.vec_to(*a_pt).length();
            let b_dist = min_point.vec_to(*b_pt).length();
            a_dist.total_cmp(&b_dist)
        } else {
            a_angle.total_cmp(b_angle)
        }
    });
    sorted_points.dedup_by_key(|(a_point, _)| *a_point);

    // Visit sorted points and keep the sequence that can be followed without
    // making any clockwise turns.
    for &(p, _) in sorted_points.iter() {
        while hull.len() >= 2 {
            let [prev2, prev] = [hull[hull.len() - 2], hull[hull.len() - 1]];
            let ac = prev2.vec_to(p);
            let bc = prev.vec_to(p);
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
    points: &[PointF],
    epsilon: f32,
    out_points: &mut Vec<PointF>,
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
pub fn simplify_polyline(points: &[PointF], epsilon: f32) -> Vec<PointF> {
    assert!(epsilon >= 0.);
    let mut result = Vec::new();
    simplify_polyline_internal(points, epsilon, &mut result, true /* keep_last */);
    result
}

/// Return a simplified version of the polygon defined by `points`.
///
/// This is very similar to [`simplify_polyline`] except that the input is
/// treated as a polygon where the last point implicitly connects to the first
/// point to close the shape.
pub fn simplify_polygon(points: &[PointF], epsilon: f32) -> Vec<PointF> {
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
/// Returns `None` if `points` is empty.
pub fn min_area_rect(points: &[PointF]) -> Option<RotatedRect> {
    // See "Exhaustive Search Algorithm" in
    // https://www.geometrictools.com/Documentation/MinimumAreaRectangle.pdf.

    let hull = convex_hull(points);
    if hull.is_empty() {
        return None;
    }

    let mut min_rect = RotatedRect::from_rect(Polygon::new(&hull).bounding_rect());

    // A hull with one vertex has no edges.
    if hull.len() == 1 {
        return Some(min_rect);
    }

    // Iterate over each edge of the polygon and find the smallest bounding
    // rect where one of the rect's edges aligns with the polygon edge. Keep
    // the rect that has the smallest area over all edges.
    for (&edge_start, &edge_end) in hull.iter().zip(hull.iter().cycle().skip(1)) {
        debug_assert!(
            edge_start != edge_end,
            "hull edges should have non-zero length"
        );

        // Project polygon points onto axes that are parallel and perpendicular
        // to the current edge. The maximum distance between the projected
        // points gives the width and height of the bounding rect.
        let par_axis = edge_start.vec_to(edge_end).normalized();

        // nb. Perpendicular axis points into the hull.
        let perp_axis = -par_axis.perpendicular();

        let (min_par, max_par, max_perp): (f32, f32, f32) = hull.iter().fold(
            (f32::MAX, f32::MIN, f32::MIN),
            |(min_par, max_par, max_perp), point| {
                let d = edge_start.vec_to(*point);
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

        if area < min_rect.area() {
            let center = Vec2::from_yx(edge_start.y, edge_start.x)
                + (par_axis * ((min_par + max_par) / 2.))
                + (perp_axis * (height / 2.));
            min_rect = RotatedRect::new(
                PointF::from_yx(center.y, center.x),
                /* up_axis */ perp_axis,
                width,
                height,
            );
        }
    }

    Some(min_rect)
}

#[cfg(test)]
mod tests {
    use super::{convex_hull, min_area_rect, simplify_polygon, simplify_polyline};
    use crate::tests::{border_points, points_from_coords};
    use crate::{BoundingRect, Point, PointF, Polygon, Rect};

    #[test]
    fn test_convex_hull() {
        struct Case {
            points: &'static [[f32; 2]],
            hull: &'static [[f32; 2]],
        }

        let cases = [
            // Empty polygon
            Case {
                points: &[],
                hull: &[],
            },
            // Single point
            Case {
                points: &[[1., 1.]],
                hull: &[[1., 1.]],
            },
            // Single line
            Case {
                points: &[[1., 1.], [2., 2.]],
                hull: &[[2., 2.], [1., 1.]],
            },
            // Simple square. The hull is a re-ordering of the input.
            Case {
                points: &[[0., 0.], [0., 4.], [4., 4.], [4., 0.]],
                hull: &[[4., 0.], [0., 0.], [0., 4.], [4., 4.]],
            },
            // Square with an indent on each edge. The hull is just a rect.
            Case {
                points: &[
                    // Top
                    [0., 0.],
                    [1., 2.],
                    [0., 4.],
                    // Right
                    [2., 3.],
                    [4., 4.],
                    // Bottom
                    [3., 2.],
                    [4., 0.],
                    // Left
                    [2., 1.],
                ],

                // Hull starts with lowest, left-most corner then proceeds
                // clockwise.
                hull: &[[4., 0.], [0., 0.], [0., 4.], [4., 4.]],
            },
            // Set of equal points
            Case {
                points: &[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                hull: &[[0., 0.]],
            },
            // Three points in a line
            Case {
                points: &[[0., 0.], [1., 1.], [2., 2.]],
                hull: &[[2., 2.], [0., 0.]],
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
            points: Vec<PointF>,
        }

        let cases = [
            // Axis-aligned rect
            Case {
                points: points_from_coords(&[[0., 0.], [0., 4.], [4., 4.], [4., 0.]]),
            },
            // Rotated rect
            Case {
                points: points_from_coords(&[[0., 2.], [2., 4.], [4., 2.], [2., 0.]]),
            },
            // Polygon with all points the same
            Case {
                points: points_from_coords(&[[0., 0.], [0., 0.], [0., 0.], [0., 0.]]),
            },
            // Polygon with an empty edge
            Case {
                points: points_from_coords(&[[0., 0.], [0., 0.], [1., 1.], [2., 2.]]),
            },
            // Single point
            Case {
                points: points_from_coords(&[[5., 5.]]),
            },
            // Empty input
            Case { points: Vec::new() },
        ];

        for case in cases {
            let Some(min_rect) = min_area_rect(&case.points) else {
                assert!(case.points.is_empty());
                continue;
            };

            // Rotated rect should never be larger than axis-aligned bounding rect.
            let bounding_rect = Polygon::new(&case.points).bounding_rect();
            assert!(min_rect.area() <= bounding_rect.area() as f32);

            // Every input point should lie within the rotated rect, otherwise
            // it is too small. Test with a slightly expanded rect to avoid
            // numerical issues with points exactly on the edge.
            let expanded_min_rect = min_rect.expanded(1e-3, 1e-3);
            assert!(case
                .points
                .iter()
                .all(|p| expanded_min_rect.contains(PointF::from_yx(p.y as f32, p.x as f32))));

            // Every edge should touch (within a threshold) an input point,
            // otherwise it is too large.
            let max_dist = min_rect.edges().into_iter().fold(f32::MIN, |dist, edge| {
                let min_dist = case
                    .points
                    .iter()
                    .fold(f32::MAX, |dist, p| edge.distance(*p).min(dist));
                min_dist.max(dist)
            });
            let threshold = 1.;
            assert!(
                max_dist <= threshold,
                "max_dist {} > {}",
                max_dist,
                threshold
            );
        }
    }

    #[test]
    fn test_simplify_polyline() {
        struct Case {
            poly: Vec<PointF>,
            epsilon: f32,
            simplified: Vec<PointF>,
        }

        let cases = [
            // Single point
            Case {
                poly: vec![Point::from_yx(0., 0.)],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(0., 0.)],
            },
            // Line of 2 points
            Case {
                poly: vec![Point::from_yx(5., 2.), Point::from_yx(3., 5.)],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5., 2.), Point::from_yx(3., 5.)],
            },
            // Line of 3 points
            Case {
                poly: vec![
                    Point::from_yx(5., 2.),
                    Point::from_yx(5., 3.),
                    Point::from_yx(5., 4.),
                ],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5., 2.), Point::from_yx(5., 4.)],
            },
            // Line of 4 points
            Case {
                poly: vec![
                    Point::from_yx(5., 2.),
                    Point::from_yx(5., 3.),
                    Point::from_yx(5., 4.),
                    Point::from_yx(5., 5.),
                ],
                epsilon: 0.1,
                simplified: vec![Point::from_yx(5., 2.), Point::from_yx(5., 5.)],
            },
            // Boundary points of rect
            Case {
                poly: border_points(Rect::from_tlbr(4, 4, 9, 9), false /* omit_corners */)
                    .into_iter()
                    .map(|p| Point::from_yx(p.y as f32, p.x as f32))
                    .collect(),
                epsilon: 0.1,
                simplified: [[4., 4.], [8., 4.], [8., 8.], [4., 8.], [4., 5.]]
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
            poly: Vec<PointF>,
            epsilon: f32,
            simplified: Vec<PointF>,
        }

        // Since `simplify_polygon` is a thin wrapper around `simplify_polyline`,
        // so we only have a few cases to cover the differences here.
        let cases = [Case {
            poly: border_points(Rect::from_tlbr(4, 4, 9, 9), false /* omit_corners */)
                .into_iter()
                .map(|p| Point::from_yx(p.y as f32, p.x as f32))
                .collect(),
            epsilon: 0.1,
            simplified: [[4., 4.], [8., 4.], [8., 8.], [4., 8.]]
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
