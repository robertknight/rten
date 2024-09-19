#[allow(unused_imports)]
use rten_tensor::prelude::*;
use rten_tensor::{MatrixLayout, NdTensor, NdTensorView};

use crate::{Point, Polygons};

enum Direction {
    Clockwise,
    CounterClockwise,
}

/// Search the neighborhood of the pixel `center` in `mask` for a pixel with
/// a non-zero value, starting from `start` and in the order given by `dir`.
///
/// If `skip_first` is true, start the search from the next neighbor of `start`
/// in the order given by `dir`.
fn find_nonzero_neighbor<
    T: Default + std::cmp::PartialEq,
    // Use a generic for `mask` rather than `NdTensorView` to avoid the (small)
    // overhead of repeated view creation.
    M: std::ops::Index<[usize; 2], Output = T>,
>(
    mask: &M,
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
        if mask[neighbors[idx].coord()] != T::default() {
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
pub fn find_contours(mask: NdTensorView<bool, 2>, mode: RetrievalMode) -> Polygons {
    // Create a copy of the mask with zero-padding around the border and i8
    // type. The padding enables the algorithm to handle objects that touch the
    // edge of the mask. The i8 type is needed because the algorithm needs to
    // store one of 3 values in each element.
    let padding = 1;
    let mut padded_mask =
        NdTensor::<i8, 2>::zeros([mask.rows() + 2 * padding, mask.cols() + 2 * padding]);

    // Use faster indexing (but with weaker bounds checks).
    let wc_mask = mask.weakly_checked_view();
    let mut wc_padded_mask = padded_mask.weakly_checked_view_mut();
    for y in 0..mask.rows() {
        for x in 0..mask.cols() {
            // Clamp values in the copied mask to { 0, 1 } so the algorithm
            // below can use other values as part of its working.
            wc_padded_mask[[y + padding, x + padding]] = wc_mask[[y, x]] as i8;
        }
    }
    let mut mask = padded_mask;

    let mut contours = Polygons::new();

    // Points of current border.
    let mut border = Vec::new();

    // Label for pixels that have been marked as part of a border. The pixels are
    // labeled with a positive or negative value depending on which side of the
    // border it is on.
    //
    // In the paper this is called `NBD` and is incremented for each border. We
    // don't increment the value because this is only needed as part of finding
    // the hierarchical structure of borders, which is not implemented here.
    // Instead we only need to distinguish background (0), object (1) and border
    // (+/- 2) pixels. Using an `i8` label reduces memory consumption for the
    // working space and speeds up finding contours in large images.
    let border_num: i8 = 2;

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
                border.clear();

                let nonzero_start_neighbor = find_nonzero_neighbor(
                    &mask,
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
                            &mask,
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

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use crate::tests::border_points;
    use crate::{fill_rect, stroke_rect, Point, Rect};

    use super::{find_contours, RetrievalMode};

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
        }

        let cases = [Case {
            rect: Rect::from_tlbr(5, 5, 10, 10),
        }];

        for case in cases {
            let mut mask = NdTensor::zeros([20, 20]);
            fill_rect(mask.view_mut(), case.rect, true);

            let contours = find_contours(mask.view(), RetrievalMode::List);

            assert_eq!(contours.len(), 1);
            let border = contours.iter().next().unwrap();
            assert_eq!(border, border_points(case.rect, false /* omit_corners */));
        }
    }

    #[test]
    fn test_find_contours_rect_touching_frame() {
        let mut mask = NdTensor::zeros([5, 5]);
        let rect = Rect::from_tlbr(0, 0, 5, 5);
        fill_rect(mask.view_mut(), rect, true);

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), 1);

        let border = contours.iter().next().unwrap();
        assert_eq!(border, border_points(rect, false /* omit_corners */));
    }

    #[test]
    fn test_find_contours_hollow_rect() {
        let mut mask = NdTensor::zeros([20, 20]);
        let rect = Rect::from_tlbr(5, 5, 12, 12);
        stroke_rect(mask.view_mut(), rect, true, 2);

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
        stroke_rect(mask.view_mut(), rect, true, 2);

        let contours = find_contours(mask.view(), RetrievalMode::External);

        // There should only be one, outermost contour.
        assert_eq!(contours.len(), 1);
        let outer_border = contours.iter().next().unwrap();
        assert_eq!(outer_border, border_points(rect, false /* omit_corners */));
    }

    #[test]
    fn test_find_contours_single_point() {
        let mut mask = NdTensor::zeros([20, 20]);
        mask[[5, 5]] = true;

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
            fill_rect(mask.view_mut(), rect, true);
        }

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in contours.iter().zip(rects) {
            assert_eq!(border, border_points(rect, false /* omit_corners */));
        }
    }

    #[test]
    fn test_find_contours_nested_rects() {
        let mut mask = NdTensor::zeros([15, 15]);

        let rects = [Rect::from_tlbr(5, 5, 11, 11), Rect::from_tlbr(7, 7, 9, 9)];
        for rect in rects {
            stroke_rect(mask.view_mut(), rect, true, 1);
        }

        let contours = find_contours(mask.view(), RetrievalMode::List);
        assert_eq!(contours.len(), rects.len());

        for (border, rect) in contours.iter().zip(rects) {
            assert_eq!(border, border_points(rect, false /* omit_corners */));
        }
    }

    /// Generate a grid of uniformly sized and spaced rects.
    ///
    /// `grid_shape` is a (rows, columns) tuple. `rect_size` and `gap_size` are
    /// (height, width) tuples.
    fn gen_rect_grid(
        top_left: Point,
        grid_shape: (i32, i32),
        rect_size: (i32, i32),
        gap_size: (i32, i32),
    ) -> Vec<Rect> {
        let mut rects = Vec::new();

        let (rows, cols) = grid_shape;
        let (rect_h, rect_w) = rect_size;
        let (gap_h, gap_w) = gap_size;

        for r in 0..rows {
            for c in 0..cols {
                let top = top_left.y + r * (rect_h + gap_h);
                let left = top_left.x + c * (rect_w + gap_w);
                rects.push(Rect::from_tlbr(top, left, top + rect_h, left + rect_w))
            }
        }

        rects
    }

    #[test]
    #[ignore]
    fn bench_find_contours() {
        use rten_bench::run_bench;

        // Fill a mask with a grid of rectangular objects.
        let mask_h = 1024;
        let mask_w = 1024;
        let row_gap = 5;
        let col_gap = 5;
        let grid_cols = 20;
        let grid_rows = 40;

        let mut mask = NdTensor::zeros([mask_h, mask_w]);
        let item_h = (mask_h / grid_rows) - row_gap;
        let item_w = (mask_w / grid_cols) - col_gap;

        let rects = gen_rect_grid(
            Point::from_yx(0, 0),
            (grid_rows as i32, grid_cols as i32),
            (item_h as i32, item_w as i32),
            (row_gap as i32, col_gap as i32),
        );

        for rect in rects {
            fill_rect(mask.view_mut(), rect, true);
        }

        let n_iters = 100;
        run_bench(n_iters, Some("find_contours"), || {
            let contours = find_contours(mask.view(), RetrievalMode::External);
            assert_eq!(contours.len(), (grid_rows * grid_cols) as usize);
        });
    }
}
