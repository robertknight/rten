use std::iter::zip;

use wasnn::Model;
use wasnn_imageproc::{
    bounding_rect, find_contours, min_area_rect, simplify_polygon, BoundingRect, Coord, Line,
    LineF, Point, PointF, Rect, RectF, RetrievalMode, RotatedRect,
};
use wasnn_tensor::{Layout, NdTensor, NdTensorCommon, NdTensorView, Tensor, TensorCommon};

use crate::empty_rects::{max_empty_rects, FilterOverlapping};
use crate::xy_tree::{PartitionOpts, XyNode};

fn rects_separated_by_line(a: &RotatedRect, b: &RotatedRect, l: LineF) -> bool {
    let a_to_b = LineF::from_endpoints(a.center(), b.center());
    a_to_b.intersects(l)
}

fn rightmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    Line::from_endpoints(corners[2], corners[3])
}

fn leftmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    Line::from_endpoints(corners[0], corners[1])
}

/// Group rects into lines. Each line is a chain of oriented rects ordered
/// left-to-right.
///
/// `separators` is a list of line segments that prevent the formation of
/// lines which cross them. They can be used to specify column boundaries
/// for example.
pub fn group_into_lines(rects: &[RotatedRect], separators: &[LineF]) -> Vec<Vec<RotatedRect>> {
    let mut sorted_rects: Vec<_> = rects.to_vec();
    sorted_rects.sort_by_key(|r| r.bounding_rect().left() as i32);

    let mut lines: Vec<Vec<_>> = Vec::new();

    // Minimum amount by which two words must overlap vertically to be
    // considered part of the same line.
    let overlap_threshold = 5;

    // Maximum amount by which a candidate word to extend a line from
    // left-to-right may horizontally overlap the current last word in the line.
    //
    // This is necessary when the code that produces the input rects can create
    // overlapping rects. `find_connected_component_rects` pads the rects it
    // produces for example.
    let max_h_overlap = 5;

    while !sorted_rects.is_empty() {
        let mut line = Vec::new();
        line.push(sorted_rects.remove(0));

        // Find the best candidate to extend the current line by one word to the
        // right, and keep going as long as we can find such a candidate.
        loop {
            let last = line.last().unwrap();
            let last_edge = rightmost_edge(last);

            if let Some((i, next_item)) = sorted_rects
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    let edge = leftmost_edge(r);
                    r.center().x > last.center().x
                        && edge.center().x - last_edge.center().x >= -max_h_overlap as f32
                        && last_edge.vertical_overlap(edge) >= overlap_threshold as f32
                        && !separators
                            .iter()
                            .any(|&s| rects_separated_by_line(last, r, s))
                })
                .min_by_key(|(_, r)| r.center().x as i32)
            {
                line.push(*next_item);
                sorted_rects.remove(i);
            } else {
                break;
            }
        }
        lines.push(line);
    }

    lines
}

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
pub fn find_connected_component_rects(
    mask: NdTensorView<i32, 2>,
    expand_dist: f32,
) -> Vec<RotatedRect> {
    // Threshold for the minimum area of returned rectangles.
    //
    // This can be used to filter out rects created by small false positives in
    // the mask, at the risk of filtering out true positives. The more accurate
    // the model producing the mask is, the less this is needed.
    let min_area_threshold = 100.;

    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .filter(|r| r.area() >= min_area_threshold)
        .collect()
}

/// Find separators between text blocks.
///
/// This includes separators between columns, as well as between sections (eg.
/// headings and article contents).
pub fn find_block_separators(words: &[RotatedRect]) -> Vec<Rect> {
    let Some(page_rect) = bounding_rect(words.iter()).map(|br| br.integral_bounding_rect()) else {
        return Vec::new();
    };

    // Estimate spacing statistics
    let mut lines = group_into_lines(words, &[]);
    lines.sort_by_key(|l| l.first().unwrap().bounding_rect().top().round() as i32);

    let mut all_word_spacings = Vec::new();
    for line in lines.iter() {
        if line.len() > 1 {
            let mut spacings: Vec<_> = zip(line.iter(), line.iter().skip(1))
                .map(|(cur, next)| {
                    (next.bounding_rect().left() - cur.bounding_rect().right()).round() as i32
                })
                .collect();
            spacings.sort();
            all_word_spacings.extend_from_slice(&spacings);
        }
    }
    all_word_spacings.sort();

    let median_word_spacing = all_word_spacings
        .get(all_word_spacings.len() / 2)
        .copied()
        .unwrap_or(10);
    let median_height = words
        .get(words.len() / 2)
        .map(|r| r.height())
        .unwrap_or(10.)
        .round() as i32;

    // Scoring function for empty rectangles. Taken from Section 3.D in [1].
    // This favors tall rectangles.
    //
    // [1] F. Shafait, D. Keysers and T. Breuel, "Performance Evaluation and
    //     Benchmarking of Six-Page Segmentation Algorithms".
    //     10.1109/TPAMI.2007.70837.
    let score = |r: Rect| {
        let aspect_ratio = (r.height() as f32) / (r.width() as f32);
        let aspect_ratio_weight = match aspect_ratio.log2().abs() {
            r if r < 3. => 0.5,
            r if r < 5. => 1.5,
            r => r,
        };
        ((r.area() as f32) * aspect_ratio_weight).sqrt()
    };

    // Find separators between columns and articles.
    let object_bboxes: Vec<_> = words
        .iter()
        .map(|r| r.bounding_rect().integral_bounding_rect())
        .collect();
    let min_width = (median_word_spacing * 3) / 2;
    let min_height = (3 * median_height.max(0)) as u32;

    max_empty_rects(
        &object_bboxes,
        page_rect,
        score,
        min_width.try_into().unwrap(),
        min_height,
    )
    .filter_overlapping(0.5)
    .take(80)
    .collect()
}

/// A collection of text lines.
#[derive(Clone, Debug)]
pub struct Paragraph {
    lines: Vec<Vec<RotatedRect>>,
}

impl BoundingRect for Paragraph {
    type Coord = f32;

    fn bounding_rect(&self) -> RectF {
        bounding_rect(self.words()).expect("paragraph should be non-empty")
    }
}

impl Paragraph {
    /// Return an iterator over all text lines in the paragraph.
    pub fn lines(&self) -> impl Iterator<Item = &[RotatedRect]> {
        self.lines.iter().map(|line| line.as_slice())
    }

    /// Return an iterator over all text words in the paragraph.
    pub fn words(&self) -> impl Iterator<Item = &RotatedRect> {
        self.lines().flatten()
    }
}

/// Describes the hierarhical layout of text in an image in terms of
/// paragraphs, lines and words, arranged in reading order.
pub struct PageLayout {
    paragraphs: Vec<Paragraph>,
}

impl PageLayout {
    /// Return an iterator over all paragraphs in the page, in reading order.
    pub fn paragraphs(&self) -> impl Iterator<Item = &Paragraph> {
        self.paragraphs.iter()
    }

    /// Return an iterator over all lines in the page, in reading order.
    pub fn lines(&self) -> impl Iterator<Item = &[RotatedRect]> {
        self.paragraphs.iter().flat_map(|p| p.lines())
    }

    /// Return an iterator over all words in the page, in reading order.
    pub fn words(&self) -> impl Iterator<Item = &RotatedRect> {
        self.lines().flatten()
    }
}

/// Perform layout analysis of the unordered set of words on a page to organize
/// them into a collection of higher-level structures (lines and paragraphs),
/// sorted into reading order.
pub fn analyze_layout(words: &[RotatedRect], layout_model: Option<&Model>) -> PageLayout {
    if let Some(model) = layout_model {
        // TESTING - Run words through model
        println!("n_words {}", words.len());

        let n_features = 6;

        // FIXME - Make the word count dynamic in the model.
        let n_words = 378;

        let mut word_features = NdTensor::zeros([1, n_words, n_features]);
        for (word_idx, word_rect) in words.iter().enumerate().take(n_words) {
            let word_br = word_rect.bounding_rect();
            let mut word_features = word_features.slice_mut([0, word_idx]);
            word_features[[0]] = word_br.left();
            word_features[[1]] = word_br.top();
            word_features[[2]] = word_br.right();
            word_features[[3]] = word_br.bottom();
            word_features[[4]] = word_br.width();
            word_features[[5]] = word_br.height();
        }

        let input_id = model
            .input_ids()
            .first()
            .copied()
            .expect("model has no inputs");
        let output_id = model
            .output_ids()
            .first()
            .copied()
            .expect("model has no outputs");
        let word_features_dyn: Tensor<f32> = word_features.into();

        let outputs = model
            .run(
                &[(input_id, (&word_features_dyn).into())],
                &[output_id],
                None,
            )
            .expect("model run failed");

        let word_labels: NdTensorView<f32, 3> = outputs[0].as_float_ref().unwrap().nd_view();
        println!("Output shape {:?}", word_labels.shape());

        let line_start_probs: NdTensorView<_, 1> = word_labels.slice((0, .., 0));
        let line_end_probs: NdTensorView<_, 1> = word_labels.slice((0, .., 1));

        let threshold = 0.5;
        let prob_to_class = |prob: &f32| if *prob > threshold { 1i32 } else { 0 };
        let n_line_starts: i32 = line_start_probs.map(prob_to_class).iter().sum();
        let n_line_ends: i32 = line_end_probs.map(prob_to_class).iter().sum();

        println!(
            "Total words {} line starts {} line ends {}",
            words.len(),
            n_line_starts,
            n_line_ends
        );
    }

    let separators = find_block_separators(words);
    let vertical_separators: Vec<_> = separators
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(r.top(), center.x).to_f32(),
                Point::from_yx(r.bottom(), center.x).to_f32(),
            )
        })
        .collect();

    let mut lines = group_into_lines(words, &vertical_separators);

    // Approximate a text line by the 1D line from the center of the left
    // edge of the first word, to the center of the right edge of the last word.
    let midpoint_line = |words: &[RotatedRect]| -> LineF {
        assert!(!words.is_empty());
        Line::from_endpoints(
            words.first().unwrap().bounding_rect().left_edge().center(),
            words.last().unwrap().bounding_rect().right_edge().center(),
        )
    };

    // Sort lines by vertical position.
    lines.sort_by_key(|words| midpoint_line(words).center().y as i32);

    #[derive(Clone)]
    struct LineIndex<'a> {
        lines: &'a [Vec<RotatedRect>],
        index: u32,
    }

    impl<'a> LineIndex<'a> {
        fn line(&self) -> &Vec<RotatedRect> {
            &self.lines[self.index as usize]
        }
    }

    impl<'a> BoundingRect for LineIndex<'a> {
        type Coord = f32;

        fn bounding_rect(&self) -> RectF {
            bounding_rect(self.lines[self.index as usize].iter()).unwrap()
        }
    }

    let line_indices: Vec<_> = lines
        .iter()
        .enumerate()
        .map(|(idx, _)| LineIndex {
            lines: &lines,
            index: idx as u32,
        })
        .collect();

    let mut paragraphs: Vec<Paragraph> = Vec::new();
    let xy_tree = XyNode::partition(
        &line_indices,
        &PartitionOpts {
            min_horizontal_gap: 5.,
            min_vertical_gap: 10.,
        },
    );

    xy_tree.visit_leaves(&mut |line_indices| {
        let lines: Vec<Vec<RotatedRect>> = line_indices
            .iter()
            .map(|li| {
                let words = li.line().to_vec();
                words
            })
            .collect();
        paragraphs.push(Paragraph { lines });
    });

    PageLayout { paragraphs }
}

/// Normalize a line so that it's endpoints are sorted from top to bottom.
fn downwards_line<T: Coord>(l: Line<T>) -> Line<T> {
    if l.start.y <= l.end.y {
        l
    } else {
        Line::from_endpoints(l.end, l.start)
    }
}

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
pub fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();

    let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(left.start));
        polygon.push(floor_point(right.start));
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(right.end));
        polygon.push(floor_point(left.end));
    }

    polygon
}

#[cfg(test)]
mod tests {
    use wasnn_imageproc::{
        fill_rect, BoundingRect, Point, Polygon, Rect, RectF, RotatedRect, Vec2,
    };
    use wasnn_tensor::NdTensor;

    use crate::page_layout::{analyze_layout, find_connected_component_rects, line_polygon};
    use crate::tests::{gen_rect_grid, union_rects};

    #[test]
    fn test_find_connected_component_rects() {
        let mut mask = NdTensor::zeros([400, 400]);
        let (grid_h, grid_w) = (5, 5);
        let (rect_h, rect_w) = (10, 50);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (grid_h, grid_w), /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            // Expand `r` because `fill_rect` does not set points along the
            // right/bottom boundary.
            let expanded = r.adjust_tlbr(0, 0, 1, 1);
            fill_rect(mask.view_mut(), expanded, 1);
        }

        let components = find_connected_component_rects(mask.view(), 0.);
        assert_eq!(components.len() as i32, grid_h * grid_w);
        for c in components.iter() {
            let mut shape = [c.height().round() as i32, c.width().round() as i32];
            shape.sort();

            // We sort the dimensions before comparison here to be invariant to
            // different rotations of the connected component that cover the
            // same pixels.
            let mut expected_shape = [rect_h, rect_w];
            expected_shape.sort();

            assert_eq!(shape, expected_shape);
        }
    }

    #[test]
    fn test_analyze_layout() {
        // Create a collection of obstacles that are laid out roughly like
        // words in a two-column document.
        let page = Rect::from_tlbr(0, 0, 80, 90);
        let col_rows = 10;
        let col_words = 5;
        let (line_gap, word_gap) = (3, 2);
        let (word_h, word_w) = (5, 5);

        let left_col = gen_rect_grid(
            Point::from_yx(0, 0),
            /* grid_shape */ (col_rows, col_words),
            /* rect_size */ (word_h, word_w),
            /* gap_size */ (line_gap, word_gap),
        );
        let left_col_boundary = union_rects(&left_col).unwrap();
        assert!(page.contains(left_col_boundary));

        let right_col = gen_rect_grid(
            Point::from_yx(0, left_col_boundary.right() + 20),
            /* grid_shape */ (col_rows, col_words),
            /* rect_size */ (word_h, word_w),
            /* gap_size */ (line_gap, word_gap),
        );
        let right_col_boundary = union_rects(&right_col).unwrap();
        assert!(page.contains(right_col_boundary));

        let mut words: Vec<_> = left_col
            .iter()
            .chain(right_col.iter())
            .copied()
            .map(|r| RotatedRect::from_rect(r.to_f32()))
            .collect();

        let rng = fastrand::Rng::with_seed(1234);
        rng.shuffle(&mut words);
        let layout = analyze_layout(&words);

        assert_eq!(layout.lines().count() as i32, col_rows * 2);
        for line in layout.lines() {
            assert_eq!(line.len() as i32, col_words);

            let bounding_rect: Option<RectF> = line.iter().fold(None, |br, r| match br {
                Some(br) => Some(br.union(r.bounding_rect())),
                None => Some(r.bounding_rect()),
            });
            let (line_height, line_width) = bounding_rect
                .map(|br| (br.height(), br.width()))
                .unwrap_or((0., 0.));

            // FIXME - The actual width/heights vary by one pixel and hence not
            // all match the expected size. Investigate why this happens.
            assert!((line_height - word_h as f32).abs() <= 1.);
            let expected_width = col_words * (word_w + word_gap) - word_gap;
            assert!((line_width - expected_width as f32).abs() <= 1.);
        }
    }

    #[test]
    fn test_line_polygon() {
        let words: Vec<RotatedRect> = (0..5)
            .map(|i| {
                let center = Point::from_yx(10., i as f32 * 20.);
                let width = 10.;
                let height = 5.;

                // Vary the orientation of words. The output of `line_polygon`
                // should be invariant to different orientations of a RotatedRect
                // that cover the same pixels.
                let up = if i % 2 == 0 {
                    Vec2::from_yx(-1., 0.)
                } else {
                    Vec2::from_yx(1., 0.)
                };
                RotatedRect::new(center, up, width, height)
            })
            .collect();
        let poly = Polygon::new(line_polygon(&words));

        assert!(poly.is_simple());
        for word in words {
            let center = word.bounding_rect().center();
            assert!(poly.contains_pixel(Point::from_yx(
                center.y.round() as i32,
                center.x.round() as i32
            )));
        }
    }
}
