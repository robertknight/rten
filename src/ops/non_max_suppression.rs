use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::buffer_pool::BufferPool;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, ValueType};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BoxOrder {
    /// Box coordinates are [y1, x1, y2, x2]
    TopLeftBottomRight,

    /// Box coordinates are [center_x, center_y, width, height]
    CenterWidthHeight,
}

#[derive(Debug)]
struct NmsBox {
    /// Top, left, bottom, right coordinates of box.
    tlbr: [f32; 4],

    /// Index of the image in the batch which this box came from.
    batch_index: usize,

    /// Index of this box within the image.
    box_index: usize,

    /// Class with maximum probability score for this box.
    class: usize,

    /// Probability score for `class` for this box.
    score: f32,
}

fn area(tlbr: [f32; 4]) -> f32 {
    let [t, l, b, r] = tlbr;
    let height = (b - t).max(0.);
    let width = (r - l).max(0.);
    height * width
}

impl NmsBox {
    /// Return the Intersection-over-Union score of this box and `other`.
    fn iou(&self, other: &NmsBox) -> f32 {
        let [top, left, bottom, right] = self.tlbr;
        let [other_top, other_left, other_bottom, other_right] = other.tlbr;
        let union_tlbr = [
            top.min(other_top),
            left.min(other_left),
            bottom.max(other_bottom),
            right.max(other_right),
        ];
        let intersection_tlbr = [
            top.max(other_top),
            left.max(other_left),
            bottom.min(other_bottom),
            right.min(other_right),
        ];
        area(intersection_tlbr) / area(union_tlbr)
    }
}

pub fn non_max_suppression(
    pool: &BufferPool,
    boxes: NdTensorView<f32, 3>,
    scores: NdTensorView<f32, 3>,
    box_order: BoxOrder,
    max_output_boxes_per_class: Option<i32>,
    iou_threshold: f32,
    score_threshold: f32,
) -> Result<NdTensor<i32, 2>, OpError> {
    let mut selected = Vec::<NmsBox>::new();
    let [batch, n_boxes, n_coords] = boxes.shape();
    let [scores_batch, n_classes, scores_n_boxes] = scores.shape();

    if n_coords != 4 {
        return Err(OpError::InvalidValue(
            "`boxes` last dimension should have size 4",
        ));
    }

    if batch != scores_batch || n_boxes != scores_n_boxes {
        return Err(OpError::IncompatibleInputShapes(
            "`boxes` and `scores` have incompatible shapes",
        ));
    }

    // Early exit so we can assume `n_classes > 0` in the main loop.
    if n_classes == 0 {
        return Ok(NdTensor::zeros([0, 3]));
    }

    for n in 0..batch {
        for b in 0..n_boxes {
            let (max_score_cls, max_score) = scores
                .slice((n, .., b))
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_cls_a, score_a), (_cls_b, score_b)| score_a.total_cmp(score_b))
                .unwrap();

            if max_score < score_threshold {
                continue;
            }

            let [c0, c1, c2, c3] = boxes.slice((n, b)).to_array();
            let [top, left, bottom, right] = match box_order {
                BoxOrder::TopLeftBottomRight => [c0, c1, c2, c3],
                BoxOrder::CenterWidthHeight => {
                    let [x, y, w, h] = [c0, c1, c2, c3];
                    [y - h / 2., x - w / 2., y + h / 2., x + w / 2.]
                }
            };

            let nms_box = NmsBox {
                tlbr: [top, left, bottom, right],
                batch_index: n,
                box_index: b,
                class: max_score_cls,
                score: max_score,
            };

            let mut keep_box = true;
            let mut i = 0;
            while i < selected.len() {
                let other = &selected[i];

                // TODO - Clarify if IoU comparisons should only be done for
                // boxes in the same class or for the whole box list. ONNX
                // Runtime's CPU implementation of NonMaxSuppression looks
                // like it does per-class. torchvision's `nms` op doesn't
                // take class inputs. Are consumers expected to pass all
                // boxes or call `nms` once per output class?
                if nms_box.iou(other) >= iou_threshold {
                    if nms_box.score > other.score {
                        selected.remove(i);
                    } else {
                        keep_box = false;
                        break;
                    }
                } else {
                    i += 1;
                }
            }

            if keep_box {
                selected.push(nms_box);
            }
        }
    }

    // The ONNX spec does not specify whether outputs should be sorted by score
    // (see https://github.com/onnx/onnx/issues/4414). However torchvision's
    // `nms` op does. See
    // https://pytorch.org/vision/main/generated/torchvision.ops.nms.html.
    selected.sort_by(|box_a, box_b| box_a.score.total_cmp(&box_b.score).reverse());

    // Drop lowest scoring boxes in each class if `max_output_boxes_per_class`
    // is set.
    if let Some(max_output_boxes_per_class) = max_output_boxes_per_class {
        let mut class_counts = vec![0; n_classes];
        selected.retain_mut(|nms_box| {
            if class_counts[nms_box.class] >= max_output_boxes_per_class {
                false
            } else {
                class_counts[nms_box.class] += 1;
                true
            }
        });
    }

    let mut selected_indices = NdTensor::zeros_in(pool, [selected.len(), 3]);
    for (i, nms_box) in selected.into_iter().enumerate() {
        selected_indices.slice_mut(i).assign_array([
            nms_box.batch_index as i32,
            nms_box.class as i32,
            nms_box.box_index as i32,
        ]);
    }

    Ok(selected_indices)
}

#[derive(Debug)]
pub struct NonMaxSuppression {
    pub box_order: BoxOrder,
}

impl Operator for NonMaxSuppression {
    fn name(&self) -> &str {
        "NonMaxSuppression"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(5)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let boxes = inputs.require_as(0)?;
        let scores = inputs.require_as(1)?;

        let max_output_boxes_per_class = inputs.get_as(2)?;
        let iou_threshold = inputs.get_as(3)?;
        let score_threshold = inputs.get_as(4)?;

        let selected_box_indices = non_max_suppression(
            ctx.pool(),
            boxes,
            scores,
            self.box_order,
            max_output_boxes_per_class,
            iou_threshold.unwrap_or(0.),
            score_threshold.unwrap_or(0.),
        )?;

        selected_box_indices.into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Int32))].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;

    use crate::buffer_pool::BufferPool;
    use crate::ops::{BoxOrder, OpError, non_max_suppression};

    struct NmsBox {
        tlbr: [f32; 4],
        class: usize,
        score: f32,
    }

    /// Generate `boxes` and `scores` inputs for `non_max_suppression` from
    /// a list of boxes.
    fn gen_boxes_scores(boxes: &[NmsBox], order: BoxOrder) -> (NdTensor<f32, 3>, NdTensor<f32, 3>) {
        let n_classes = boxes
            .iter()
            .map(|b| b.class)
            .max()
            .map(|max_class| max_class + 1)
            .unwrap_or(0);
        let mut out_boxes = NdTensor::zeros([1, boxes.len(), 4]);
        let mut out_scores = NdTensor::zeros([1, n_classes, boxes.len()]);

        for (i, nms_box) in boxes.iter().enumerate() {
            let coords = match order {
                BoxOrder::TopLeftBottomRight => nms_box.tlbr,
                BoxOrder::CenterWidthHeight => {
                    let [t, l, b, r] = nms_box.tlbr;
                    let cx = (l + r) / 2.;
                    let cy = (t + b) / 2.;
                    let w = r - l;
                    let h = b - t;
                    [cx, cy, w, h]
                }
            };
            out_boxes.slice_mut((0, i)).assign_array(coords);
            out_scores[[0, nms_box.class, i]] = nms_box.score;
        }

        (out_boxes, out_scores)
    }

    fn example_boxes(order: BoxOrder) -> (NdTensor<f32, 3>, NdTensor<f32, 3>) {
        let boxes = [
            // Two overlapping boxes. These overlap at a "medium" threshold of
            // 0.5, but not at a very high threshold (eg. 0.99).
            //
            // The left/right coords don't overlap with the top/bottom coords,
            // to catch mistakes if these are mixed up.
            NmsBox {
                tlbr: [0., 20., 20., 40.],
                class: 0,
                score: 0.8,
            },
            NmsBox {
                tlbr: [2., 22., 22., 42.],
                class: 0,
                score: 0.71,
            },
            // Separated box
            NmsBox {
                tlbr: [200., 0., 100., 100.],
                class: 1,
                score: 0.7,
            },
        ];
        gen_boxes_scores(&boxes, order)
    }

    #[test]
    fn test_non_max_suppression() {
        let (boxes, scores) = example_boxes(BoxOrder::TopLeftBottomRight);
        let iou_threshold = 0.5;
        let score_threshold = 0.;

        let pool = BufferPool::new();
        let selected = non_max_suppression(
            &pool,
            boxes.view(),
            scores.view(),
            BoxOrder::TopLeftBottomRight,
            None, // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        )
        .unwrap();

        assert_eq!(selected.size(0), 2);

        let [batch, class, box_idx] = selected.slice(0).to_array();
        assert_eq!([batch, class, box_idx], [0, 0, 0]);

        let [batch, class, box_idx] = selected.slice(1).to_array();
        assert_eq!([batch, class, box_idx], [0, 1, 2]);
    }

    #[test]
    fn test_non_max_suppression_box_order() {
        let pool = BufferPool::new();

        let (boxes_tlbr, scores) = example_boxes(BoxOrder::TopLeftBottomRight);
        let (boxes_chw, _) = example_boxes(BoxOrder::CenterWidthHeight);

        let iou_threshold = 0.99;
        let score_threshold = 0.;

        let selected_tlbr = non_max_suppression(
            &pool,
            boxes_tlbr.view(),
            scores.view(),
            BoxOrder::TopLeftBottomRight,
            None, // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        );
        let selected_chw = non_max_suppression(
            &pool,
            boxes_chw.view(),
            scores.view(),
            BoxOrder::CenterWidthHeight,
            None, // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        );

        assert_eq!(selected_tlbr, selected_chw);
    }

    #[test]
    fn test_non_max_suppression_iou_threshold() {
        let pool = BufferPool::new();

        let (boxes, scores) = example_boxes(BoxOrder::TopLeftBottomRight);
        let iou_threshold = 0.99;
        let score_threshold = 0.;

        let selected = non_max_suppression(
            &pool,
            boxes.view(),
            scores.view(),
            BoxOrder::TopLeftBottomRight,
            None, // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        )
        .unwrap();

        // Since we set a high IoU, the overlapping boxes for class 0 will be
        // returned.
        assert_eq!(selected.size(0), 3);

        let [batch, class, box_idx] = selected.slice(0).to_array();
        assert_eq!([batch, class, box_idx], [0, 0, 0]);

        let [batch, class, box_idx] = selected.slice(1).to_array();
        assert_eq!([batch, class, box_idx], [0, 0, 1]);
    }

    #[test]
    fn test_non_max_suppression_max_outputs_per_class() {
        let pool = BufferPool::new();
        let (boxes, scores) = example_boxes(BoxOrder::TopLeftBottomRight);
        let iou_threshold = 1.0;
        let score_threshold = 0.;

        let selected = non_max_suppression(
            &pool,
            boxes.view(),
            scores.view(),
            BoxOrder::TopLeftBottomRight,
            Some(1), // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        )
        .unwrap();

        // Even though we requested all exactly overlapping boxes, only one will
        // be returned from each class.
        assert!(selected.size(0) == 2);

        let [batch, class, box_idx] = selected.slice(0).to_array();
        assert_eq!([batch, class, box_idx], [0, 0, 0]);

        let [batch, class, box_idx] = selected.slice(1).to_array();
        assert_eq!([batch, class, box_idx], [0, 1, 2]);
    }

    #[test]
    fn test_non_max_suppression_score_threshold() {
        let pool = BufferPool::new();
        let (boxes, scores) = example_boxes(BoxOrder::TopLeftBottomRight);
        let iou_threshold = 0.5;
        let score_threshold = 0.8;

        let selected = non_max_suppression(
            &pool,
            boxes.view(),
            scores.view(),
            BoxOrder::TopLeftBottomRight,
            None, // max_output_boxes_per_class
            iou_threshold,
            score_threshold,
        )
        .unwrap();

        // Only the box with score exceeding `score_threshold` will be returned.
        assert!(selected.size(0) == 1);

        let [batch, class, box_idx] = selected.slice(0).to_array();
        assert_eq!([batch, class, box_idx], [0, 0, 0]);
    }

    #[test]
    fn test_non_max_suppression_invalid() {
        let pool = BufferPool::new();
        let apply_nms = |boxes, scores| {
            let iou_threshold = 0.5;
            let score_threshold = 0.;
            non_max_suppression(
                &pool,
                boxes,
                scores,
                BoxOrder::TopLeftBottomRight,
                None, // max_output_boxes_per_class
                iou_threshold,
                score_threshold,
            )
        };

        let n_boxes = 10;
        let n_classes = 10;
        let boxes = NdTensor::zeros([1, n_boxes, 4]);
        let scores = NdTensor::zeros([1, n_classes, n_boxes + 1]);

        let result = apply_nms(boxes.view(), scores.view());
        assert_eq!(
            result,
            Err(OpError::IncompatibleInputShapes(
                "`boxes` and `scores` have incompatible shapes"
            ))
        );

        let boxes = NdTensor::zeros([1, n_boxes, 3]);
        let scores = NdTensor::zeros([1, n_classes, n_boxes]);
        let result = apply_nms(boxes.view(), scores.view());
        assert_eq!(
            result,
            Err(OpError::InvalidValue(
                "`boxes` last dimension should have size 4"
            ))
        );
    }
}
