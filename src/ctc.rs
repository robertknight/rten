//! Connectionist Temporal Classification (CTC) sequence decoding tools.

use std::collections::HashMap;
use std::num::NonZeroU32;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

use crate::Operators;

/// Connectionist Temporal Classification (CTC) [^1] [^2] sequence decoder.
///
/// The decoder takes an input of shape `[sequence, n_labels]`, where the values
/// are log probabilities of a label, and infers the most likely sequence of
/// output class labels. The label 0 is reserved for the CTC blank label.
///
/// Different decoding methods are available. Greedy decoding with
/// [`CtcDecoder::decode_greedy`] is very fast, but considers only the most likely
/// label at each time step. Beam searches with [`CtcDecoder::decode_beam`]
/// consider the N most probable paths through the matrix. This may produce more
/// accurate results, but is significantly slower.
///
/// [^1]: <https://en.wikipedia.org/wiki/Connectionist_temporal_classification>
///
/// [^2]: <https://distill.pub/2017/ctc/>
pub struct CtcDecoder {}

/// Item in an output sequence produced by [`CtcDecoder`].
#[derive(Clone, Copy, Debug)]
pub struct DecodeStep {
    /// Class label.
    pub label: u32,

    /// Position in the input sequence that corresponds to this label.
    ///
    /// CTC decoding skips blanks and repeated labels. If a label is repeated,
    /// the position will correspond to the first occurence in the sequence. eg.
    /// If the most likely output sequence is "a--bb", the output labels and
    /// their positions will be `('a', 0), ('b', 3)`.
    pub pos: u32,
}

/// A search state for beam decoding by [`CtcDecoder`]. This consists of a
/// decoded sequence and associated probabilities.
#[derive(Debug)]
struct BeamState {
    /// Label sequence and associated positions for this state.
    prefix: Vec<DecodeStep>,

    /// Log probability of prefix, followed by one or more blanks.
    prob_blank: f32,

    /// Log probability of prefix not followed by a blank.
    prob_no_blank: f32,
}

/// Compute the sum of probabilities in log space.
///
/// In other words, this computes:
///
/// ```text
/// log(exp(log_probs[0]) + exp(log_probs[1]) ...)
/// ```
///
/// The implementation follows `torch.logsumexp` in PyTorch and should produce
/// the same results.
fn log_sum_exp<const N: usize>(log_probs: [f32; N]) -> f32 {
    // Handle all-infinity case separately to avoid nan result.
    if log_probs.iter().all(|&x| x == f32::NEG_INFINITY) {
        f32::NEG_INFINITY
    } else {
        let lp_max = log_probs
            .into_iter()
            .reduce(f32::max)
            .unwrap_or(f32::NEG_INFINITY);
        lp_max
            + log_probs
                .iter()
                .map(|x| (x - lp_max).exp())
                .sum::<f32>()
                .ln()
    }
}

/// Result of decoding a sequence using [`CtcDecoder`].
///
/// This consists of a sequence of class labels and a score.
#[derive(Clone, Debug)]
pub struct CtcHypothesis {
    steps: Vec<DecodeStep>,
    score: f32,
}

impl CtcHypothesis {
    fn new(steps: Vec<DecodeStep>, score: f32) -> CtcHypothesis {
        CtcHypothesis { steps, score }
    }

    fn from_beam_state(state: BeamState) -> CtcHypothesis {
        Self::new(
            state.prefix,
            log_sum_exp([state.prob_blank, state.prob_no_blank]),
        )
    }

    /// Convert the label sequence to a string, using the given alphabet.
    pub fn to_string(&self, alphabet: &str) -> String {
        self.steps()
            .iter()
            .map(|step| {
                alphabet
                    .chars()
                    .nth((step.label - 1) as usize)
                    .unwrap_or('?')
            })
            .collect()
    }

    /// Return the score of this hypothesis, as a log probability.
    ///
    /// For hypotheses produced by greedy decoding, this is the product of
    /// probabilities of the most likely label at each time step. For beam
    /// search decoding, this is the sum of probabilities of all paths that
    /// produce this hypothesis's label sequence.
    ///
    /// This score is not normalized by the input length, so longer input
    /// sequences will tend to lead to lower scores.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Return the sequence of labels and associated input positions.
    pub fn steps(&self) -> &[DecodeStep] {
        self.steps.as_ref()
    }
}

impl CtcDecoder {
    pub fn new() -> CtcDecoder {
        CtcDecoder {}
    }

    /// Decode sequence using a greedy method.
    ///
    /// This method chooses the label with the highest probability at each
    /// time step. This method is very fast, but may return less accurate
    /// results than [`CtcDecoder::decode_beam`].
    ///
    /// `prob_seq` is a `[sequence, n_labels]` matrix of log probabilities of
    /// labels at each time step, where the label value 0 is reserved for the
    /// CTC blank label.
    pub fn decode_greedy(&self, prob_seq: NdTensorView<f32, 2>) -> CtcHypothesis {
        let label_seq = prob_seq
            .arg_max(/* axis */ 1, /* keep_dims */ false)
            .expect("argmax failed");

        let mut last_label = 0;
        let mut steps = Vec::new();
        let mut score = 0.;

        for (pos, label) in label_seq.iter().copied().enumerate() {
            score += prob_seq[[pos, label as usize]];

            if label == last_label {
                continue;
            }
            last_label = label;

            if label > 0 {
                steps.push(DecodeStep {
                    label: label as u32,
                    pos: pos as u32,
                })
            }
        }

        CtcHypothesis::new(steps, score)
    }

    /// Decode sequence using a beam search and return the N best hypotheses.
    ///
    /// See also [`CtcDecoder::decode_beam`].
    pub fn decode_beam_nbest(
        &self,
        prob_seq: NdTensorView<f32, 2>,
        beam_size: u32,
        n_best: u32,
    ) -> Vec<CtcHypothesis> {
        self.decode_beam_impl(prob_seq, beam_size)
            .into_iter()
            .take(n_best as usize)
            .map(CtcHypothesis::from_beam_state)
            .collect()
    }

    /// Decode sequence using a beam search and return the best hypothesis.
    ///
    /// This method retains the `beam_size` best hypotheses after each decoding
    /// step and discards the rest.
    ///
    /// `prob_seq` is a `[sequence, n_labels]` matrix of log probabilities of
    /// labels at each time step, where the label value 0 is reserved for the
    /// CTC blank label. `beam_size` is the maximum number of hyptheses to
    /// keep after each step. Higher values may produce more accurate results,
    /// but will make decoding slower.
    ///
    /// The implementation was originally based on
    /// <https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0>. See also
    /// "Inference" section of <https://distill.pub/2017/ctc/> for an explanation
    /// of the algorithm.
    pub fn decode_beam(&self, prob_seq: NdTensorView<f32, 2>, beam_size: u32) -> CtcHypothesis {
        CtcHypothesis::from_beam_state(self.decode_beam_impl(prob_seq, beam_size).remove(0))
    }

    fn decode_beam_impl(&self, prob_seq: NdTensorView<f32, 2>, beam_size: u32) -> Vec<BeamState> {
        let [seq, n_labels] = prob_seq.shape();

        // Current beam states and their log probabilities, sorted by descending
        // total probability.
        //
        // Each state in the beam should have a unique prefix after each step.
        let mut beam = vec![BeamState {
            prefix: Vec::new(),
            prob_blank: 0.,
            prob_no_blank: f32::NEG_INFINITY,
        }];

        // Probabilities for extensions to beam. The label 0 is used to mean
        // keeping the beam's prefix unchanged.
        let mut next_prob_blank = NdTensor::zeros([beam_size as usize, n_labels]);
        let mut next_prob_no_blank = NdTensor::zeros([beam_size as usize, n_labels]);

        // Map of `(beam_index, label) => other_beam_index` for
        // extensions to current prefixes which will produce a prefix that
        // matches an existing beam state.
        let mut merges: HashMap<(usize, u32), usize> = HashMap::new();

        // Top-K beam extensions, sorted by probability descending.
        struct BeamExtension {
            /// Index of beam state to extend.
            index: u32,

            /// Label to extend prefix with.
            label: Option<NonZeroU32>,

            /// Probability of new beam state, with this extension.
            prob: f32,
        }
        let mut topk_extensions: Vec<BeamExtension> = Vec::new();

        for pos in 0..seq {
            // Initialize extension probs to zero (-inf in log space).
            next_prob_blank.apply(|_| f32::NEG_INFINITY);
            next_prob_no_blank.apply(|_| f32::NEG_INFINITY);

            // Compute all cases where extending a state's prefix (s1) will
            // produce the same prefix as another state (s2).
            merges.clear();
            for (s1_index, s1) in beam.iter().enumerate() {
                for (s2_index, s2) in beam.iter().enumerate() {
                    if s2.prefix.len() == s1.prefix.len() + 1
                        && s1
                            .prefix
                            .iter()
                            .map(|step| step.label)
                            .eq(s2.prefix[..s1.prefix.len()].iter().map(|step| step.label))
                    {
                        merges.insert((s1_index, s2.prefix[s1.prefix.len()].label), s2_index);
                    }
                }
            }

            // Compute probabilities of all possible extensions to beam states.
            for (
                beam_index,
                BeamState {
                    prefix,
                    prob_blank,
                    prob_no_blank,
                },
            ) in beam.iter().enumerate()
            {
                // Compute extension by a blank. The prefix stays the same
                // but the probabilities are updated.
                let prob = prob_seq[[pos, 0]];
                let np_blank = &mut next_prob_blank[[beam_index, 0]];
                *np_blank = log_sum_exp([*np_blank, prob_blank + prob, prob_no_blank + prob]);

                // Compute extension by non-blank labels.
                let prev_label = prefix.last().map(|step| step.label);
                for label in 1..n_labels {
                    let prob = prob_seq[[pos, label]];

                    // Find the existing state, if any, with the same prefix as
                    // extending the current state by this label.
                    //
                    // If we find one, we update the probabilities of that
                    // state instead. This effectively merges the states.
                    let target_index = merges.get(&(beam_index, label as u32));
                    let np_no_blank = if let Some(&target_index) = target_index {
                        // TODO - Do we need to do anything with `prev_label`
                        // in the event of a merge.
                        &mut next_prob_no_blank[[target_index, 0]]
                    } else {
                        &mut next_prob_no_blank[[beam_index, label]]
                    };

                    if Some(label as u32) != prev_label {
                        *np_no_blank =
                            log_sum_exp([*np_no_blank, prob_blank + prob, prob_no_blank + prob]);
                    } else {
                        // The CTC algorithm merges repeats that are not
                        // separated by blanks. Consequently if the current
                        // label repeats the previous one, we need to distribute
                        // the `prob_no_blank + prob` update to the unchanged
                        // prefix.
                        *np_no_blank = log_sum_exp([*np_no_blank, prob_blank + prob]);

                        let np_no_blank = &mut next_prob_no_blank[[beam_index, 0]];
                        *np_no_blank = log_sum_exp([*np_no_blank, prob_no_blank + prob]);
                    };
                }
            }

            // Compute the best new beam states from all possible extensions.
            topk_extensions.clear();
            for bi in 0..beam.len() {
                for label in 0..n_labels {
                    let prob_sum = log_sum_exp([
                        next_prob_blank[[bi, label]],
                        next_prob_no_blank[[bi, label]],
                    ]);
                    if topk_extensions.len() < beam_size as usize
                        || prob_sum
                            > topk_extensions
                                .last()
                                .map(|ext| ext.prob)
                                .unwrap_or(f32::NEG_INFINITY)
                    {
                        topk_extensions.push(BeamExtension {
                            index: bi as u32,
                            label: NonZeroU32::new(label as u32),
                            prob: prob_sum,
                        });

                        // Sort by probability descending.
                        topk_extensions.sort_by(|a, b| (-a.prob).total_cmp(&-b.prob));

                        // Keep only the best new beams.
                        topk_extensions.truncate(beam_size as usize);
                    }
                }
            }

            beam = topk_extensions
                .iter()
                .map(|ext| {
                    let i = ext.index as usize;
                    let mut prefix = beam[i].prefix.clone();
                    if let Some(label) = ext.label {
                        prefix.push(DecodeStep {
                            label: label.get(),
                            pos: pos as u32,
                        });
                    }
                    BeamState {
                        prefix,
                        prob_blank: next_prob_blank
                            [[i, ext.label.map(|l| l.get() as usize).unwrap_or(0)]],
                        prob_no_blank: next_prob_no_blank
                            [[i, ext.label.map(|l| l.get() as usize).unwrap_or(0)]],
                    }
                })
                .collect();
        }

        beam
    }
}

impl Default for CtcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;

    use super::{CtcDecoder, CtcHypothesis, log_sum_exp};

    const ALPHABET: &str = " abcdefghijklmnopqrstuvwxyz";

    /// Encode a string as a sequence of labels suitable for creating an input
    /// matrix for CTC decoding.
    fn encode_str(s: &str, separate_repeats: bool) -> Vec<u32> {
        let mut seq = Vec::new();
        let mut prev_ch = None;
        for s_ch in s.chars() {
            if separate_repeats && Some(s_ch) == prev_ch {
                // Repeated characters are ignored during CTC decoding, so
                // a blank must be inserted.
                seq.push(0);
            }
            prev_ch = Some(s_ch);
            let class = ALPHABET
                .chars()
                .position(|a_ch| a_ch == s_ch)
                .map(|idx| idx + 1)
                .unwrap_or(0) as u32;
            seq.push(class);
        }
        seq
    }

    /// Create a `[seq, class]` matrix of log probabilities from a sequence of
    /// class labels.
    fn onehot_tensor(seq: &[u32]) -> NdTensor<f32, 2> {
        let mut x = NdTensor::zeros([seq.len(), ALPHABET.len()]);
        x.iter_mut().for_each(|el| *el = f32::NEG_INFINITY);
        for (t, class) in seq.iter().copied().enumerate() {
            x[[t, class as usize]] = 0.
        }
        x
    }

    fn label_positions(hyp: &CtcHypothesis) -> Vec<u32> {
        hyp.steps().iter().map(|s| s.pos).collect()
    }

    #[test]
    fn test_decode_greedy() {
        let decoder = CtcDecoder::new();
        let input = onehot_tensor(&encode_str("foobar", true));

        let output = decoder.decode_greedy(input.view());
        let output_str = output.to_string(ALPHABET);
        let output_pos = label_positions(&output);

        assert_eq!(output_str, "foobar");
        // nb. Sequence has a gap due to blank inserted by `encode_str`
        // betweened repeated characters.
        assert_eq!(output_pos, [0, 1, 3, 4, 5, 6]);
        // Probability is 1 (0 in log-space) since target label has prob of 1
        // at each time step.
        assert_eq!(output.score, 0.);
    }

    #[test]
    fn test_decode_beam() {
        let decoder = CtcDecoder::new();
        let input = onehot_tensor(&encode_str("foobar", true));

        let output = decoder.decode_beam(input.view(), 10);
        let output_str = output.to_string(ALPHABET);
        let output_pos = label_positions(&output);

        assert_eq!(output_str, "foobar");
        // nb. Sequence has a gap due to blank inserted by `encode_str`
        // betweened repeated characters.
        assert_eq!(output_pos, [0, 1, 3, 4, 5, 6]);
        // Probability is 1 (0 in log-space) since target label has prob of 1
        // at each time step.
        assert_eq!(output.score, 0.);
    }

    #[test]
    fn test_decode_skips_repeats() {
        let decoder = CtcDecoder::new();
        let input = onehot_tensor(&encode_str("foobar", /* separate_repeats */ false));
        assert_eq!(
            decoder.decode_greedy(input.view()).to_string(ALPHABET),
            "fobar"
        );
        assert_eq!(
            decoder.decode_beam(input.view(), 10).to_string(ALPHABET),
            "fobar"
        );
    }

    #[test]
    fn test_decode_beam_sums_paths() {
        let decoder = CtcDecoder::new();

        // Set up an input where the greedy path is "", but the probability of
        // all paths that decode to "a" is higher than the greedy path. Hence
        // the output of beam and greedy decoding should be different.
        //
        // Example taken from https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7.
        let blank_label = 0;
        let a_label = ALPHABET.chars().position(|c| c == 'a').unwrap() + 1;
        let mut input = NdTensor::<f32, 2>::zeros([2, a_label + 1]);

        input[[0, blank_label]] = 0.8;
        input[[0, a_label]] = 0.2;
        input[[1, a_label]] = 0.4;
        input[[1, blank_label]] = 0.6;

        // Convert to log probabilities.
        input.apply(|x| x.ln());

        let beam_output = decoder.decode_beam(input.view(), 10);
        let beam_str = beam_output.to_string(ALPHABET);
        let beam_positions = label_positions(&beam_output);
        let greedy_str = decoder.decode_greedy(input.view()).to_string(ALPHABET);

        assert_eq!(greedy_str, "");
        assert_eq!(beam_str, "a");

        // The position of the letter should be the first position where it
        // appears in the output.
        assert_eq!(beam_positions, [0]);

        // Score should be the sum of probabilities of paths that produce this
        // output.
        let expected_score = log_sum_exp([
            input[[0, blank_label]] + input[[1, a_label]],
            input[[0, a_label]] + input[[1, blank_label]],
            input[[0, a_label]] + input[[1, a_label]],
        ]);
        assert_eq!(beam_output.score(), expected_score);

        // With a beam width of 1 however, we'll get the same output as greedy,
        // because only one path is kept.
        let beam_output = decoder.decode_beam(input.view(), 1);
        let beam_str = beam_output.to_string(ALPHABET);
        assert_eq!(beam_str, "");
        let expected_score = log_sum_exp([input[[0, blank_label]] + input[[1, blank_label]]]);
        assert_eq!(beam_output.score(), expected_score);
    }
}
