use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use rten::ops::FloatOperators;
use rten::{Input, Model, NodeId};
use rten_tensor::prelude::*;
use rten_tensor::*;
use rten_text::tokenizers::{EncodeOptions, Encoded, Tokenizer};

struct Args {
    model: String,
    tokenizer: String,
    context_doc: String,
    query: String,
    n_best: usize,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();
    let mut n_best = 1;

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Short('n') | Long("n-best") => {
                n_best = parser.value()?.parse()?;
            }
            Long("help") => {
                println!(
                    "Find answers to questions in a text file.

Usage: {bin_name} <model> <tokenizer> <context_doc> <query...> [options]

Args:

  <model>       - Input BERT or RoBERTa model
  <tokenizer>   - Tokenizer configuration (tokenizer.json file)
  <context_doc> - Text document to search
  <query>       - Question to search for answer to

Options:

  -n, --n-best [n]  - Number of answers to produce (default 1)
",
                    bin_name = parser.bin_name().unwrap_or("bert_qa")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let model = values.pop_front().ok_or("missing `model` arg")?;
    let tokenizer = values.pop_front().ok_or("missing `tokenizer` arg")?;
    let context_doc = values.pop_front().ok_or("missing `context_doc` arg")?;
    let query = values.make_contiguous().join(" ");

    let args = Args {
        context_doc,
        model,
        n_best,
        query,
        tokenizer,
    };

    Ok(args)
}

struct Answer<'a> {
    score: f32,
    text: &'a str,
}

/// Extract the the spans of the context from `query_context` which best
/// answer the query.
///
/// `query_context` is the tokenized query and context, `model` is a BERT model
/// finetuned for extractive QA. `n_best` is the number of results to return.
fn extract_nbest_answers<'a>(
    query_context: Encoded<'a>,
    model: &Model,
    n_best: usize,
) -> Result<Vec<Answer<'a>>, Box<dyn Error>> {
    let batch = 1;
    let input_ids: Tensor<i32> = query_context
        .token_ids()
        .iter()
        .map(|tid| *tid as i32)
        .collect::<Tensor<_>>()
        .into_shape([1, query_context.token_ids().len()].as_slice());
    let attention_mask = Tensor::full(&[batch, input_ids.len()], 1i32);

    let input_ids_id = model.node_id("input_ids")?;
    let attention_mask_id = model.node_id("attention_mask")?;
    let start_logits_id = model.node_id("start_logits")?;
    let end_logits_id = model.node_id("end_logits")?;

    let mut inputs: Vec<(NodeId, Input)> = vec![
        (input_ids_id, input_ids.view().into()),
        (attention_mask_id, attention_mask.view().into()),
    ];

    // Generate token type IDs if this model needs them. The original BERT
    // uses them, DistilBERT for example does not.
    let type_ids: Tensor<i32>;
    if let Some(type_ids_id) = model.find_node("token_type_ids") {
        type_ids = query_context
            .token_type_ids()
            .map(|tid| tid as i32)
            .collect::<Tensor<_>>()
            .into_shape([1, query_context.token_ids().len()].as_slice());
        inputs.push((type_ids_id, type_ids.view().into()));
    }

    let [start_logits, end_logits] =
        model.run_n(&inputs, [start_logits_id, end_logits_id], None)?;

    // Extract (batch, sequence)
    let mut start_logits: NdTensor<f32, 2> = start_logits.try_into()?;
    let mut end_logits: NdTensor<f32, 2> = end_logits.try_into()?;

    // Mask of positions that are part of the context, excluding the final `[SEP]`.
    let mut context_mask: Vec<bool> = query_context
        .token_type_ids()
        .map(|ttid| ttid == 1)
        .collect();
    context_mask[start_logits.len() - 1] = false;

    // Set logits for positions outside of context to a large negative value
    // before applying softmax, so those positions don't affect the values
    // of in-context positions.
    for (pos, (start, end)) in start_logits
        .iter_mut()
        .zip(end_logits.iter_mut())
        .enumerate()
    {
        if !context_mask[pos] {
            *start = -10_000.0;
            *end = -10_000.0;
        }
    }

    let start_probs: NdTensor<f32, 2> = start_logits.softmax(1)?.try_into()?;
    let end_probs: NdTensor<f32, 2> = end_logits.softmax(1)?.try_into()?;

    // Extract the answer as the highest scoring span where both the start and
    // end positions are inside the context window.
    //
    // In the original BERT paper (see Section 4.2) the score is the sum of
    // logits for the start and end positions for a span. Here we take the
    // product of probabilities for the start and end positions, following the
    // HF Transformers implementation [1].
    //
    // [1] https://github.com/huggingface/transformers/blob/df5c5c62ae253055336f5bb0828ca8e3e15ab6bd/src/transformers/pipelines/question_answering.py#L72
    let max_answer_len = 15;
    let min_start = 1; // Ignore [CLS] token at start.
    let max_end = end_probs.size(1) - 1; // Ignore [SEP] token at end.
    let mut span_scores: Vec<(usize, usize, f32)> = start_probs
        .slice::<1, _>((0, min_start..max_end))
        .iter()
        .enumerate()
        .map(|(start_pos, start_score)| {
            let start_pos = start_pos + min_start;
            let (relative_end_pos, end_score) = end_probs
                .slice::<1, _>((0, start_pos..(start_pos + max_answer_len).min(max_end)))
                .iter()
                .enumerate()
                .max_by(|(_pos_a, score_a), (_pos_b, score_b)| score_a.total_cmp(score_b))
                .unwrap();
            let end_pos = relative_end_pos + start_pos;

            let span_score = start_score * end_score;

            (start_pos, end_pos, span_score)
        })
        .collect();
    span_scores.sort_by(|(_, _, score_a), (_, _, score_b)| score_a.total_cmp(score_b).reverse());

    let n_best_answers: Vec<_> = span_scores
        .into_iter()
        .take(n_best)
        .map(|(start_pos, end_pos, score)| {
            let text = query_context
                .text_for_token_range(start_pos..end_pos + 1)
                .expect("failed to get answer text");
            Answer { score, text }
        })
        .collect();

    Ok(n_best_answers)
}

/// This example finds passages in a document that best answer a given query,
/// aka. extractive QA [^1].
///
/// It works with BERT-based models that have been fine-tuned for question
/// answering, such as <https://huggingface.co/deepset/bert-base-cased-squad2> or
/// <https://huggingface.co/distilbert-base-cased-distilled-squad>.
///
/// You can export a BERT model in ONNX format from Hugging Face and convert
/// it as follows, using Optimium [^2].
///
/// ```
/// optimum-cli export onnx --model distilbert-base-cased-distilled-squad distilbert
/// rten-convert distilbert/model.onnx distilbert/distilbert.rten
/// ```
///
/// Then run the example with:
///
/// ```
/// cargo run -r --bin bert_qa distilbert/distilbert.rten distilbert/tokenizer.json <context> <query>
/// ```
///
/// Where `<context>` is a text file to search, and `<query>` is a question.
/// For example, given a text file containing "My name is Robert and I
/// live in London" and the query "what am I called" the model should output
/// the substring "Robert".
///
/// [^1]: <https://huggingface.co/tasks/question-answering>
/// [^2]: <https://huggingface.co/docs/optimum/index>
fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model = Model::load_file(args.model)?;

    let context = fs::read_to_string(args.context_doc)?;

    let tokenizer_json = fs::read_to_string(args.tokenizer)?;
    let tokenizer = Tokenizer::from_json(&tokenizer_json)?;

    // Tokenize the query and context, breaking the context up into chunks to
    // fit the model's context length.
    let enc_opts = EncodeOptions {
        // Max chunk length chosen as 384 to match what is used by the original
        // BERT training scripts + the Hugging Face QA pipeline.
        max_chunk_len: Some(384),

        // Overlap controls how many tokens successive chunks overlap by.
        // This can avoid the model failing to find answers if the answer
        // crosses a chunk boundary.
        overlap: 0,
        ..Default::default()
    };
    let encoded =
        tokenizer.encode_chunks((args.query.as_str(), context.as_str()).into(), enc_opts)?;

    let mut answers = Vec::new();
    for chunk in encoded {
        let nbest_per_chunk = args.n_best;
        let chunk_answers = extract_nbest_answers(chunk, &model, nbest_per_chunk)?;
        answers.extend(chunk_answers.into_iter());
    }
    answers.sort_by(|ans_a, ans_b| ans_a.score.total_cmp(&ans_b.score).reverse());

    println!("Question: {}", args.query);
    for answer in answers.into_iter().take(args.n_best) {
        println!("Answer (score {:.2}): {}", answer.score, answer.text);
    }

    Ok(())
}
