use std::iter::StepBy;
use std::ops::Range;
use std::slice::Windows;

/// Iterator over chunks of a slice, with an overlap between each chunk and
/// the next.
pub struct OverlappingChunks<'a, T> {
    /// Iterator over full chunks.
    inner: StepBy<Windows<'a, T>>,
    /// The final non-full chunk.
    remainder: Option<&'a [T]>,
}

impl<'a, T> Iterator for OverlappingChunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().or_else(|| self.remainder.take())
    }
}

/// Additional methods for splitting slices.
pub trait SliceExt {
    type Elem;

    /// Split a slice into chunks with an overlap of `overlap` elements between
    /// successive chunks.
    fn chunks_with_overlap(
        &self,
        chunk_size: usize,
        overlap: usize,
    ) -> OverlappingChunks<'_, Self::Elem>;

    /// Return the offset range occupied by a subslice within this slice.
    ///
    /// This is a backport of `subslice_range` from nightly Rust, renamed
    /// to avoid a conflict when that method is stabilized.
    ///
    /// See https://doc.rust-lang.org/std/primitive.slice.html#method.subslice_range.
    fn subslice_offsets(&self, subslice: &[Self::Elem]) -> Option<Range<usize>>;
}

impl<T> SliceExt for [T] {
    type Elem = T;

    fn chunks_with_overlap(&self, chunk_size: usize, overlap: usize) -> OverlappingChunks<'_, T> {
        // Iterator cannot make progress unless each chunk contains at least
        // one new element.
        assert!(overlap < chunk_size);

        let stride = chunk_size - overlap;
        let remainder_size = if self.len() < chunk_size {
            self.len()
        } else {
            self.len().saturating_sub(chunk_size) % stride
        };

        OverlappingChunks {
            inner: self.windows(chunk_size).step_by(stride),
            remainder: if remainder_size > 0 {
                Some(&self[self.len() - remainder_size..])
            } else {
                None
            },
        }
    }

    fn subslice_offsets(&self, subslice: &[T]) -> Option<Range<usize>> {
        // Implementation copied from `subslice_range` in nightly Rust.
        if size_of::<T>() == 0 {
            panic!("elements are zero-sized");
        }

        let self_start = self.as_ptr() as usize;
        let subslice_start = subslice.as_ptr() as usize;

        let byte_start = subslice_start.wrapping_sub(self_start);

        if !byte_start.is_multiple_of(core::mem::size_of::<T>()) {
            return None;
        }

        let start = byte_start / core::mem::size_of::<T>();
        let end = start.wrapping_add(subslice.len());

        if start <= self.len() && end <= self.len() {
            Some(start..end)
        } else {
            None
        }
    }
}

/// Iterator returned by [`SplitExt::split_keep_delimeters`].
pub struct SplitKeepDelim<'a, P: FnMut(char) -> bool> {
    remainder: &'a str,
    predicate: P,
}

impl<'a, P: FnMut(char) -> bool> Iterator for SplitKeepDelim<'a, P> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        for (index, ch) in self.remainder.char_indices() {
            if !(self.predicate)(ch) {
                continue;
            }
            if index == 0 {
                let mut next_index = 1;
                while !self.remainder.is_char_boundary(next_index) {
                    next_index += 1;
                }
                let substr = &self.remainder[..next_index];
                self.remainder = &self.remainder[next_index..];
                return Some(substr);
            } else {
                let substr = &self.remainder[..index];
                self.remainder = &self.remainder[index..];
                return Some(substr);
            }
        }
        if !self.remainder.is_empty() {
            let substr = self.remainder;
            self.remainder = "";
            Some(substr)
        } else {
            None
        }
    }
}

pub trait SplitExt<'a> {
    /// Split a string but retain the delimeters.
    ///
    /// ```text
    /// use crate::split::SplitExt;
    ///
    /// let str = "foo.bar";
    /// let tokens: Vec<_> = str.split_keep_delimeters(|ch| ch.is_ascii_punctuation()).collect();
    /// assert_eq!(tokens, &["foo", ".", "bar"]);
    /// ```
    fn split_keep_delimeters<P: FnMut(char) -> bool>(self, predicate: P) -> SplitKeepDelim<'a, P>;
}

impl<'a> SplitExt<'a> for &'a str {
    fn split_keep_delimeters<P: FnMut(char) -> bool>(self, predicate: P) -> SplitKeepDelim<'a, P> {
        SplitKeepDelim {
            remainder: self,
            predicate,
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{SliceExt, SplitExt};

    #[test]
    fn test_chunks_overlap() {
        #[derive(Debug)]
        struct Case<'a> {
            input: &'a [i32],
            chunk_size: usize,
            overlap: usize,
            expected: &'a [&'a [i32]],
        }

        let cases = [
            // No overlap, no remainder
            Case {
                input: &[1, 2, 3, 4],
                chunk_size: 2,
                overlap: 0,
                expected: &[&[1, 2], &[3, 4]],
            },
            // Overlap, no remainder
            Case {
                input: &[1, 2, 3, 4],
                chunk_size: 2,
                overlap: 1,
                expected: &[&[1, 2], &[2, 3], &[3, 4]],
            },
            // No overlap, remainder
            Case {
                input: &[3, 4, 5, 6, 7],
                chunk_size: 3,
                overlap: 0,
                expected: &[&[3, 4, 5], &[6, 7]],
            },
            // Overlap, remainder
            Case {
                input: &[3, 4, 5, 6, 7, 8],
                chunk_size: 3,
                overlap: 1,
                expected: &[&[3, 4, 5], &[5, 6, 7], &[8]],
            },
            // One chunk that is smaller than chunk size
            Case {
                input: &[1, 2, 3],
                chunk_size: 10,
                overlap: 0,
                expected: &[&[1, 2, 3]],
            },
        ];

        cases.test_each(|case| {
            let &Case {
                input,
                chunk_size,
                overlap,
                expected,
            } = case;
            let chunks: Vec<_> = input.chunks_with_overlap(chunk_size, overlap).collect();
            assert_eq!(chunks, expected);
        })
    }

    #[test]
    #[should_panic(expected = "overlap < chunk_size")]
    fn test_chunks_overlap_panic() {
        let input = &[1, 2, 3, 4];
        let chunk_size = 4;
        let overlap = 4;
        input.chunks_with_overlap(chunk_size, overlap);
    }

    #[test]
    fn test_split_keep_delimeters() {
        #[derive(Debug)]
        struct Case<'a> {
            text: &'a str,
            expected: &'a [&'a str],
        }

        let cases = [
            Case {
                text: "",
                expected: &[],
            },
            Case {
                text: "foo",
                expected: &["foo"],
            },
            Case {
                text: "foo.",
                expected: &["foo", "."],
            },
            Case {
                text: "foo..",
                expected: &["foo", ".", "."],
            },
            Case {
                text: "Mary had a little lamb, it's face was white as snow.",
                expected: &[
                    "Mary", "had", "a", "little", "lamb", ",", "it", "'", "s", "face", "was",
                    "white", "as", "snow", ".",
                ],
            },
        ];

        cases.test_each(|case| {
            let &Case { text, expected } = case;

            let words_and_puncs: Vec<_> = text
                .split_whitespace()
                .flat_map(|s| s.split_keep_delimeters(|c| c.is_ascii_punctuation()))
                .collect();
            assert_eq!(words_and_puncs, expected,);
        })
    }
}
