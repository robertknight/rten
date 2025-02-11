//! Internal testing utilities for the rten crates.

use std::fmt::Debug;
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Utility for creating parametrized (aka. table-driven) tests.
///
/// To create a table driven test:
///
/// 1. Import the `TestCases` trait
/// 2. Create a struct, conventionally named `Case`, that contains the data
///    for a single test case. This struct must implement `Debug`.
/// 3. Create a collection of `Case` instances (eg. an array or Vec),
///    conventionally named `cases`.
/// 4. Call `cases.test_each`, passing the test function as a closure
///
/// `test_each` will run all of the test cases and catch any panics. If all
/// cases succeed (ie. run without panicking), `test_each` will return.
/// Otherwise it will panic with a message that includes the count and debug
/// representations of failing test cases.
///
/// ## Example
///
/// ```
/// use rten_testing::TestCases;
///
/// // Add #[test] attribute
/// fn test_multiply() {
///   #[derive(Debug)]
///   struct Case {
///     a: i32,
///     b: i32,
///     expected: i32,
///   }
///
///   let cases = [
///     Case { a: 3, b: 5, expected: 15 },
///   ];
///
///   cases.test_each(|&Case { a, b, expected }| {
///     assert_eq!(a * b, expected);
///   });
/// }
/// # test_multiply();
/// ```
///
/// ## Passing cases by reference vs. value
///
/// `test_each` passes the test case to the function by reference, so that it
/// can capture a debug representation of the case in the event of a panic.  If
/// passing by reference causes difficulties, `test_each_clone` or
/// `test_each_value` can be used instead, at the cost of some extra runtime
/// overhead.
///
/// ## Unwind safety
///
/// Both test cases and the test function are required to be
/// [unwind safe](https://doc.rust-lang.org/std/panic/fn.catch_unwind.html).
///
/// Practically this means that the test case items and any values _captured_
/// by the test function closure must not contain interior mutability. Values
/// created and used _within_ the test function can contain interior mutability
/// however.
///
/// If some fields in a test case are not unwind safe, this can be handled in
/// several ways:
///
///  - Replace the field value by a simpler unwind-safe type. For example instead of
///    having a field that contains a `ComplexFoo`, use a field that describes
///    how to create the `ComplexFoo`, and create it within the test function.
///  - Wrap the field with [`AssertUnwindSafe`](std::panic::AssertUnwindSafe)
///
/// If the test function captures values that are not unwind safe, the simplest
/// solution is usually to move the value into the test function. If this is
/// problematic, `AssertUnwindSafe` can be used to wrap the value.
pub trait TestCases {
    /// The data for a single test case.
    type Case;

    /// Call test function `test` with each test case in `self`, catching any panics.
    ///
    /// After all cases have been evaluated, return if no panics occurred or
    /// panic with details of failing cases otherwise.
    fn test_each(self, test: impl Fn(&Self::Case) + RefUnwindSafe)
    where
        Self::Case: Debug + RefUnwindSafe;

    /// Variant of [`test_each`](TestCases::test_each) which passes a clone
    /// of each test case to the test function, rather than a reference.
    ///
    /// This is useful for tests where working with an owned test case is
    /// more convenient than working with a reference, and the cost of cloning
    /// is low.
    fn test_each_clone(self, test: impl Fn(Self::Case) + RefUnwindSafe)
    where
        Self::Case: Debug + Clone + UnwindSafe;

    /// Variant of [`test_each`](TestCases::test_each) which passes test cases
    /// to the test function by value.
    ///
    /// To support printing a debug representation of the case in the event
    /// of an error, each test case is formatted to a string before the test
    /// function is called. This adds a small amount of overhead compared to
    /// [`test_each`](TestCases::test_each).
    fn test_each_value(self, test: impl Fn(Self::Case) + RefUnwindSafe)
    where
        Self::Case: Debug + UnwindSafe;
}

impl<I: IntoIterator> TestCases for I {
    type Case = I::Item;

    fn test_each(self, test: impl Fn(&I::Item) + RefUnwindSafe)
    where
        Self::Case: Debug + RefUnwindSafe,
    {
        let mut failures = Vec::new();
        for case in self {
            if std::panic::catch_unwind(|| {
                test(&case);
            })
            .is_err()
            {
                failures.push(case);
            }
        }
        assert_eq!(
            failures.len(),
            0,
            "{} test cases failed: {:?}",
            failures.len(),
            failures
        );
    }

    fn test_each_clone(self, test: impl Fn(I::Item) + RefUnwindSafe)
    where
        Self::Case: Clone + Debug + UnwindSafe,
    {
        let mut failures = Vec::new();
        for case in self {
            let value = case.clone();
            let test = &test;

            if std::panic::catch_unwind(move || {
                test(value);
            })
            .is_err()
            {
                failures.push(case);
            }
        }
        assert_eq!(
            failures.len(),
            0,
            "{} test cases failed: {:?}",
            failures.len(),
            failures
        );
    }

    fn test_each_value(self, test: impl Fn(I::Item) + RefUnwindSafe)
    where
        Self::Case: Debug + UnwindSafe,
    {
        let mut failures = Vec::new();
        for case in self {
            let test = &test;
            let case_str = format!("{:?}", case);

            if std::panic::catch_unwind(move || {
                test(case);
            })
            .is_err()
            {
                failures.push(case_str);
            }
        }
        assert_eq!(
            failures.len(),
            0,
            "{} test cases failed: {:?}",
            failures.len(),
            failures
        );
    }
}

#[cfg(test)]
mod tests {
    use super::TestCases;

    #[test]
    fn test_test_cases_success() {
        #[derive(Clone, Debug)]
        struct Case {
            x: i32,
        }

        let cases = [Case { x: 1 }, Case { x: 2 }];
        cases.clone().test_each(|case| _ = case.x);
        cases.clone().test_each_clone(|case| _ = case.x);
        cases.clone().test_each_value(|case| _ = case.x);
    }

    #[test]
    #[should_panic(expected = "2 test cases failed")]
    fn test_test_each_failure() {
        #[derive(Debug)]
        struct Case {
            x: i32,
        }

        let cases = [Case { x: 1 }, Case { x: 2 }];
        cases.test_each(|case| {
            _ = case.x;
            panic!("oh no");
        })
    }

    #[test]
    #[should_panic(expected = "2 test cases failed")]
    fn test_test_each_clone_failure() {
        #[derive(Clone, Debug)]
        struct Case {
            x: i32,
        }

        let cases = [Case { x: 1 }, Case { x: 2 }];
        cases.test_each_clone(|case| {
            _ = case.x;
            panic!("oh no");
        })
    }

    #[test]
    #[should_panic(expected = "2 test cases failed")]
    fn test_test_each_value_failure() {
        #[derive(Debug)]
        struct Case {
            x: i32,
        }

        let cases = [Case { x: 1 }, Case { x: 2 }];
        cases.test_each_value(|case| {
            _ = case.x;
            panic!("oh no");
        })
    }
}
