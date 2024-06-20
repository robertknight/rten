use std::any::Any;

/// Allows downcasting a trait object to a concrete type.
///
/// This can be implemented for `dyn SomeTrait`, where `SomeTrait` is a trait
/// that has `Any` as a supertrait, by using [`impl_downcastdyn`].
///
/// When the `trait_upcasting` feature is stabilized, this can be removed
/// as callers can upcast the trait object to `dyn Any` and then downcast.
pub(crate) trait DowncastDyn {
    fn is<T: Any>(&self) -> bool;
    fn downcast_ref<T: Any>(&self) -> Option<&T>;
}

/// Implement [`DowncastDyn`] for a trait. The trait must have `Any` as a
/// supertrait.
macro_rules! impl_downcastdyn {
    ($trait:ident) => {
        // Trigger compile error if `Any` is not a supertrait of `$trait`.
        //
        // Credit: https://stackoverflow.com/a/64826111/434243
        fn _assert_any_supertrait<T: $trait>() {
            fn has_any_supertrait<T: Any>() {}
            let _ = has_any_supertrait::<T>;
        }

        // The implementation approach was taken from how `downcast_ref` is implemented
        // for `dyn Error` in the standard library.
        impl $crate::downcast::DowncastDyn for dyn $trait {
            fn is<T: std::any::Any>(&self) -> bool {
                // nb. If `$trait` does not have an `Any` supertrait, this code
                // still compiles, but `type_id` will return a different value.
                // Hence the `_assert_any_supertrait` check above.
                std::any::TypeId::of::<T>() == <Self as std::any::Any>::type_id(self)
            }

            fn downcast_ref<T: std::any::Any>(&self) -> Option<&T> {
                if self.is::<T>() {
                    // SAFETY: `is` ensures the cast is correct.
                    Some(unsafe { &*(self as *const dyn $trait as *const T) })
                } else {
                    None
                }
            }
        }
    };
}

pub(crate) use impl_downcastdyn;

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::{impl_downcastdyn, DowncastDyn};

    trait Foo: Any {}
    impl_downcastdyn!(Foo);

    struct TypeA {}
    impl Foo for TypeA {}

    struct TypeB {}
    impl Foo for TypeB {}

    #[test]
    fn test_downcast_ref() {
        let type_a = TypeA {};
        let type_b = TypeB {};

        let type_a_dyn: &dyn Foo = &type_a;
        let type_b_dyn: &dyn Foo = &type_b;

        assert!(type_a_dyn.is::<TypeA>());
        assert!(!type_a_dyn.is::<TypeB>());
        assert!(std::ptr::eq(
            type_a_dyn.downcast_ref::<TypeA>().unwrap(),
            &type_a
        ));
        assert!(type_a_dyn.downcast_ref::<TypeB>().is_none());

        assert!(type_b_dyn.is::<TypeB>());
        assert!(!type_b_dyn.is::<TypeA>());
        assert!(std::ptr::eq(
            type_b_dyn.downcast_ref::<TypeB>().unwrap(),
            &type_b
        ));
        assert!(type_b_dyn.downcast_ref::<TypeA>().is_none());
    }
}
