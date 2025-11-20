//! Internal utilities related to serde deserialization.

use std::borrow::Cow;

use serde::Deserializer;

/// A wrapper around [`Cow<str>`] which implements [`serde::Deserialize`] as
/// expected.
///
/// By default serde_json will always allocate when deserializing into a
/// `Cow<str>`, instead of borrowing where possible.
///
/// Code based on https://github.com/GnomedDev/serde_cow. See also
/// https://users.rust-lang.org/t/cow-serde-json/72359.
#[derive(Clone, Debug, PartialEq)]
pub struct CowStr<'de>(pub Cow<'de, str>);

impl<'de> AsRef<str> for CowStr<'de> {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl serde::Serialize for CowStr<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for CowStr<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_string(CowStrVisitor)
    }
}

struct CowStrVisitor;

impl<'de> serde::de::Visitor<'de> for CowStrVisitor {
    type Value = CowStr<'de>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_borrowed_str<E>(self, val: &'de str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(CowStr(Cow::Borrowed(val)))
    }

    fn visit_str<E>(self, val: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_string(val.into())
    }

    fn visit_string<E>(self, val: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(CowStr(Cow::Owned(val)))
    }
}
