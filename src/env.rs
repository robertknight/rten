/// Interpret a string value such as "1" or "no" as a boolean.
pub fn str_as_bool(s: &str) -> bool {
    match s {
        "1" | "true" | "t" | "yes" | "y" => true,
        "0" | "false" | "f" | "no" | "n" => false,
        _ => {
            eprintln!("Unrecognized boolean value \"{}\"", s);
            false
        }
    }
}

/// Return whether a feature flag controlled by an environment variable is
/// enabled.
pub fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .as_ref()
        .map(|s| str_as_bool(s))
        .unwrap_or(default)
}
