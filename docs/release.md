# Release process

The release process uses
[cargo-release](https://github.com/crate-ci/cargo-release) to simplify releasing
multiple crates from a single repository.

1. Update `CHANGELOG.md` in root of repo. To find PRs merged since the last
   release you can use `git log --oneline v{prev_version}..main --merges`.
2. Run `cargo release changes` to find out which crates changed since the
   previous release
3. Run `cargo release --workspace <new_version>` to do a dry run. Alternatively
   use `cargo release -p crate1 -p crate2 <new_version>` to do this just for
   crates with changes in the new release. You can also use the `--exclude` flag
   to exclude packages that haven't changed.
4. If the dry run looks good, run step 2 again with the `--execute` flag
5. Bump the package version of `rten-convert` in its `pyproject.toml` file
6. Publish `rten-convert` to PyPI following the steps at
   https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives
