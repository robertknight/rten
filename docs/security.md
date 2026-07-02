# Security and resource usage

This page describes the guarantees that RTen does and does not make when
loading and running machine learning models, particularly models that come
from untrusted sources.

## Threat model

Conceptually a model is like a program that RTen runs inside a sandbox.
The sandbox constrains what the model can *observe* and *affect*, but it does
not constrain the resources (memory, CPU usage) the model can consume.

## Safety guarantees

Loading or running a model is always *safe*, in the sense that a model, even a
maliciously crafted one, cannot:

- Cause undefined behavior, such as out-of-bounds reads or writes.
- Read data other than the model's own weights and the inputs you provide.
- Write data anywhere other than the model's outputs.

If you find an exception, please [file an issue](https://github.com/robertknight/rten/issues).

## Non-guarantees: resource usage

RTen does **not** make any guarantees about the resources a model may use. In
particular it does not limit:

- How long inference will take, or whether it will terminate at all.
- How much memory will be allocated.

## Loading untrusted models

In typical usage the model is chosen by the application embedding RTen, and
resource usage is driven by the size of the inputs (for example, the number of
tokens in a prompt). In that case you can bound resource usage by limiting the
size of the inputs you pass to [`Model::run`](crate::Model::run).

If your application needs to load *arbitrary*, untrusted models and enforce
limits on how much CPU time or memory they can use, you must impose those limits
at the process level - for example by running inference in a container or other
sandbox.
