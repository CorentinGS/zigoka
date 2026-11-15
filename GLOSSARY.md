# zigoka

A Zig/WebAssembly engine for Moka's Go networks, at 9×9 and 19×19, with the tools that build, check and train its models.

## Models

**Model**:
A Go network the engine runs, named by an identifier such as `moka-9` or `moka-19`. It plays at exactly one board size.
_Avoid_: network (for the whole model), weights

**Model registry**:
The single versioned list of the models the repository builds and tests. A model that isn't registered isn't part of the build.
_Avoid_: model list, model index

**Trained model**:
A registered model whose weights come from training: one of Moka's published models or a student distilled by the pipeline.

**Synthetic model**:
A registered model that keeps the trunk of a trained model and gets generated heads. It measures the engine at a board size before any trained model exists there, and its outputs carry no Go knowledge.
_Avoid_: fake model, random model

**Local model**:
An unversioned export of a pipeline checkpoint, built next to the registered models but never part of the model registry.
_Avoid_: draft model, unregistered model

## Engines

**Engine**:
The Zig/WebAssembly build of a model, instantiated by a host. An infer engine evaluates positions; a search engine also searches, and holds one search tree.
_Avoid_: runtime (Moka's TypeScript reference implementation)

**Host**:
A program that runs engines: the published worker, the site, the demo match, the match CLI, the bench and the tests.
