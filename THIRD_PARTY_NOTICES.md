# Third-party notices

## Moka

This project vendors parts of [Moka](https://github.com/millionco/moka), a Go
9×9 neural network published by Million Software, Inc.

*   Upstream: <https://github.com/millionco/moka>
*   Vendored commit: `a418039e9d823779e3e6d0ea17a244230abc8379`
*   Vendored path: `vendor/moka/`
*   License: MIT, reproduced in `vendor/moka/LICENSE`

The vendored copy is never modified. Its file list and per-file SHA-256
digests are recorded in `vendor/moka/VENDORED.md`.

`models/synthetic-19/` is derived from Moka's weights: its trunk is copied
byte for byte from `vendor/moka/model/go-model.bin`, and its heads are
resampled from Moka's head rows. It is distributed under the same MIT license.
Its origin is recorded in `models/synthetic-19/PROVENANCE.md`.

`vendor/moka-v2/` holds the Moka v2 model and its documentation, from commit
`1ad518a3300cf14885293f1c8cfe4d840b975136` of the upstream `moka-v2` branch
([millionco/moka#3](https://github.com/millionco/moka/pull/3), open when
copied). It is under the same MIT license, reproduced in
`vendor/moka-v2/LICENSE`, and is never modified either; its digests are in
`vendor/moka-v2/VENDORED.md`.

> MIT License
>
> Copyright (c) 2026 Million Software, Inc.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
> the Software, and to permit persons to whom the Software is furnished to do so,
> subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Notices of Moka itself

Moka's own `THIRD_PARTY_NOTICES.md`, reproduced verbatim:

> # Third-party notices
>
> The pass-history, liberty-map, and scoring regression cases were adapted from Web KaTrain at commit `7a0a4876ed0577bac3e511df4938ba5223446e6a`.
>
> Copyright (c) 2026 Web KatRain Contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
> the Software, and to permit persons to whom the Software is furnished to do so,
> subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
