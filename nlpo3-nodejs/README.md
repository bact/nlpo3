---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpO3 Node.js binding

[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/licenses/Apache-2.0)

Node.js binding for nlpO3 Thai tokenization.

## Overview

- Class-based tokenizer API for Node.js.
- Three tokenizer classes:
  - `NewmmTokenizer`
  - `NewmmFstTokenizer`
  - `DeepcutTokenizer`
- Improved runtime stability and long-text handling.

For implementation details and design choices, see
[../docs/impl-notes.md](../docs/impl-notes.md).

## Install

```shell
npm i nlpo3-nodejs
```

## Requirements

- Node.js 24 or newer

## Quick start

```javascript
const { NewmmTokenizer, NewmmFstTokenizer, DeepcutTokenizer } = require("nlpo3-nodejs");

const tok = new NewmmTokenizer("/path/to/dict.txt");
const tokens = tok.segment("สวัสดีครับ");

const tokFst = new NewmmFstTokenizer("/path/to/dict.txt");
const tokensFst = tokFst.segment("สวัสดีครับ");

const deepcut = new DeepcutTokenizer();
const tokensDeepcut = deepcut.segment("สวัสดีครับ");
```

## Segment options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `safe` | `boolean` | `false` | For `NewmmTokenizer` and `NewmmFstTokenizer`: avoid long run times on inputs with many ambiguous word boundaries |
| `parallel` | `boolean` | `false` | For `NewmmTokenizer` and `NewmmFstTokenizer`: use Rayon to process a large input in parallel chunks (higher memory use) |

`DeepcutTokenizer.segment()` in Node.js currently uses a fixed path and does
not accept segment options.

[tcc]: https://dl.acm.org/doi/10.1145/355214.355225

## Dictionary

To keep the library small, nlpO3 does not include a dictionary.
For dictionary-based tokenizers you must provide one.

Recommended dictionaries:

- [words_th.txt][dict-pythainlp] from [PyThaiNLP][pythainlp] (~62,000 words, CC0-1.0)
- [word break dictionary][dict-libthai] from [libthai][libthai] (LGPL-2.1)

[pythainlp]: https://github.com/PyThaiNLP/pythainlp
[libthai]: https://github.com/tlwg/libthai/
[dict-pythainlp]: https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt
[dict-libthai]: https://github.com/tlwg/libthai/tree/master/data

## Support

- Issues: <https://github.com/PyThaiNLP/nlpo3/issues>

## License

nlpO3 Node.js binding is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.
