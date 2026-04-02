---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpO3 Node.js binding

[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/licenses/Apache-2.0)

Node.js binding for nlpO3, a Thai natural language processing library in Rust.

## Features

- Three interchangeable Thai word tokenizers:
  - `NewmmTokenizer` — dictionary-based maximal-matching (TrieChar backend,
    fastest, ~43 MB for 62k words)
  - `NewmmFstTokenizer` — same algorithm with FST backend (~49× less memory)
  - `DeepcutTokenizer` — neural CNN model (no dictionary needed)
- All tokenizers honor [Thai Character Cluster][tcc] boundaries
- Each instance is read-only after construction and safe to reuse across
  many concurrent async calls on the Node.js event loop

[tcc]: https://dl.acm.org/doi/10.1145/355214.355225

## Build

### Requirements

- [Rust 2024 Edition](https://www.rust-lang.org/tools/install)
- Node.js 24 (LTS) or newer
- TypeScript 6

### Steps

```bash
# In this directory
npm run release
```

After build, `nlpo3/` contains:

```text
nlpo3/
  index.js          ← compiled TypeScript (main entry point)
  index.d.ts        ← TypeScript declarations
  index.ts          ← TypeScript source
  rust_mod.d.ts     ← Rust native module declarations
  rust_mod.node     ← compiled Rust native addon
```

## Install

```shell
npm i nlpo3-nodejs
```

## Usage

### JavaScript

```javascript
const { NewmmTokenizer, NewmmFstTokenizer, DeepcutTokenizer } = require("nlpo3-nodejs");

// Dictionary-based tokenizer (TrieChar backend — fastest)
// Create once, reuse for every call — the dictionary is loaded only once.
const tok = new NewmmTokenizer("/path/to/dict.txt");
tok.segment("สวัสดีครับ");
// => ["สวัสดี", "ครับ"]

// Same options as newmm, but FST backend uses ~49x less memory
const fstTok = new NewmmFstTokenizer("/path/to/dict.txt");
fstTok.segment("สวัสดีครับ");

// Neural CNN tokenizer — no dictionary needed
const dc = new DeepcutTokenizer();
dc.segment("สวัสดีครับ");
```

### TypeScript

```typescript
import { NewmmTokenizer, NewmmFstTokenizer, DeepcutTokenizer, SegmentOptions } from "nlpo3-nodejs";

// Create once, reuse the same instance across all calls.
// The dictionary is loaded only once; all segment() calls share it.
const tok = new NewmmTokenizer("/path/to/dict.txt");

const tokens: string[] = tok.segment("สวัสดีครับ");

// Optional flags
const opts: SegmentOptions = { safe: true, parallel: false };
tok.segment("สวัสดีครับ", opts);

// FST backend — less memory, same API
const fstTok = new NewmmFstTokenizer("/path/to/dict.txt");
fstTok.segment("สวัสดีครับ");

// Neural tokenizer
const dc = new DeepcutTokenizer();
dc.segment("สวัสดีครับ");
```

### Parallel / distributed environments

Each tokenizer instance is **read-only after construction** and safe to call
from many concurrent async operations on the same Node.js event loop:

```typescript
import express from "express";
import { NewmmTokenizer } from "nlpo3-nodejs";

// One tokenizer serves all requests — the dictionary is never reloaded.
const tok = new NewmmTokenizer("/path/to/dict.txt");
const app = express();

app.get("/segment", (req, res) => {
  const text = String(req.query.text ?? "");
  res.json({ tokens: tok.segment(text) });
});

app.listen(3000, () => {
  console.log("Listening on http://localhost:3000");
});
```

For Node.js worker threads, each worker creates its own instance.
Loading a 62k-word dictionary typically takes less than 200 ms.

### Segment options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `safe` | `boolean` | `false` | Avoid long run times on inputs with many ambiguous word boundaries |
| `parallel` | `boolean` | `false` | Use Rayon to tokenize one document in parallel chunks (higher memory use). Best for a single large document; for many short texts, use caller-side worker threads instead. |

`DeepcutTokenizer.segment()` does not accept options; the neural model has a fixed inference path.

### Dictionary

To keep the library small, nlpO3 does not include a dictionary.
For dictionary-based tokenizers you must provide one.

Recommended dictionaries:

- [words_th.txt][dict-pythainlp] from [PyThaiNLP][pythainlp] (~62,000 words, CC0-1.0)
- [word break dictionary][dict-libthai] from [libthai][libthai] (LGPL-2.1)

[pythainlp]: https://github.com/PyThaiNLP/pythainlp
[libthai]: https://github.com/tlwg/libthai/
[dict-pythainlp]: https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt
[dict-libthai]: https://github.com/tlwg/libthai/tree/master/data

## License

nlpO3 Node.js binding is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.
