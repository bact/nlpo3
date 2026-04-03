---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Major release with breaking changes.

### Changed

- Improved API consistency.

JavaScript:

```javascript
// Before (v1.x)
loadDict("path/to/dict.txt", "mydict");
const tokens = segment("สวัสดีครับ", "mydict", false, false);

// Since v2.0.0
import { NewmmTokenizer } from "nlpo3";
const tok = new NewmmTokenizer("path/to/dict.txt");
const tokens = tok.segment("สวัสดีครับ");
```
