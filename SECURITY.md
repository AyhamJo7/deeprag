# Security Policy

## Supported versions

DeepRAG is an actively developed research framework. Security fixes are applied to the latest `main` branch. There is no long-term-support branch.

## Reporting a vulnerability

If you discover a security vulnerability, please report it **privately** — do not open a public issue.

- Email: **ayhamjumran7@gmail.com** with the subject line `SECURITY: deeprag`.
- Include a description, reproduction steps, and the potential impact.

You can expect an acknowledgement within a few business days. Because this is a research/portfolio project maintained on a best-effort basis, there is no formal disclosure SLA, but credible reports will be investigated and fixed as quickly as is practical.

## Scope

DeepRAG loads models, tokenizers, and datasets that may execute code or download remote artifacts. Only load checkpoints and data from sources you trust. Reports about untrusted-model execution are in scope; misuse of intentionally untrusted inputs is not.
