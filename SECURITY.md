# Security Policy

## Supported versions

Only the latest release on `main` receives security fixes. The package is
pre-1.0 (alpha); breaking changes may occur between minor versions.

## Reporting a vulnerability

Do not open a public issue for security problems. Report privately to the
maintainer via the GitHub security advisory flow
(<https://github.com/diegoramfin/AB-Test-Pkg/security/advisories/new>) or
open a draft pull request with the fix and no public discussion.

Include, when possible:

- Affected version and Python environment.
- Steps to reproduce, including input data shape.
- Impact: what an attacker could do.
- A suggested fix if you have one.

## Scope

The package reads data and writes reports. It never reads, writes, or logs
Kaggle credentials. Datasets flagged `teaching-sample` are provided for
pipeline demonstration only and must not be used as evidence about product
effects. Validation is designed to reject ambiguous or non-finite input
rather than silently handling it; any path where malformed input produces a
misleading report instead of a clear error is in scope.