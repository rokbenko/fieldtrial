# Security policy

## Supported versions

fieldtrial is pre-1.0. Security fixes go into the latest release only.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/rokbenko/fieldtrial/security/advisories/new).
Don't open a public issue for a security problem.

Include what you found, how to reproduce it, and the fieldtrial version. You can expect an
acknowledgement within a week. Fixes are announced in the changelog and in a GitHub security
advisory.

## Scope

The operator console (`fieldtrial serve`, coming in 0.1.0) is designed for a trusted local
network. In LAN mode it will require a random access token on every request. It is not meant
to be exposed to the public internet.
