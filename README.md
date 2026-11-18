# avrig-lib
Shared foundational systems libraries, pure domain entities, storage drivers, and Cedar policy engines for avrig-ai (Rust crates)

## Logging crate

`crates/avrig-logging` provides the shared `avrig-logging` Rust crate. Call
`avrig_logging::init(LoggingConfig::default())` once during process startup to
write newline-delimited JSON records to `logs/avrig.log`. Each record includes
an RFC 3339 UTC timestamp, level, tracing target, source file, line number,
message, and structured event fields.

The active file rotates at 10 MiB by default. Five numbered archives are
retained as `avrig.1.log` through `avrig.5.log`. `LoggingConfig` lets a binary
override the log directory, maximum file size, archive count, and tracing filter.
