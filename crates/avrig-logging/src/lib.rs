use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tracing_subscriber::{
    EnvFilter, fmt::time::UtcTime, fmt::writer::MakeWriter, layer::SubscriberExt,
    util::SubscriberInitExt,
};

pub const LOG_FILE_NAME: &str = "avrig.log";
pub const DEFAULT_MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;
pub const DEFAULT_MAX_ARCHIVE_FILES: usize = 5;

#[derive(Clone, Debug)]
pub struct LoggingConfig {
    /// Directory where `avrig.log` and its rotated archives are written.
    pub directory: PathBuf,
    /// Maximum size of the active log file before it is rotated.
    pub max_file_size_bytes: u64,
    /// Number of rotated files to retain, named `avrig.1.log`, `avrig.2.log`, etc.
    pub max_archive_files: usize,
    /// tracing filter directive, for example `info` or `avrig=debug,warn`.
    pub filter: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("logs"),
            max_file_size_bytes: DEFAULT_MAX_FILE_SIZE_BYTES,
            max_archive_files: DEFAULT_MAX_ARCHIVE_FILES,
            filter: "info".to_owned(),
        }
    }
}

/// Initialize the process-wide tracing subscriber.
///
/// Output is written as one JSON object per line to `logs/avrig.log`
/// by default. The file rotates before a write would exceed the configured
/// size. Rotated files are retained as `avrig.1.log` through `avrig.N.log`.
pub fn init(config: LoggingConfig) -> io::Result<()> {
    if config.max_file_size_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "max_file_size_bytes must be greater than zero",
        )
        .into());
    }

    fs::create_dir_all(&config.directory)?;
    let writer = RotatingMakeWriter::new(
        &config.directory,
        config.max_file_size_bytes,
        config.max_archive_files,
    )?;
    let filter = EnvFilter::try_new(&config.filter)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_timer(UtcTime::rfc_3339())
            .with_current_span(false)
            .with_span_list(false)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(false)
            .with_writer(writer),
    );
    subscriber
        .with(filter)
        .try_init()
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

#[derive(Clone)]
struct RotatingMakeWriter {
    inner: Arc<Mutex<RotatingFile>>,
}

impl RotatingMakeWriter {
    fn new(directory: &Path, max_size: u64, max_archives: usize) -> io::Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(RotatingFile::open(
                directory,
                max_size,
                max_archives,
            )?)),
        })
    }
}

struct LockedWriter {
    inner: Arc<Mutex<RotatingFile>>,
    pending: Vec<u8>,
}

impl<'a> MakeWriter<'a> for RotatingMakeWriter {
    type Writer = LockedWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LockedWriter {
            inner: Arc::clone(&self.inner),
            pending: Vec::new(),
        }
    }
}

impl Write for LockedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buffer);
        while let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
            let record = self.pending.drain(..=newline).collect::<Vec<_>>();
            let mut writer = self
                .inner
                .lock()
                .map_err(|_| io::Error::other("log writer lock poisoned"))?;
            writer.write_all(&record)?;
            writer.flush()?;
        }
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_pending()
    }
}

impl LockedWriter {
    fn flush_pending(&mut self) -> io::Result<()> {
        if !self.pending.is_empty() {
            let mut writer = self
                .inner
                .lock()
                .map_err(|_| io::Error::other("log writer lock poisoned"))?;
            writer.write_all(&self.pending)?;
            writer.flush()?;
            self.pending.clear();
        }
        Ok(())
    }
}

impl Drop for LockedWriter {
    fn drop(&mut self) {
        let _ = self.flush_pending();
    }
}

struct RotatingFile {
    directory: PathBuf,
    active_path: PathBuf,
    file: Option<File>,
    current_size: u64,
    max_size: u64,
    max_archives: usize,
}

impl RotatingFile {
    fn open(directory: &Path, max_size: u64, max_archives: usize) -> io::Result<Self> {
        fs::create_dir_all(directory)?;
        let active_path = directory.join(LOG_FILE_NAME);
        let current_size = fs::metadata(&active_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&active_path)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            active_path,
            file: Some(file),
            current_size,
            max_size,
            max_archives,
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        if let Some(mut file) = self.file.take() {
            file.flush()?;
            drop(file);
        }

        let result = self.rotate_files();
        let reopen_result = self.reopen_active();
        match (result, reopen_result) {
            (Err(rotation_error), Err(reopen_error)) => Err(io::Error::new(
                reopen_error.kind(),
                format!(
                    "log rotation failed ({rotation_error}) and active log could not be reopened ({reopen_error})"
                ),
            )),
            (Err(rotation_error), Ok(())) => Err(rotation_error),
            (Ok(()), Err(reopen_error)) => Err(reopen_error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn rotate_files(&mut self) -> io::Result<()> {
        if self.max_archives == 0 {
            if self.active_path.exists() {
                fs::remove_file(&self.active_path)?;
            }
            return Ok(());
        }

        let oldest = self.archive_path(self.max_archives);
        if oldest.exists() {
            fs::remove_file(oldest)?;
        }

        for index in (1..self.max_archives).rev() {
            let from = self.archive_path(index);
            if from.exists() {
                fs::rename(from, self.archive_path(index + 1))?;
            }
        }

        if self.active_path.exists() {
            fs::rename(&self.active_path, self.archive_path(1))?;
        }
        Ok(())
    }

    fn reopen_active(&mut self) -> io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.active_path)?;
        self.current_size = file.metadata()?.len();
        self.file = Some(file);
        Ok(())
    }

    fn archive_path(&self, index: usize) -> PathBuf {
        self.directory.join(format!("avrig.{index}.log"))
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.current_size > 0
            && self.current_size.saturating_add(buffer.len() as u64) > self.max_size
        {
            self.rotate()?;
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("active log file is not open"))?;
        file.write_all(buffer)?;
        self.current_size = self.current_size.saturating_add(buffer.len() as u64);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        RotatingFile::flush(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_by_size_with_expected_file_names_and_retention() {
        let directory = tempfile::tempdir().unwrap();
        let mut log = RotatingFile::open(directory.path(), 12, 2).unwrap();

        log.write_all(b"first line\n").unwrap();
        log.write_all(b"second line\n").unwrap();
        log.write_all(b"third line\n").unwrap();
        log.flush().unwrap();

        assert_eq!(
            fs::read(directory.path().join("avrig.log")).unwrap(),
            b"third line\n"
        );
        assert_eq!(
            fs::read(directory.path().join("avrig.1.log")).unwrap(),
            b"second line\n"
        );
        assert_eq!(
            fs::read(directory.path().join("avrig.2.log")).unwrap(),
            b"first line\n"
        );
    }

    #[test]
    fn starts_with_the_required_active_file_name() {
        let directory = tempfile::tempdir().unwrap();
        let _log = RotatingFile::open(directory.path(), 128, 1).unwrap();
        assert!(directory.path().join("avrig.log").exists());
    }

    #[test]
    fn appends_to_existing_active_log_before_rotating() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("avrig.log"), b"existing\n").unwrap();
        let mut log = RotatingFile::open(directory.path(), 64, 1).unwrap();
        log.write_all(b"appended\n").unwrap();
        log.flush().unwrap();
        assert_eq!(
            fs::read(directory.path().join("avrig.log")).unwrap(),
            b"existing\nappended\n"
        );
    }

    #[test]
    fn keeps_a_formatted_record_together_when_rotation_occurs() {
        let directory = tempfile::tempdir().unwrap();
        let writer = RotatingMakeWriter::new(directory.path(), 16, 2).unwrap();
        let mut event_writer = writer.make_writer();
        event_writer
            .write_all(b"2026-01-01T00:00:00Z INFO ")
            .unwrap();
        event_writer.write_all(b"avrig: first record\n").unwrap();
        event_writer.flush().unwrap();
        drop(event_writer);

        let mut next_writer = writer.make_writer();
        next_writer
            .write_all(b"2026-01-01T00:00:01Z INFO ")
            .unwrap();
        next_writer.write_all(b"avrig: second record\n").unwrap();
        next_writer.flush().unwrap();

        assert_eq!(
            fs::read(directory.path().join("avrig.1.log")).unwrap(),
            b"2026-01-01T00:00:00Z INFO avrig: first record\n"
        );
        assert_eq!(
            fs::read(directory.path().join("avrig.log")).unwrap(),
            b"2026-01-01T00:00:01Z INFO avrig: second record\n"
        );
    }
}
