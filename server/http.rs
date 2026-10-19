//! Minimal, defensive HTTP/1.1 request parser and response writer.
//!
//! Everything here works on any `BufRead` / `Write` so it can be unit tested
//! with in-memory cursors. All sizes are capped *before* allocating, header
//! names are matched case-insensitively with exact equality, and ambiguous
//! framing (conflicting `Content-Length`, `Content-Length` together with
//! `Transfer-Encoding`, unknown transfer codings, obs-fold, whitespace before
//! the colon) is rejected to avoid request smuggling.

use std::io::{self, BufRead, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Parser limits.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Max bytes in the request line or any single header / chunk-size line
    /// (including the line terminator).
    pub max_line: usize,
    /// Max number of header fields (trailers count too).
    pub max_headers: usize,
    /// Max decoded body size.
    pub max_body: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_line: 8 * 1024,
            max_headers: 100,
            max_body: 4 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    Http10,
    Http11,
}

#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    pub version: Version,
    /// (name as sent, value with surrounding OWS trimmed)
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// First header value with this (case-insensitive) name.
    #[cfg(test)]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Whether the client allows the connection to be reused after this
    /// request (HTTP/1.1 default on, HTTP/1.0 default off).
    pub fn wants_keep_alive(&self) -> bool {
        let mut close = false;
        let mut keep = false;
        for (n, v) in &self.headers {
            if n.eq_ignore_ascii_case("connection") {
                for tok in v.split(',') {
                    let tok = tok.trim();
                    if tok.eq_ignore_ascii_case("close") {
                        close = true;
                    } else if tok.eq_ignore_ascii_case("keep-alive") {
                        keep = true;
                    }
                }
            }
        }
        if close {
            return false;
        }
        match self.version {
            Version::Http11 => true,
            Version::Http10 => keep,
        }
    }
}

#[derive(Debug)]
pub enum ParseError {
    /// Connection closed cleanly before any byte of a request.
    Closed,
    /// I/O error (including timeouts) while reading.
    Io(io::Error),
    BadRequest(&'static str),
    UriTooLong,
    HeadersTooLarge,
    PayloadTooLarge,
    NotImplemented(&'static str),
    VersionNotSupported,
}

impl ParseError {
    /// HTTP status to answer with, or `None` when nothing should be sent.
    pub fn status(&self) -> Option<u16> {
        match self {
            ParseError::Closed => None,
            ParseError::Io(e) if is_timeout(e) => Some(408),
            ParseError::Io(_) => None,
            ParseError::BadRequest(_) => Some(400),
            ParseError::UriTooLong => Some(414),
            ParseError::HeadersTooLarge => Some(431),
            ParseError::PayloadTooLarge => Some(413),
            ParseError::NotImplemented(_) => Some(501),
            ParseError::VersionNotSupported => Some(505),
        }
    }

    pub fn message(&self) -> String {
        match self {
            ParseError::Closed => "connection closed".into(),
            ParseError::Io(e) if is_timeout(e) => "request timed out".into(),
            ParseError::Io(e) => format!("i/o error: {e}"),
            ParseError::BadRequest(m) => format!("bad request: {m}"),
            ParseError::UriTooLong => "request line too long".into(),
            ParseError::HeadersTooLarge => "request header fields too large".into(),
            ParseError::PayloadTooLarge => "request body too large".into(),
            ParseError::NotImplemented(m) => format!("not implemented: {m}"),
            ParseError::VersionNotSupported => "HTTP version not supported".into(),
        }
    }
}

impl From<io::Error> for ParseError {
    fn from(e: io::Error) -> Self {
        ParseError::Io(e)
    }
}

pub fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

enum LineErr {
    TooLong,
    Eof,
    Io(io::Error),
}

/// Read one line terminated by `\n` (the `\r` before it, if any, is
/// stripped). Never buffers more than `max` bytes. Returns `Eof` if the
/// stream ends before a terminator.
fn read_line_capped<R: BufRead>(r: &mut R, max: usize) -> Result<Vec<u8>, LineErr> {
    let mut out = Vec::new();
    loop {
        let buf = match r.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(LineErr::Io(e)),
        };
        if buf.is_empty() {
            return Err(LineErr::Eof);
        }
        let (chunk, found) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (&buf[..=i], true),
            None => (buf, false),
        };
        if out.len() + chunk.len() > max {
            return Err(LineErr::TooLong);
        }
        out.extend_from_slice(chunk);
        let n = chunk.len();
        r.consume(n);
        if found {
            out.pop(); // '\n'
            if out.last() == Some(&b'\r') {
                out.pop();
            }
            return Ok(out);
        }
    }
}

fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

fn line_err(e: LineErr, too_long: ParseError, first: bool) -> ParseError {
    match e {
        LineErr::TooLong => too_long,
        LineErr::Eof if first => ParseError::Closed,
        LineErr::Eof => ParseError::BadRequest("unexpected end of stream"),
        LineErr::Io(e) => ParseError::Io(e),
    }
}

/// Parse header fields up to (and consuming) the empty line.
fn read_fields<R: BufRead>(
    r: &mut R,
    lim: &Limits,
    out: &mut Vec<(String, String)>,
) -> Result<(), ParseError> {
    loop {
        let line = read_line_capped(r, lim.max_line)
            .map_err(|e| line_err(e, ParseError::HeadersTooLarge, false))?;
        if line.is_empty() {
            return Ok(());
        }
        if out.len() >= lim.max_headers {
            return Err(ParseError::HeadersTooLarge);
        }
        if line[0] == b' ' || line[0] == b'\t' {
            return Err(ParseError::BadRequest("obsolete header line folding"));
        }
        let colon = line
            .iter()
            .position(|&b| b == b':')
            .ok_or(ParseError::BadRequest("header without ':'"))?;
        let name = &line[..colon];
        if name.is_empty() || !name.iter().all(|&b| is_tchar(b)) {
            return Err(ParseError::BadRequest("invalid header name"));
        }
        let value = &line[colon + 1..];
        if value
            .iter()
            .any(|&b| b == 0 || b == b'\r' || b == b'\n' || (b < 0x20 && b != b'\t') || b == 0x7f)
        {
            return Err(ParseError::BadRequest("invalid header value"));
        }
        let value = String::from_utf8_lossy(value)
            .trim_matches(|c| c == ' ' || c == '\t')
            .to_string();
        // tchar is ASCII so this is lossless.
        out.push((String::from_utf8_lossy(name).into_owned(), value));
    }
}

fn read_exact_capped<R: Read>(r: &mut R, n: usize, out: &mut Vec<u8>) -> Result<(), ParseError> {
    let start = out.len();
    out.resize(start + n, 0);
    match r.read_exact(&mut out[start..]) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            Err(ParseError::BadRequest("body shorter than declared"))
        }
        Err(e) => Err(ParseError::Io(e)),
    }
}

fn read_chunked<R: BufRead>(r: &mut R, lim: &Limits) -> Result<Vec<u8>, ParseError> {
    let mut body = Vec::new();
    loop {
        let line = read_line_capped(r, lim.max_line)
            .map_err(|e| line_err(e, ParseError::BadRequest("chunk size line too long"), false))?;
        let size_part = match line.iter().position(|&b| b == b';') {
            Some(i) => &line[..i],
            None => &line[..],
        };
        // Allow BWS before a chunk extension, nothing else.
        let size_part = size_part
            .iter()
            .rposition(|&b| b != b' ' && b != b'\t')
            .map(|i| &size_part[..=i])
            .unwrap_or(&[]);
        if size_part.is_empty()
            || size_part.len() > 16
            || !size_part.iter().all(u8::is_ascii_hexdigit)
        {
            return Err(ParseError::BadRequest("invalid chunk size"));
        }
        // <= 16 hex digits always fits in u64.
        let mut size: u64 = 0;
        for &b in size_part {
            let d = (b as char).to_digit(16).unwrap_or(0) as u64;
            size = (size << 4) | d;
        }
        if size == 0 {
            // Trailers: parsed with the same caps, then dropped.
            let mut trailers = Vec::new();
            read_fields(r, lim, &mut trailers)?;
            return Ok(body);
        }
        let total = (body.len() as u64).saturating_add(size);
        if total > lim.max_body as u64 {
            return Err(ParseError::PayloadTooLarge);
        }
        read_exact_capped(r, size as usize, &mut body)?;
        let crlf = read_line_capped(r, 2)
            .map_err(|e| line_err(e, ParseError::BadRequest("missing CRLF after chunk"), false))?;
        if !crlf.is_empty() {
            return Err(ParseError::BadRequest("missing CRLF after chunk"));
        }
    }
}

/// Read and validate one request (head + body).
#[cfg(test)]
pub fn read_request<R: BufRead>(r: &mut R, lim: &Limits) -> Result<Request, ParseError> {
    read_request_with(r, lim, &mut || Ok(()))
}

/// Like [`read_request`], calling `on_continue` (to send `100 Continue`)
/// before reading the body when the client sent `Expect: 100-continue` and
/// the framing was accepted.
pub fn read_request_with<R: BufRead>(
    r: &mut R,
    lim: &Limits,
    on_continue: &mut dyn FnMut() -> io::Result<()>,
) -> Result<Request, ParseError> {
    // RFC 9112 2.2: ignore at least one empty line before the request-line.
    let mut line;
    let mut first = true;
    let mut skipped = 0;
    loop {
        line = read_line_capped(r, lim.max_line)
            .map_err(|e| line_err(e, ParseError::UriTooLong, first))?;
        first = false;
        if !line.is_empty() {
            break;
        }
        skipped += 1;
        if skipped > 4 {
            return Err(ParseError::BadRequest("empty request line"));
        }
    }
    let line =
        std::str::from_utf8(&line).map_err(|_| ParseError::BadRequest("non-UTF-8 request line"))?;
    let mut parts = line.split(' ');
    let (method, target, version) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(t), Some(v), None) if !m.is_empty() && !t.is_empty() => (m, t, v),
        _ => return Err(ParseError::BadRequest("malformed request line")),
    };
    if !method.bytes().all(is_tchar) {
        return Err(ParseError::BadRequest("invalid method"));
    }
    if target.bytes().any(|b| b <= 0x20 || b == 0x7f) {
        return Err(ParseError::BadRequest("invalid request target"));
    }
    let version = match version {
        "HTTP/1.1" => Version::Http11,
        "HTTP/1.0" => Version::Http10,
        v if v.starts_with("HTTP/") => return Err(ParseError::VersionNotSupported),
        _ => return Err(ParseError::BadRequest("malformed request line")),
    };

    let mut headers = Vec::new();
    read_fields(r, lim, &mut headers)?;

    // ---- framing ----
    let mut content_length: Option<u64> = None;
    let mut transfer_encoding: Option<&str> = None;
    for (n, v) in &headers {
        if n.eq_ignore_ascii_case("content-length") {
            if v.is_empty() || v.len() > 20 || !v.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ParseError::BadRequest("invalid Content-Length"));
            }
            let cl: u64 = v
                .parse()
                .map_err(|_| ParseError::BadRequest("invalid Content-Length"))?;
            match content_length {
                Some(prev) if prev != cl => {
                    return Err(ParseError::BadRequest("conflicting Content-Length"));
                }
                _ => content_length = Some(cl),
            }
        } else if n.eq_ignore_ascii_case("transfer-encoding") {
            if transfer_encoding.is_some() {
                return Err(ParseError::BadRequest("multiple Transfer-Encoding headers"));
            }
            transfer_encoding = Some(v.as_str());
        }
    }

    let expects_continue = version == Version::Http11
        && headers.iter().any(|(n, v)| {
            n.eq_ignore_ascii_case("expect") && v.eq_ignore_ascii_case("100-continue")
        });
    let mut maybe_continue = || -> Result<(), ParseError> {
        if expects_continue {
            on_continue()?;
        }
        Ok(())
    };

    let body = match (transfer_encoding, content_length) {
        (Some(_), Some(_)) => {
            return Err(ParseError::BadRequest(
                "both Content-Length and Transfer-Encoding",
            ));
        }
        (Some(_), None) if version == Version::Http10 => {
            return Err(ParseError::BadRequest("Transfer-Encoding in HTTP/1.0"));
        }
        (Some(te), None) => {
            if !te.eq_ignore_ascii_case("chunked") {
                return Err(ParseError::NotImplemented("unsupported Transfer-Encoding"));
            }
            maybe_continue()?;
            read_chunked(r, lim)?
        }
        (None, Some(cl)) => {
            if cl > lim.max_body as u64 {
                return Err(ParseError::PayloadTooLarge);
            }
            let mut b = Vec::new();
            if cl > 0 {
                maybe_continue()?;
            }
            read_exact_capped(r, cl as usize, &mut b)?;
            b
        }
        (None, None) => Vec::new(),
    };

    let path = target.split(['?', '#']).next().unwrap_or("").to_string();
    Ok(Request {
        method: method.to_string(),
        path,
        version,
        headers,
        body,
    })
}

pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        414 => "URI Too Long",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

/// A complete (non-streamed) response.
#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, v: &serde_json::Value) -> Self {
        Self {
            status,
            headers: vec![("Content-Type", "application/json".into())],
            body: v.to_string().into_bytes(),
        }
    }

    pub fn empty(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    pub fn write_to<W: Write>(
        &self,
        w: &mut W,
        keep_alive: bool,
        extra: &[(&'static str, String)],
    ) -> io::Result<()> {
        let mut head = format!("HTTP/1.1 {} {}\r\n", self.status, reason(self.status));
        for (n, v) in self.headers.iter().chain(extra.iter()) {
            head.push_str(n);
            head.push_str(": ");
            head.push_str(v);
            head.push_str("\r\n");
        }
        head.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        head.push_str(if keep_alive {
            "Connection: keep-alive\r\n\r\n"
        } else {
            "Connection: close\r\n\r\n"
        });
        let mut out = head.into_bytes();
        out.extend_from_slice(&self.body);
        w.write_all(&out)?;
        w.flush()
    }
}

/// Write the head of an SSE response (close-delimited, no Content-Length).
pub fn write_sse_head<W: Write>(w: &mut W, extra: &[(&'static str, String)]) -> io::Result<()> {
    let mut head = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n",
    );
    for (n, v) in extra {
        head.push_str(n);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    head.push_str("Connection: close\r\n\r\n");
    w.write_all(head.as_bytes())?;
    w.flush()
}

/// Write one SSE `data:` event. JSON produced by serde_json never contains a
/// raw newline, so a single `data:` line is always correct framing.
pub fn write_sse_data<W: Write>(w: &mut W, data: &str) -> io::Result<()> {
    debug_assert!(!data.contains('\n'));
    let mut s = String::with_capacity(data.len() + 8);
    s.push_str("data: ");
    s.push_str(data);
    s.push_str("\n\n");
    w.write_all(s.as_bytes())?;
    w.flush()
}

/// `Read` adapter over a `TcpStream` that enforces an overall deadline (for
/// slowloris protection) in addition to a per-read timeout.
pub struct TimedReader<'a> {
    stream: &'a TcpStream,
    per_read: Duration,
    deadline: Option<Instant>,
}

impl<'a> TimedReader<'a> {
    pub fn new(stream: &'a TcpStream, per_read: Duration) -> Self {
        Self {
            stream,
            per_read,
            deadline: None,
        }
    }

    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
    }

    pub fn set_per_read(&mut self, d: Duration) {
        self.per_read = d;
    }
}

impl Read for TimedReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut timeout = self.per_read;
        if let Some(dl) = self.deadline {
            let now = Instant::now();
            if now >= dl {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "request deadline"));
            }
            timeout = timeout.min(dl - now);
        }
        // A zero timeout means "block forever" to the OS; never pass it.
        let timeout = timeout.max(Duration::from_millis(1));
        self.stream.set_read_timeout(Some(timeout))?;
        let mut s = self.stream;
        s.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(raw: &[u8]) -> Result<Request, ParseError> {
        read_request(&mut Cursor::new(raw.to_vec()), &Limits::default())
    }

    fn parse_lim(raw: &[u8], lim: &Limits) -> Result<Request, ParseError> {
        read_request(&mut Cursor::new(raw.to_vec()), lim)
    }

    fn status(r: Result<Request, ParseError>) -> Option<u16> {
        match r {
            Ok(_) => Some(200),
            Err(e) => e.status(),
        }
    }

    #[test]
    fn simple_post() {
        let r = parse(
            b"POST /v1/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\nhello",
        )
        .unwrap();
        assert_eq!(r.method, "POST");
        assert_eq!(r.path, "/v1/completions");
        assert_eq!(r.body, b"hello");
        assert_eq!(r.header("host"), Some("a"));
        assert!(r.wants_keep_alive());
    }

    #[test]
    fn bare_lf_and_leading_empty_line() {
        let r = parse(b"\r\nGET /health HTTP/1.0\nConnection: keep-alive\n\n").unwrap();
        assert_eq!(r.path, "/health");
        assert!(r.wants_keep_alive());
        let r = parse(b"GET / HTTP/1.0\r\n\r\n").unwrap();
        assert!(!r.wants_keep_alive());
        let r = parse(b"GET / HTTP/1.1\r\nConnection: Close\r\n\r\n").unwrap();
        assert!(!r.wants_keep_alive());
    }

    #[test]
    fn closed_before_request() {
        assert!(matches!(parse(b""), Err(ParseError::Closed)));
        assert_eq!(status(parse(b"")), None);
        // Truncated head is a 400, not a silent close.
        assert_eq!(status(parse(b"GET / HTTP/1.1\r\nHost: x")), Some(400));
    }

    #[test]
    fn content_length_header_is_exact_name() {
        // `X-Content-Length` must not be treated as Content-Length.
        let r = parse(b"POST / HTTP/1.1\r\nX-Content-Length: 5\r\n\r\nhello").unwrap();
        assert!(r.body.is_empty());
        let r = parse(b"POST / HTTP/1.1\r\ncOnTeNt-LeNgTh: 2\r\n\r\nhi").unwrap();
        assert_eq!(r.body, b"hi");
    }

    #[test]
    fn oversized_request_line() {
        let mut raw = b"GET /".to_vec();
        raw.extend(std::iter::repeat_n(b'a', 9000));
        raw.extend_from_slice(b" HTTP/1.1\r\n\r\n");
        assert_eq!(status(parse(&raw)), Some(414));
    }

    #[test]
    fn oversized_header_line_and_count() {
        let mut raw = b"GET / HTTP/1.1\r\nX-Big: ".to_vec();
        raw.extend(std::iter::repeat_n(b'a', 9000));
        raw.extend_from_slice(b"\r\n\r\n");
        assert_eq!(status(parse(&raw)), Some(431));

        let mut raw = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..101 {
            raw.extend_from_slice(format!("X-H{i}: v\r\n").as_bytes());
        }
        raw.extend_from_slice(b"\r\n");
        assert_eq!(status(parse(&raw)), Some(431));
    }

    #[test]
    fn huge_content_length_does_not_allocate() {
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n"
            )),
            Some(413)
        );
        // Larger than u64: rejected as malformed.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n"
            )),
            Some(400)
        );
        let lim = Limits {
            max_body: 4,
            ..Limits::default()
        };
        assert_eq!(
            status(parse_lim(
                b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello",
                &lim
            )),
            Some(413)
        );
    }

    #[test]
    fn bad_content_length() {
        for v in ["-1", "+5", "5, 5", "0x10", "", "1 2", "abc"] {
            let raw = format!("POST / HTTP/1.1\r\nContent-Length: {v}\r\n\r\n");
            assert_eq!(status(parse(raw.as_bytes())), Some(400), "value {v:?}");
        }
        // Short body.
        assert_eq!(
            status(parse(b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc")),
            Some(400)
        );
    }

    #[test]
    fn smuggling_cases() {
        // Conflicting CL.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabcd"
            )),
            Some(400)
        );
        // Identical duplicate CL is allowed.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc"
            )),
            Some(200)
        );
        // CL + TE.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )),
            Some(400)
        );
        // Duplicate TE.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )),
            Some(400)
        );
        // Unknown / stacked codings.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"
            )),
            Some(501)
        );
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: xchunked\r\n\r\n"
            )),
            Some(501)
        );
        // Whitespace before colon / obs-fold.
        assert_eq!(
            status(parse(b"POST / HTTP/1.1\r\nContent-Length : 3\r\n\r\nabc")),
            Some(400)
        );
        assert_eq!(
            status(parse(b"GET / HTTP/1.1\r\nX-A: a\r\n b\r\n\r\n")),
            Some(400)
        );
        // TE in HTTP/1.0.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )),
            Some(400)
        );
    }

    #[test]
    fn request_line_validation() {
        assert_eq!(status(parse(b"GET  / HTTP/1.1\r\n\r\n")), Some(400));
        assert_eq!(status(parse(b"GET /\r\n\r\n")), Some(400));
        assert_eq!(status(parse(b"GET / HTTP/2.0\r\n\r\n")), Some(505));
        assert_eq!(status(parse(b"G(T / HTTP/1.1\r\n\r\n")), Some(400));
    }

    #[test]
    fn chunked_body() {
        let r = parse(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: Chunked\r\n\r\n5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\nX-Trailer: t\r\n\r\nGET",
        )
        .unwrap();
        assert_eq!(r.body, b"hello world");
    }

    #[test]
    fn chunked_errors() {
        let lim = Limits {
            max_body: 8,
            ..Limits::default()
        };
        // Exceeds body cap across chunks.
        assert_eq!(
            status(parse_lim(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n",
                &lim
            )),
            Some(413)
        );
        // Giant chunk size never allocates.
        assert_eq!(
            status(parse(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\n"
            )),
            Some(413)
        );
        // Overlong chunk size / bad hex / missing CRLF / truncated.
        for body in [
            &b"fffffffffffffffff\r\n"[..],
            b"zz\r\n",
            b"-1\r\n",
            b"\r\n",
            b"5\r\nhelloXX0\r\n\r\n",
            b"5\r\nhel",
        ] {
            let mut raw = b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            raw.extend_from_slice(body);
            assert_eq!(
                status(parse(&raw)),
                Some(400),
                "{:?}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn pipelined_requests() {
        let raw = b"POST /a HTTP/1.1\r\nContent-Length: 2\r\n\r\nhiGET /b HTTP/1.1\r\n\r\n";
        let mut c = Cursor::new(raw.to_vec());
        let a = read_request(&mut c, &Limits::default()).unwrap();
        let b = read_request(&mut c, &Limits::default()).unwrap();
        assert_eq!((a.path.as_str(), b.path.as_str()), ("/a", "/b"));
        assert!(matches!(
            read_request(&mut c, &Limits::default()),
            Err(ParseError::Closed)
        ));
    }

    #[test]
    fn response_and_sse_framing() {
        let mut out = Vec::new();
        Response::json(200, &serde_json::json!({"a":1}))
            .write_to(&mut out, false, &[])
            .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Length: 7\r\n"));
        assert!(s.contains("Connection: close\r\n\r\n{\"a\":1}"));

        let mut out = Vec::new();
        write_sse_head(&mut out, &[]).unwrap();
        write_sse_data(&mut out, "{\"x\":\"a\\nb\"}").unwrap();
        write_sse_data(&mut out, "[DONE]").unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("Content-Type: text/event-stream\r\n"));
        assert!(s.ends_with("\r\n\r\ndata: {\"x\":\"a\\nb\"}\n\ndata: [DONE]\n\n"));
    }
}
