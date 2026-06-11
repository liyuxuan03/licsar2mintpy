#!/usr/bin/env python3
"""
LiCSAR InSAR Data Downloader

This version uses LiCSAR manifest-style entries by default.

For epochs and interferograms, the script directly opens:
   .../{orbit}/{frame}/epochs/{YYYYMMDD}
   .../{orbit}/{frame}/interferograms/{YYYYMMDD}_{YYYYMMDD}

It intentionally does NOT try directory-style child URLs such as:
   .../{orbit}/{frame}/epochs/{YYYYMMDD}/
   .../{orbit}/{frame}/interferograms/{YYYYMMDD}_{YYYYMMDD}/

The manifest page may be text/plain or HTML-like, and contains the real product
file names or links. The script parses that manifest and then downloads the
resolved product URLs.

Examples:
  python download.py --list-orbits
  python download.py --orbits 172 --list-frames
  python download.py --frames 172A_05661_131313 --list-epochs
  python download.py --orbits 172 --products unw --dates 20200101-20201231
  python download.py --frames 172A_05661_131313 --products metadata
  python download.py --frames 124A_06996_091406 --dates 20240101-20250101 --products all --proxy http://127.0.0.1:7890 --no-verify-ssl
  python download.py --orbits 1 --products unw --dry-run
"""

import argparse
import html as html_lib
import os
import re
import socket
import sys
import time
from collections import namedtuple
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

# ---------------------------------------------------------------------------
# Required dependency
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    sys.exit(
        "Error: 'requests' is required. Install it with:\n"
        "    pip install requests"
    )

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/"
ALT_BASE_URL = "https://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/"

# Product type -> file-matching predicate.
# Use endswith() instead of exact names because many LiCSAR metadata files are
# frame-prefixed, e.g. 124A_06996_091406.geo.E.tif.
PRODUCTS = {
    "unw":      lambda name: name.endswith(".geo.unw.tif"),
    "cc":       lambda name: name.endswith(".geo.cc.tif"),
    "diff_pha": lambda name: name.endswith(".geo.diff_pha.tif"),
    "hgt":      lambda name: name.endswith(".geo.hgt.tif"),
    "dem":      lambda name: name.endswith(".geo.hgt.tif"),
    "los_E":    lambda name: name.endswith(".geo.E.tif"),
    "los_N":    lambda name: name.endswith(".geo.N.tif"),
    "los_U":    lambda name: name.endswith(".geo.U.tif"),
    "inc":      lambda name: name.endswith(".geo.inc.tif"),
    "png":      lambda name: name.lower().endswith(".png"),
    "all":      lambda name: True,
}

# 'metadata' is a group name. This group includes common MintPy/LiCSAR
# metadata-related files. In addition, --products all will still match every
# file in metadata/.
def is_metadata_product(name):
    lower = name.lower()
    base = os.path.basename(lower)
    return (
        lower.endswith(".geo.e.tif")
        or lower.endswith(".geo.n.tif")
        or lower.endswith(".geo.u.tif")
        or lower.endswith(".geo.hgt.tif")
        or lower.endswith(".geo.inc.tif")
        or lower.endswith(".geo.landmask.tif")
        or lower.endswith(".azirg.csv")
        or base in {"metadata.txt", "baselines", "network.png", "lackifg.txt"}
        or base.endswith("-poly.txt")
    )


# Download task
Task = namedtuple("Task", ["url", "dest_path", "size_bytes", "description"])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_size(text):
    """Parse Apache size strings like '123K', '45M', '1.2G' into bytes."""
    text = (text or "").strip()
    if text in ("", "-"):
        return 0

    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    match = re.match(r"([\d.]+)\s*([KMGT]?)", text, flags=re.I)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        return int(value * units.get(unit, 1))

    try:
        return int(text)
    except ValueError:
        return 0


def parse_orbit_list(text):
    """Parse '70,126,172' or '1-175' into a set of integers."""
    if text is None:
        return None

    result = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def parse_date_range(text):
    """Parse '20180101-20181231' or '20180101' into (start, end_or_none)."""
    if text is None:
        return None, None
    text = text.strip()
    if "-" in text:
        parts = text.split("-", 1)
        return parts[0].strip(), parts[1].strip()
    return text, None


def safe_basename_from_url_or_name(value):
    """Return a safe local filename from a URL, href, or plain filename."""
    value = (value or "").strip()
    value = value.split("?", 1)[0].split("#", 1)[0]
    path = urlparse(value).path if re.match(r"^https?://", value, re.I) else value
    name = os.path.basename(path.rstrip("/"))
    return unquote(name)


def strip_html_tags(text):
    """Small fallback helper for plain text manifests containing simple HTML."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return text.strip()


def is_probable_file_name(name):
    """Heuristic to distinguish real product files from manifest entries."""
    name = safe_basename_from_url_or_name(name)
    if not name:
        return False
    if name in (".", "..", "Parent Directory"):
        return False
    if name.startswith("?"):
        return False
    if name.endswith("/"):
        return False
    # Manifest entries are usually 20240107 or 20240107_20240119 with no dot.
    # Real products almost always have an extension, e.g. .tif, .png, .nc, .xml.
    return "." in name


def dedup_entries(entries):
    """Deduplicate file entries by URL if available, otherwise by name."""
    out = []
    seen = set()
    for e in entries:
        key = e.get("url") or e.get("href") or e.get("name")
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# LiCSAR Downloader
# ---------------------------------------------------------------------------

class LiCSARDownloader:
    """Downloader for LiCSAR InSAR products from the JASMIN/CEDA server."""

    def __init__(
        self,
        output_dir="./downloads",
        proxy=None,
        delay=0.5,
        max_retries=3,
        resume=True,
        quiet=False,
        dry_run=False,
        timeout=60,
        verify_ssl=True,
    ):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_retries = max_retries
        self.resume = resume
        self.quiet = quiet
        self.dry_run = dry_run
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers["User-Agent"] = "LiCSAR-Downloader/1.1 (Python)"

        # Never use system proxy settings from environment variables unless
        # explicitly provided by --proxy.
        self.session.trust_env = False

        if proxy:
            if proxy.startswith("socks"):
                try:
                    import socks  # noqa: F401
                except ImportError:
                    sys.exit(
                        "Error: SOCKS proxy requires 'PySocks'. Install it:\n"
                        "    pip install PySocks"
                    )
            self.session.proxies = {"http": proxy, "https": proxy}

        self.verify_ssl = verify_ssl
        if not verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _list_url(self, rel_path=""):
        """Build the full URL for a directory or manifest listing."""
        return urljoin(BASE_URL, rel_path) if rel_path else BASE_URL

    def _file_url(self, *parts):
        """Build absolute URL to a file from path parts."""
        return urljoin(BASE_URL, "/".join(str(p).strip("/") for p in parts))

    @staticmethod
    def _resolve_href(base_url, href, force_child_context=False):
        """Resolve a href relative to base_url.

        If force_child_context=True and base_url has no trailing slash, append
        '/' before urljoin(). This is important for LiCSAR manifest entries:
            .../epochs/20240107
        contains links that should resolve as:
            .../epochs/20240107/<filename>
        """
        href = html_lib.unescape((href or "").strip())
        if not href:
            return ""
        if href.startswith("//"):
            return "https:" + href
        if re.match(r"^https?://", href, re.I):
            return href
        if force_child_context:
            base_url = base_url.rstrip("/") + "/"
        return urljoin(base_url, href)

    def _get(self, url):
        """GET a URL with retry and exponential backoff."""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0 and not self.quiet:
                    print(f"  Retry {attempt}/{self.max_retries} for {url}")

                resp = self.session.get(
                    url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                resp.raise_for_status()
                return resp

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_exc = e

                if isinstance(e, requests.HTTPError) and e.response is not None:
                    code = e.response.status_code
                    # Do not retry ordinary client-side errors.
                    if code == 404 or code < 500:
                        raise

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise last_exc

    @staticmethod
    def _diagnose_error(exc):
        """Return a human-readable diagnosis string for a request exception."""
        msg = str(exc)

        if isinstance(exc, requests.ConnectionError):
            if "ProxyError" in type(exc).__name__ or "Proxy" in msg:
                if "FileNotFoundError" in msg or "No such file" in msg:
                    return (
                        "Proxy client is not running. Start your proxy software "
                        "first, or verify the proxy address."
                    )
                if "ConnectionRefusedError" in msg or "Connection refused" in msg:
                    return (
                        "Proxy refused the connection. Check that the proxy "
                        "address and port are correct."
                    )
                if "SSLEOFError" in msg or "SSL" in msg:
                    return (
                        "Proxy SSL handshake failed. The proxy may not support "
                        "HTTPS tunneling (CONNECT method). Try a different proxy "
                        "or use --no-verify-ssl."
                    )
                return (
                    f"Proxy connection failed. Check that your proxy is running "
                    f"and the address is correct. ({msg[:120]})"
                )

            if "Name or service not known" in msg or "getaddrinfo" in msg:
                return "DNS resolution failed. Check your network connection and DNS settings."

            if "timed out" in msg.lower() or "Timeout" in msg:
                return (
                    "Connection timed out. The server may be unreachable. "
                    "If you are in China, a proxy/VPN may be required."
                )

            return f"Connection failed: {msg[:200]}"

        if isinstance(exc, requests.Timeout):
            return (
                "Request timed out. The server may be slow or unreachable. "
                "Try increasing --timeout or use a proxy."
            )

        return msg[:200]

    @staticmethod
    def _test_proxy_port(host, port):
        """Test if a TCP port is open on the proxy host using raw socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _test_connection(self):
        """Test connectivity to the base URL and report diagnostics."""
        if self.quiet:
            return True

        print(f"Testing connection to {BASE_URL} ...")

        if self.session.proxies:
            proxy_url = self.session.proxies.get("https", "")
            print(f"  Proxy: {proxy_url}")

            m = re.match(r"(?:https?|socks[45]?)://([^:/]+):(\d+)", proxy_url)
            if m:
                host, port = m.group(1), int(m.group(2))
                if not self._test_proxy_port(host, port):
                    print(f"  ERROR: No service listening at {host}:{port}")
                    print(f"  -> Make sure your proxy client is running on port {port}")
                    print(f"  -> Try: netstat -an | findstr {port}")
                    print("  -> Common proxy ports: Clash=7890, v2ray=10809, SS=1080\n")
                    return False
                print(f"  Proxy port {host}:{port} is open")
            else:
                print(f"  WARNING: Could not parse proxy URL '{proxy_url}'")

            if proxy_url.startswith("http://"):
                print("  Note: Using HTTP proxy for HTTPS target (CONNECT tunnel)")
                print("  If your proxy is SOCKS5, use socks5://127.0.0.1:<port> instead")
        else:
            print("  No proxy configured (direct connection)")

        try:
            resp = self.session.head(
                BASE_URL,
                timeout=min(self.timeout, 15),
                verify=self.verify_ssl,
            )
            print(f"  Server OK (HTTP {resp.status_code})\n")
            return True

        except Exception as e:
            diagnosis = self._diagnose_error(e)
            print(f"  FAILED: {diagnosis}\n")
            if not self.session.proxies:
                print(
                    "Tip: If you need a proxy to access this server, use:\n"
                    "    --proxy http://127.0.0.1:<port>    (HTTP proxy)\n"
                    "    --proxy socks5://127.0.0.1:<port>  (SOCKS5 proxy)"
                )
            else:
                print(
                    "Tip: Check the proxy address and protocol.\n"
                    "  HTTP proxy:   --proxy http://127.0.0.1:<port>\n"
                    "  SOCKS5 proxy: --proxy socks5://127.0.0.1:<port>\n"
                    "  Try a different port or check if your proxy supports HTTPS CONNECT."
                )
            return False

    # ------------------------------------------------------------------
    # HTML directory listing parsing
    # ------------------------------------------------------------------

    def _parse_directory(self, html, debug_hint=""):
        """Parse Apache/Nginx autoindex HTML into entries.

        Returns a list of dicts with keys:
            name, is_dir, href, size_bytes
        """
        if BS4_AVAILABLE:
            entries = self._parse_with_bs4(html)
        else:
            entries = self._parse_with_regex(html)

        if not entries and debug_hint:
            if not self.quiet:
                print(
                    f"    [debug] {debug_hint}: no links found "
                    f"(first 400 chars): {html[:400]!r}"
                )

        return entries

    @staticmethod
    def _parse_with_bs4(html):
        """Parse an Apache/Nginx autoindex page using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        entries = []

        # Apache autoindex commonly uses table rows.
        for row in soup.find_all("tr"):
            link = row.find("a")
            if not link:
                continue

            href = link.get("href", "")
            name = link.get_text(strip=True)

            if not name or name == "Parent Directory" or href.startswith("?"):
                continue

            is_dir = href.endswith("/")
            entry = {
                "name": name.rstrip("/") if is_dir else name,
                "is_dir": is_dir,
                "href": href,
                "size_bytes": 0,
            }

            if not is_dir:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    size_text = cells[2].get_text(strip=True)
                    entry["size_bytes"] = parse_size(size_text)

            entries.append(entry)

        # Some servers use pre/a instead of tr/td.
        if not entries:
            for link in soup.find_all("a"):
                href = link.get("href", "")
                name = link.get_text(strip=True)

                if not name or name == "Parent Directory" or href.startswith("?"):
                    continue

                is_dir = href.endswith("/")
                entries.append(
                    {
                        "name": name.rstrip("/") if is_dir else name,
                        "is_dir": is_dir,
                        "href": href,
                        "size_bytes": 0,
                    }
                )

        return entries

    @staticmethod
    def _parse_with_regex(html):
        """Parse links using stdlib regex."""
        entries = []

        for match in re.finditer(
            r"<a\s+[^>]*href=[\"']([^\"']*)[\"'][^>]*>\s*([^<]+?)\s*</a>",
            html,
            flags=re.I,
        ):
            href = html_lib.unescape(match.group(1).strip())
            name = html_lib.unescape(match.group(2).strip())

            if not name or name == "Parent Directory" or href.startswith("?"):
                continue

            is_dir = href.endswith("/")
            entries.append(
                {
                    "name": name.rstrip("/") if is_dir else name,
                    "is_dir": is_dir,
                    "href": href,
                    "size_bytes": 0,
                }
            )

        return entries

    # ------------------------------------------------------------------
    # Manifest parsing
    # ------------------------------------------------------------------

    def _parse_manifest(self, text, manifest_url, debug_hint=""):
        """Parse a LiCSAR manifest page into real product file entries.

        The manifest may be:
          - text/plain with one file per line,
          - text/plain/HTML-like containing <a href=...> links,
          - an HTML page with anchors.

        Returns entries with keys:
            name, is_dir, href, url, size_bytes
        """
        entries = []

        # 1) First parse any explicit HTML links.
        link_entries = self._parse_directory(text)
        for e in link_entries:
            if e.get("is_dir"):
                continue
            name = safe_basename_from_url_or_name(e.get("name") or e.get("href"))
            if not is_probable_file_name(name):
                continue
            url = self._resolve_href(
                manifest_url,
                e.get("href", ""),
                force_child_context=True,
            )
            entries.append(
                {
                    "name": name,
                    "is_dir": False,
                    "href": e.get("href", ""),
                    "url": url,
                    "size_bytes": e.get("size_bytes", 0),
                }
            )

        # 2) Also scan raw text lines. This catches plain manifest files.
        #    Examples:
        #      20240107.geo.mli.tif
        #      https://.../20240107.geo.mli.tif
        #      <a href="...">20240107.geo.mli.tif</a>
        url_pattern = re.compile(r"https?://[^\s\"'<>]+", flags=re.I)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Extract absolute URLs, if any.
            for url in url_pattern.findall(line):
                name = safe_basename_from_url_or_name(url)
                if is_probable_file_name(name):
                    entries.append(
                        {
                            "name": name,
                            "is_dir": False,
                            "href": url,
                            "url": url,
                            "size_bytes": 0,
                        }
                    )

            # Remove URLs and HTML tags, then parse remaining tokens.
            line_without_urls = url_pattern.sub(" ", line)
            clean = strip_html_tags(line_without_urls)
            if not clean:
                continue

            # Split on whitespace and common separators.
            for token in re.split(r"[\s,;]+", clean):
                token = token.strip().strip("\"'")

                if not token:
                    continue
                if token in ("[ICO]", "[DIR]", "[PARENTDIR]", "[ ]"):
                    continue
                if token.lower() in ("name", "last", "modified", "size", "description"):
                    continue

                name = safe_basename_from_url_or_name(token)
                if not is_probable_file_name(name):
                    continue

                url = self._resolve_href(
                    manifest_url,
                    token,
                    force_child_context=True,
                )
                entries.append(
                    {
                        "name": name,
                        "is_dir": False,
                        "href": token,
                        "url": url,
                        "size_bytes": 0,
                    }
                )

        entries = dedup_entries(entries)

        if not entries and debug_hint and not self.quiet:
            print(
                f"    [debug] {debug_hint}: manifest parsed but no product files found "
                f"(first 300 chars): {text[:300]!r}"
            )

        return entries

    # ------------------------------------------------------------------
    # Directory traversal
    # ------------------------------------------------------------------

    def list_orbits(self):
        """Return sorted list of available orbit numbers."""
        url = self._list_url()
        resp = self._get(url)
        entries = self._parse_directory(resp.text, debug_hint="orbits")

        orbits = []
        for e in entries:
            if e["is_dir"] and e["name"].isdigit():
                orbits.append(int(e["name"]))

        return sorted(orbits)

    def list_frames(self, orbit):
        """Return list of frame names for an orbit."""
        orbit_str = str(orbit)
        url = self._list_url(f"{orbit_str}/")
        resp = self._get(url)
        entries = self._parse_directory(resp.text, debug_hint=f"orbit {orbit}")

        frames = []
        for e in entries:
            if e["is_dir"]:
                name = e["name"]
                if name.startswith(orbit_str.zfill(3)):
                    frames.append(name)

        return sorted(frames)

    def list_epochs(self, orbit, frame):
        """Return list of epoch dates.

        This version reads epoch entries from the parent listing and expects
        each child epoch to be opened later as manifest-style:
          epochs/20240107
        """
        url = self._list_url(f"{orbit}/{frame}/epochs/")
        resp = self._get(url)
        entries = self._parse_directory(resp.text, debug_hint=f"epochs/{orbit}/{frame}")

        dates = []
        non_matching = []

        for e in entries:
            name = e["name"].rstrip("/")
            if re.match(r"^\d{8}$", name):
                dates.append(name)
            else:
                non_matching.append(name)

        dates = sorted(set(dates))

        if not self.quiet:
            dirs = [e for e in entries if e["is_dir"]]
            files = [e for e in entries if not e["is_dir"]]
            print(
                f"    [debug] {url} -> {len(entries)} entries, "
                f"{len(dirs)} dirs, {len(files)} file-like entries, "
                f"{len(dates)} epoch dates matched"
            )
            if non_matching and not dates:
                print(
                    "    [debug] non-matching entry names: "
                    f"{non_matching[:5]}{'...' if len(non_matching) > 5 else ''}"
                )

        return dates

    def list_interferogram_pairs(self, orbit, frame):
        """Return list of (date1, date2) tuples for interferogram pairs.

        This version reads pair entries from the parent listing and expects
        each child interferogram pair to be opened later as manifest-style:
          interferograms/20240107_20240119
        """
        url = self._list_url(f"{orbit}/{frame}/interferograms/")
        resp = self._get(url)
        entries = self._parse_directory(resp.text, debug_hint=f"ifg/{orbit}/{frame}")

        pairs = []
        non_matching = []

        for e in entries:
            name = e["name"].rstrip("/")
            match = re.match(r"^(\d{8})_(\d{8})$", name)
            if match:
                pairs.append((match.group(1), match.group(2)))
            else:
                non_matching.append(name)

        pairs = sorted(set(pairs))

        if not self.quiet:
            dirs = [e for e in entries if e["is_dir"]]
            files = [e for e in entries if not e["is_dir"]]
            print(
                f"    [debug] {url} -> {len(entries)} entries, "
                f"{len(dirs)} dirs, {len(files)} file-like entries, "
                f"{len(pairs)} pairs matched"
            )
            if non_matching and not pairs:
                print(
                    "    [debug] non-matching entry names: "
                    f"{non_matching[:5]}{'...' if len(non_matching) > 5 else ''}"
                )
            if not entries:
                print(
                    "    [debug] WARNING: empty directory listing - "
                    "the server may have returned unexpected HTML"
                )

        return pairs

    def list_metadata_files(self, orbit, frame):
        """Return metadata file entries with absolute URLs."""
        rel_path = f"{orbit}/{frame}/metadata/"
        return self.list_directory_files(rel_path)

    def list_directory_files(self, rel_path):
        """List all files at a relative directory path.

        This function is for a true directory listing. It returns entries with:
            name, is_dir, href, url, size_bytes
        """
        url = self._list_url(rel_path)
        resp = self._get(url)
        entries = self._parse_directory(resp.text, debug_hint=rel_path)

        files = []
        for e in entries:
            if e["is_dir"]:
                continue
            name = safe_basename_from_url_or_name(e["name"])
            if not name or name == "Parent Directory":
                continue

            item = dict(e)
            item["name"] = name
            item["url"] = self._resolve_href(url, e.get("href", ""))
            files.append(item)

        return files

    def list_child_files(self, parent_rel_path, child_name):
        """List files under a child entry using manifest-style only.

        For example:
          parent_rel_path = "124/124A_06996_091406/epochs"
          child_name      = "20240107"

        This function intentionally does NOT try directory-style URLs such as:
          .../epochs/20240107/

        Instead, it directly opens the manifest-style URL:
          .../epochs/20240107

        This is suitable for current LiCSAR JASMIN/CEDA listings where epoch
        dates and interferogram pairs are exposed as small manifest entries
        rather than true directory links.
        """
        parent_rel_path = parent_rel_path.strip("/")
        child_name = child_name.strip("/")

        manifest_rel = f"{parent_rel_path}/{child_name}"
        manifest_url = self._list_url(manifest_rel)

        try:
            resp = self._get(manifest_url)
            manifest_files = self._parse_manifest(
                resp.text,
                manifest_url,
                debug_hint=manifest_rel,
            )

            if not self.quiet:
                print(
                    f"    [debug] {manifest_rel}: manifest-style only, "
                    f"{len(manifest_files)} files"
                )

            return manifest_files

        except Exception as e:
            if not self.quiet:
                print(f"    [debug] {manifest_rel}: manifest failed ({e})")
            return []

    # ------------------------------------------------------------------
    # Product matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_product(filename, product_list):
        """Check if a filename matches any selected product type."""
        name = safe_basename_from_url_or_name(filename)

        for p in product_list:
            if p == "metadata":
                if is_metadata_product(name):
                    return True
                continue

            if p == "epochs":
                # If user asks --products epochs, all epoch files are allowed.
                return True

            pred = PRODUCTS.get(p)
            if pred and pred(name):
                return True

        return False

    @staticmethod
    def _wants_product(product_list, *names):
        """Check if any of the named products are in the product list."""
        return any(p in product_list for p in names)

    @staticmethod
    def _date_in_range(date_str, start, end):
        """Check if date_str falls within [start, end]."""
        if start is None:
            return True
        if end is None:
            return date_str == start
        return start <= date_str <= end

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, orbits=None, frames=None, dates=None, products=None):
        """Traverse the LiCSAR hierarchy and build a download plan."""
        tasks = []
        start_date, end_date = parse_date_range(dates)
        product_list = products or ["all"]

        has_metadata = self._wants_product(
            product_list,
            "all",
            "metadata",
            "dem",
            "hgt",
            "los_E",
            "los_N",
            "los_U",
            "inc",
        )
        has_epochs = self._wants_product(product_list, "all", "epochs")
        has_interferograms = self._wants_product(
            product_list,
            "all",
            "unw",
            "cc",
            "diff_pha",
            "png",
        )

        # Resolve frames to process.
        if frames:
            frame_list = frames
        elif orbits:
            frame_list = []
            for orbit in sorted(orbits):
                try:
                    frame_list.extend(self.list_frames(orbit))
                except Exception as e:
                    print(f"Warning: could not list frames for orbit {orbit}: {e}")
                    continue
        else:
            raise ValueError("Must specify --orbits or --frames")

        for frame_name in frame_list:
            frame_name = frame_name.strip()
            if not frame_name:
                continue

            orbit = int(frame_name[:3])
            frame_dir = f"{orbit}/{frame_name}"

            # ---- Metadata files ----
            if has_metadata:
                try:
                    meta_entries = self.list_metadata_files(orbit, frame_name)
                    matched = 0

                    for mf in meta_entries:
                        if self._match_product(mf["name"], product_list):
                            tasks.append(
                                Task(
                                    url=mf.get("url") or self._file_url(frame_dir, "metadata", mf["name"]),
                                    dest_path=self.output_dir / frame_dir / "metadata" / mf["name"],
                                    size_bytes=mf.get("size_bytes", 0),
                                    description=f"{frame_name}/metadata/{mf['name']}",
                                )
                            )
                            matched += 1

                    if not self.quiet:
                        print(
                            f"  {frame_name}/metadata/: {len(meta_entries)} files "
                            f"({matched} matched)"
                        )

                except Exception as e:
                    print(f"Warning: could not list metadata for {frame_name}: {e}")

            # ---- Epochs ----
            if has_epochs:
                try:
                    all_epochs = self.list_epochs(orbit, frame_name)
                    epoch_file_count = 0
                    epoch_entries_in_range = 0

                    for ed in all_epochs:
                        if not self._date_in_range(ed, start_date, end_date):
                            continue

                        epoch_entries_in_range += 1
                        epoch_files = self.list_child_files(f"{frame_dir}/epochs", ed)

                        for ef in epoch_files:
                            name = ef["name"]
                            tasks.append(
                                Task(
                                    url=ef.get("url") or self._file_url(frame_dir, "epochs", ed, name),
                                    dest_path=self.output_dir / frame_dir / "epochs" / ed / name,
                                    size_bytes=ef.get("size_bytes", 0),
                                    description=f"{frame_name}/epochs/{ed}/{name}",
                                )
                            )

                        epoch_file_count += len(epoch_files)

                    if not self.quiet:
                        print(
                            f"  {frame_name}/epochs/: {len(all_epochs)} dates "
                            f"({epoch_entries_in_range} in range, {epoch_file_count} files)"
                        )

                except Exception as e:
                    print(f"Warning: could not list epochs for {frame_name}: {e}")

            # ---- Interferograms ----
            if has_interferograms:
                try:
                    all_pairs = self.list_interferogram_pairs(orbit, frame_name)
                    ifg_file_count = 0
                    pair_entries_in_range = 0

                    for d1, d2 in all_pairs:
                        # Keep original behavior: filter by the first date of the pair.
                        if not self._date_in_range(d1, start_date, end_date):
                            continue

                        pair_entries_in_range += 1
                        pair = f"{d1}_{d2}"
                        ifg_files = self.list_child_files(f"{frame_dir}/interferograms", pair)

                        for ifg_file in ifg_files:
                            name = ifg_file["name"]
                            if self._match_product(name, product_list):
                                tasks.append(
                                    Task(
                                        url=ifg_file.get("url") or self._file_url(frame_dir, "interferograms", pair, name),
                                        dest_path=self.output_dir / frame_dir / "interferograms" / pair / name,
                                        size_bytes=ifg_file.get("size_bytes", 0),
                                        description=f"{frame_name}/ifg/{pair}/{name}",
                                    )
                                )

                        ifg_file_count += len(ifg_files)

                    if not self.quiet:
                        print(
                            f"  {frame_name}/interferograms/: {len(all_pairs)} pairs "
                            f"({pair_entries_in_range} in range, {ifg_file_count} files listed)"
                        )

                except Exception as e:
                    print(f"Warning: could not list interferograms for {frame_name}: {e}")

        return tasks

    # ------------------------------------------------------------------
    # File download
    # ------------------------------------------------------------------

    def download_file(self, url, dest_path, size_bytes=0):
        """Download a single file with streaming, resume, and progress.

        Returns:
            'downloaded', 'skipped', or 'failed'
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        local_size = 0
        if self.resume and dest_path.exists():
            local_size = dest_path.stat().st_size
            if size_bytes > 0 and local_size >= size_bytes:
                if not self.quiet:
                    print(f"  SKIP (complete): {dest_path.name}")
                self.stats["skipped"] += 1
                return "skipped"

        last_error = None
        mode = "wb"

        for attempt in range(self.max_retries + 1):
            try:
                headers = {}
                if self.resume and dest_path.exists():
                    local_size = dest_path.stat().st_size
                    if local_size > 0:
                        headers["Range"] = f"bytes={local_size}-"
                else:
                    local_size = 0

                resp = self.session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                if resp.status_code == 416:
                    if not self.quiet:
                        print(f"  SKIP (complete): {dest_path.name}")
                    self.stats["skipped"] += 1
                    return "skipped"

                resp.raise_for_status()

                if resp.status_code == 206:
                    mode = "ab"
                    content_range = resp.headers.get("Content-Range", "")
                    rm = re.search(r"/(\d+)", content_range)
                    total_size = int(rm.group(1)) if rm else (
                        local_size + int(resp.headers.get("Content-Length", 0))
                    )
                    downloaded = local_size
                else:
                    mode = "wb"
                    total_size = int(resp.headers.get("Content-Length", size_bytes or 0))
                    downloaded = 0

                filename = dest_path.name

                with open(dest_path, mode) as f:
                    if TQDM_AVAILABLE and not self.quiet and total_size > 0:
                        with tqdm(
                            total=total_size,
                            initial=downloaded,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"  {filename[:50]}",
                            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                       "{n_fmt}/{total_fmt} [{elapsed}]",
                        ) as pbar:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        last_report = downloaded
                        for chunk in resp.iter_content(chunk_size=65536):
                            if not chunk:
                                continue

                            f.write(chunk)
                            downloaded += len(chunk)

                            if not self.quiet and total_size > 0:
                                if downloaded - last_report >= 5 * 1024 * 1024:
                                    pct = 100 * downloaded // total_size
                                    mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    print(
                                        f"\r  {filename[:40]}: "
                                        f"{mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                                        end="",
                                        flush=True,
                                    )
                                    last_report = downloaded

                        if not self.quiet and total_size > 0 and downloaded > last_report:
                            print()

                self.stats["downloaded"] += 1
                added_bytes = dest_path.stat().st_size - (local_size if mode == "ab" else 0)
                self.stats["bytes"] += max(0, added_bytes)
                return "downloaded"

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_error = e

                if isinstance(e, requests.HTTPError) and e.response is not None:
                    code = e.response.status_code
                    if code in (401, 403, 404) or code < 500:
                        break

                if attempt < self.max_retries:
                    delay = 2 ** attempt
                    if not self.quiet:
                        print(f"  Retry {attempt + 1} in {delay}s: {e}")
                    time.sleep(delay)

                    # If append mode failed, the partial file could be corrupt.
                    if mode == "ab":
                        try:
                            dest_path.unlink()
                        except OSError:
                            pass

            except Exception as e:
                last_error = e
                break

        if not self.quiet:
            print(f"  FAILED: {dest_path.name}: {last_error}")
            print(f"  URL: {url}")

        self.stats["failed"] += 1
        return "failed"

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def execute(self, tasks):
        """Execute a list of download tasks with rate limiting."""
        total = len(tasks)
        total_bytes = sum(t.size_bytes for t in tasks)

        if not self.quiet:
            print(f"\nDownload plan: {total} files")
            if total_bytes > 0:
                print(f"Estimated size: {total_bytes / (1024 ** 3):.2f} GiB")
            else:
                print("Estimated size: unknown")
            print(f"Output directory: {self.output_dir.resolve()}")
            print("-" * 60)

        if self.dry_run:
            for task in tasks:
                size_str = (
                    f" ({task.size_bytes / 1024**2:.1f} MB)"
                    if task.size_bytes else ""
                )
                print(f"  {task.description}{size_str}")
                print(f"    URL: {task.url}")
                print(f"    ->  {task.dest_path}")
            return

        start_time = time.time()

        for i, task in enumerate(tasks, 1):
            if not self.quiet:
                print(f"[{i}/{total}] {task.description}")

            result = self.download_file(task.url, task.dest_path, task.size_bytes)

            if i < total and result != "skipped":
                time.sleep(self.delay)

        elapsed = time.time() - start_time

        if not self.quiet:
            print("-" * 60)
            print(
                f"Summary: {self.stats['downloaded']} downloaded, "
                f"{self.stats['skipped']} skipped, "
                f"{self.stats['failed']} failed"
            )
            if self.stats["bytes"] > 0:
                gib = self.stats["bytes"] / (1024 ** 3)
                print(f"Total downloaded: {gib:.2f} GiB in {elapsed / 60:.1f} minutes")

    def run(
        self,
        orbits=None,
        frames=None,
        dates=None,
        products=None,
        list_orbits=False,
        list_frames=False,
        list_epochs=False,
    ):
        """Main entry point."""
        if not self._test_connection():
            print("Aborted: cannot reach the server.")
            sys.exit(1)

        if list_orbits:
            try:
                orbit_list = self.list_orbits()
                print(f"Available orbits ({len(orbit_list)}):")
                for o in orbit_list:
                    print(f"  {o}")
            except Exception as e:
                print(f"Error listing orbits: {e}")
                sys.exit(1)
            return

        orbit_set = parse_orbit_list(orbits) if orbits else None
        frame_list = None
        if frames:
            frame_list = [f.strip() for f in frames.split(",") if f.strip()]

        if list_frames:
            if frame_list:
                for fn in frame_list:
                    print(fn)
            elif orbit_set:
                for orbit in sorted(orbit_set):
                    try:
                        f_list = self.list_frames(orbit)
                        print(f"Orbit {orbit} ({len(f_list)} frames):")
                        for f in f_list:
                            print(f"  {f}")
                    except Exception as e:
                        print(f"  (error listing orbit {orbit}: {e})")
            else:
                try:
                    all_orbits = self.list_orbits()
                    for orbit in all_orbits:
                        try:
                            f_list = self.list_frames(orbit)
                            print(f"Orbit {orbit} ({len(f_list)} frames):")
                            for f in f_list:
                                print(f"  {f}")
                        except Exception as e:
                            print(f"  (error listing orbit {orbit}: {e})")
                except Exception as e:
                    print(f"Error: {e}")
                    sys.exit(1)
            return

        if list_epochs:
            if not frame_list:
                print("Error: --list-epochs requires --frames")
                sys.exit(1)

            for fn in frame_list:
                orbit = int(fn[:3])
                try:
                    e_list = self.list_epochs(orbit, fn)
                    print(f"{fn} epochs ({len(e_list)}):")
                    for ed in e_list:
                        print(f"  {ed}")
                except Exception as e:
                    print(f"  (error listing epochs for {fn}: {e})")
            return

        if orbit_set is None and frame_list is None:
            print("Error: Must specify --orbits or --frames. Use --help for usage.")
            sys.exit(1)

        try:
            tasks = self.discover(
                orbits=orbit_set,
                frames=frame_list,
                dates=dates,
                products=products,
            )
        except Exception as e:
            print(f"Error during discovery: {e}")
            sys.exit(1)

        if not tasks:
            print("No files matched the specified filters.")
            return

        self.execute(tasks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="LiCSAR InSAR Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python download.py --list-orbits
  python download.py --orbits 172 --list-frames
  python download.py --frames 172A_05661_131313 --list-epochs
  python download.py --orbits 172 --products unw --dates 20200101-20201231
  python download.py --frames 172A_05661_131313 --products metadata
  python download.py --frames 124A_06996_091406 --dates 20240101-20250101 --products all --proxy http://127.0.0.1:7890 --no-verify-ssl
  python download.py --orbits 1 --products unw --dry-run
        """,
    )

    sel = parser.add_argument_group("Selection filters")
    sel.add_argument(
        "--orbits",
        "-o",
        type=str,
        default=None,
        help="Orbit numbers: comma-separated (70,126,172) or range (1-175)",
    )
    sel.add_argument(
        "--frames",
        "-f",
        type=str,
        default=None,
        help=(
            "Frame IDs: comma-separated "
            "(172A_05661_131313,070A_06876_131313). "
            "Overrides --orbits for download."
        ),
    )
    sel.add_argument(
        "--dates",
        "-d",
        type=str,
        default=None,
        help="Date filter: YYYYMMDD or YYYYMMDD-YYYYMMDD",
    )

    prod = parser.add_argument_group("Product selection")
    prod.add_argument(
        "--products",
        "-p",
        type=str,
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "unw",
            "cc",
            "diff_pha",
            "metadata",
            "dem",
            "hgt",
            "los_E",
            "los_N",
            "los_U",
            "inc",
            "epochs",
            "png",
        ],
        help="Product types to download (default: all). Multiple values: -p unw cc metadata",
    )

    out = parser.add_argument_group("Output and behavior")
    out.add_argument(
        "--output",
        "-O",
        type=str,
        default="./downloads",
        help="Output directory (default: ./downloads)",
    )
    out.add_argument(
        "--proxy",
        type=str,
        default=None,
        metavar="URL",
        help="Proxy URL, e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:7890",
    )
    out.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between HTTP requests in seconds (default: 0.5)",
    )
    out.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum retries per download (default: 3)",
    )
    out.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-download entire files)",
    )
    out.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60)",
    )
    out.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification",
    )

    disc = parser.add_argument_group("Discovery (no download)")
    disc.add_argument(
        "--list-orbits",
        action="store_true",
        help="List available orbits and exit",
    )
    disc.add_argument(
        "--list-frames",
        action="store_true",
        help="List frames for given --orbits and exit",
    )
    disc.add_argument(
        "--list-epochs",
        action="store_true",
        help="List epoch dates for given --frames and exit",
    )
    disc.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    disc.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with LiCSARDownloader(
        output_dir=args.output,
        proxy=args.proxy,
        delay=args.delay,
        max_retries=args.retries,
        resume=not args.no_resume,
        quiet=args.quiet,
        dry_run=args.dry_run,
        timeout=args.timeout,
        verify_ssl=not args.no_verify_ssl,
    ) as dl:
        dl.run(
            orbits=args.orbits,
            frames=args.frames,
            dates=args.dates,
            products=args.products,
            list_orbits=args.list_orbits,
            list_frames=args.list_frames,
            list_epochs=args.list_epochs,
        )


if __name__ == "__main__":
    main()
