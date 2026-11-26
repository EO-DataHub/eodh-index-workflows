"""The input item: staged by ADES (`--stac_item_dir`), or fetched by the workflow from `--stac_item=<url>`."""

import http.client
import json
import logging
import math
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

from eodh_workflow import StacItemInput, WorkflowInputs

logger = logging.getLogger(__name__)

ITEM_URL_PARAM = "stac_item"
FETCH_TIMEOUT = 30.0
FETCH_RETRIES = 4
BACKOFF_SECONDS = 1.0
MAX_RETRY_AFTER = 30.0
RETRY_STATUSES = frozenset({429}) | frozenset(range(500, 600))

INPUTS_HELP = (
    "This workflow takes one STAC item: either stac_item, the URL of the item (http(s):// or file://), "
    "or stac_item_dir, a directory written by ADES stage-in"
)


def item_input(inputs: WorkflowInputs) -> StacItemInput:
    """The input item, from the staged directory or fetched from the `stac_item` URL. Exactly one must be given."""
    url = (inputs.params.get(ITEM_URL_PARAM) or "").strip()
    if inputs.stac_item is not None and url:
        raise ValueError(f"{INPUTS_HELP}. Both were given.")
    if inputs.stac_item is not None:
        return inputs.stac_item
    if url:
        return fetch_item(url)
    raise ValueError(f"{INPUTS_HELP}. Neither was given.")


def fetch_item(url: str, timeout: float = FETCH_TIMEOUT) -> StacItemInput:
    """Read the STAC item at `url` (http(s):// or file://), with its relative asset hrefs made absolute."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        if parsed.netloc.lower() not in ("", "localhost"):
            raise ValueError(f"Cannot fetch STAC item {url!r}: a file:// URL must have no host or 'localhost'")
        item_path = Path(unquote(parsed.path))
        item = json.loads(item_path.read_text())
    elif scheme in ("http", "https"):
        body = _get(url, timeout)
        try:
            item = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(f"STAC item {url} is not JSON: {error}") from error
        item_path = Path(Path(unquote(parsed.path)).name or "item.json")
    else:
        raise ValueError(f"Cannot fetch STAC item {url!r}: the URL must be http://, https:// or file://")

    for asset in item.get("assets", {}).values():
        href = asset.get("href")
        if href and not urlparse(href).scheme:
            asset["href"] = urljoin(url, href)
    logger.info(f"Fetched item {item.get('id')!r} from {url}, assets {sorted(item.get('assets', {}))}")
    return StacItemInput(item=item, item_path=item_path)


def _retry_after(error: urllib.error.HTTPError) -> Optional[float]:
    """The Retry-After header of `error` in seconds, capped at MAX_RETRY_AFTER; None if absent or not finite seconds."""
    value = (error.headers.get("Retry-After") if error.headers else None) or ""
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds == math.inf:
        return MAX_RETRY_AFTER
    if not math.isfinite(seconds):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


def _cause(error: BaseException) -> object:
    """The underlying error: the reason urllib wrapped in a URLError, or the error itself."""
    if isinstance(error, urllib.error.URLError) and not isinstance(error, urllib.error.HTTPError):
        return error.reason
    return error


def _is_timeout(error: BaseException) -> bool:
    return isinstance(_cause(error), (socket.timeout, TimeoutError))


def _get(url: str, timeout: float) -> bytes:
    """The body at `url`. Retries timeouts, connection errors, 429 and 5xx; honours Retry-After in seconds."""
    problem = ""
    wait: Optional[float] = None
    last_error: Optional[BaseException] = None
    for attempt in range(FETCH_RETRIES + 1):
        if attempt:
            delay = wait if wait is not None else BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(f"Fetching {url} gave {problem}; retrying in {delay:g}s")
            time.sleep(delay)
        wait = None
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUSES:
                raise RuntimeError(f"Could not fetch STAC item {url}: HTTP {error.code}") from error
            problem, wait, last_error = f"HTTP {error.code}", _retry_after(error), error
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            if _is_timeout(error):
                problem, last_error = f"a timeout after {timeout:g}s", error
            elif isinstance(_cause(error), (http.client.IncompleteRead, ConnectionError)):
                problem, last_error = f"a connection error ({_cause(error)!r})", error
            else:
                raise RuntimeError(f"Could not fetch STAC item {url}: {_cause(error)}") from error
    attempts = FETCH_RETRIES + 1
    if last_error is not None and _is_timeout(last_error):
        message = f"Fetching STAC item {url} timed out after {timeout:g}s, {attempts} attempts"
        raise TimeoutError(message) from last_error
    raise RuntimeError(f"Could not fetch STAC item {url}: {problem} after {attempts} attempts") from last_error
