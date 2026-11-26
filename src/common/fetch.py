"""The input item: staged by ADES (`--stac_item_dir`), or fetched by the workflow from `--stac_item=<url>`."""

import json
import logging
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
    scheme = urlparse(url).scheme.lower()
    if scheme == "file":
        item_path = Path(unquote(urlparse(url).path))
        item = json.loads(item_path.read_text())
    elif scheme in ("http", "https"):
        item = json.loads(_get(url, timeout))
        item_path = Path(Path(unquote(urlparse(url).path)).name or "item.json")
    else:
        raise ValueError(f"Cannot fetch STAC item {url!r}: the URL must be http://, https:// or file://")

    for asset in item.get("assets", {}).values():
        href = asset.get("href")
        if href and not urlparse(href).scheme:
            asset["href"] = urljoin(url, href)
    logger.info(f"Fetched item {item.get('id')!r} from {url}, assets {sorted(item.get('assets', {}))}")
    return StacItemInput(item=item, item_path=item_path)


def _get(url: str, timeout: float) -> bytes:
    """The body at `url`. Retries 429 and 5xx with exponential backoff; a timeout is not retried."""
    status: Optional[int] = None
    for attempt in range(FETCH_RETRIES + 1):
        if attempt:
            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(f"Fetching {url} gave HTTP {status}; retrying in {delay:g}s")
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            if status not in RETRY_STATUSES:
                raise RuntimeError(f"Could not fetch STAC item {url}: HTTP {status}") from error
        except (socket.timeout, TimeoutError) as error:
            raise TimeoutError(f"Fetching STAC item {url} timed out after {timeout:g}s") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise TimeoutError(f"Fetching STAC item {url} timed out after {timeout:g}s") from error
            raise RuntimeError(f"Could not fetch STAC item {url}: {error.reason}") from error
    raise RuntimeError(f"Could not fetch STAC item {url}: HTTP {status} after {FETCH_RETRIES + 1} attempts")
