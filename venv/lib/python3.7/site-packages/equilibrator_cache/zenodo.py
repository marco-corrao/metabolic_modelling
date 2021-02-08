"""Handles downloading an caching of files from Zenodo."""
# The MIT License (MIT)
#
# Copyright (c) 2013 The Weizmann Institute of Science.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
# Copyright (c) 2018 Institute for Molecular Systems Biology,
# ETH Zurich, Switzerland.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import asyncio
import hashlib
import logging
import pathlib
import warnings
from io import BytesIO
from typing import Dict, Optional

import appdirs
import httpx
import pyzenodo3
from tenacity import retry, stop_after_attempt
from tqdm.asyncio import tqdm

from . import DEFAULT_COMPOUND_CACHE_FNAME


logger = logging.getLogger(__name__)


zen = pyzenodo3.Zenodo()


async def download_from_url(url: str) -> BytesIO:
    """Download a file from a given URL using httpx.

    Parameters
    ----------
    url : str
        The URL address of the file.
    md5 : str, optional
        The MD5 checksum of the file, if given and the checksum doesn't match
        the downaloded file, an IOError is raised. The default is None.

    Returns
    -------
    BytesIO
        Containing the downloaded file.

    """
    data = BytesIO()
    client = httpx.AsyncClient()
    async with client.stream("GET", url) as response:
        total = int(response.headers["Content-Length"])
        md5 = response.headers["content-md5"]

        num_bytes = 0
        with tqdm(
            total=total, unit_scale=True, unit_divisor=1024, unit="B"
        ) as progress:
            async for chunk in response.aiter_bytes():
                data.write(chunk)
                progress.update(len(chunk))
                num_bytes += len(chunk)
        await client.aclose()

    if num_bytes < total:
        raise IOError(f"Failed to download file from {url}")

    data.seek(0)
    if hashlib.md5(data.read()).hexdigest() != md5:
        raise IOError(f"MD5 mismatch while trying to download file from {url}")

    data.seek(0)
    return data


async def _get_zenodo_files(zenodo_doi: str) -> Dict[str, BytesIO]:
    """Run the get_zenodo_files coroutine asynchronously."""
    rec = zen.find_record_by_doi(zenodo_doi)
    fnames = [d["key"] for d in rec.data["files"]]
    urls = [d["links"]["self"] for d in rec.data["files"]]
    tasks = [download_from_url(url) for url in urls]
    data_streams = await asyncio.gather(*tasks)
    return dict(zip(fnames, data_streams))


@retry(stop=stop_after_attempt(3))
def get_zenodo_files(zenodo_doi: str) -> Dict[str, BytesIO]:
    """Download all the files stored in Zenodo (under a specific DOI).

    Parameters
    ----------
    zenodo_doi : str
        the DOI of the Zenodo entry.

    Returns
    -------
    Dict
        the dictionary with file names as keys, and the file contents as
        values.

    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_get_zenodo_files(zenodo_doi))


def get_zenodo_checksum(zenodo_doi: str, zenodo_fname: str) -> Optional[str]:
    """Download all the files stored in Zenodo (under a specific DOI).

    Parameters
    ----------
    zenodo_doi : str
        the DOI of the Zenodo entry.

    Returns
    -------
    str
        latest version of the Zenodo entry.

    """
    try:
        rec = zen.find_record_by_doi(zenodo_doi)
    except pyzenodo3.base.requests.exceptions.ConnectionError:
        warnings.warn("No connection to Zenodo, cannot verify local version.")
        return None

    for d in rec.data["files"]:
        if d["key"] == zenodo_fname:
            fmt, checksum = d["checksum"].split(":", 1)
            assert fmt == "md5", "Checksum format must be MD5"
            return checksum

    raise KeyError(
        "The file {zenodo_fname} was not found in the Zenodo entry: "
        f"{zenodo_doi}"
    )


def get_cached_filepath(zenodo_doi: str, zenodo_fname: str) -> pathlib.Path:
    """Get data from a file stored in Zenodo (or from cache, if available).

    Parameters
    ----------
    zenodo_doi : str
        the DOI of the Zenodo entry.
    zenodo_fname : str
        the specific filename to fetch from Zenodo.

    Returns
    -------
    str
        the path to the locally cached file.

    """

    cache_directory = pathlib.Path(
        appdirs.user_cache_dir(appname="equilibrator")
    )
    cache_directory.mkdir(parents=True, exist_ok=True)

    cache_fname = cache_directory / DEFAULT_COMPOUND_CACHE_FNAME

    if cache_fname.exists():
        # make sure that it is in the correction version and not corrupted.

        logging.info("Fetching metadata about the Compound Cache from Zenodo")
        md5 = get_zenodo_checksum(zenodo_doi, zenodo_fname)
        if md5 is None:
            # we cannot perform the checksum test, so we assume that everything
            # is okay.
            return cache_fname

        # verify that the checksum from Zenodo matches the cached file.
        logging.info("Validate the cached copy using MD5 checksum")
        with cache_fname.open("rb") as fp:
            if md5 == hashlib.md5(fp.read()).hexdigest():
                return cache_fname

        # if the checksum is not okay, it mean the file is corrupted or
        # exists in an older version. therefore, we ignore it an override
        # it with a newly downloaded version

    logging.info("Fetching a new version of the Compound Cache from Zenodo")
    dataframe_dict = get_zenodo_files(zenodo_doi)
    cache_fname.write_bytes(dataframe_dict[zenodo_fname].getbuffer())

    return cache_fname
