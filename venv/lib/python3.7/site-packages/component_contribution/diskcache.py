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
import hashlib
import pathlib
from io import BytesIO
from typing import Dict, Optional

import appdirs
import asyncio
import diskcache
import httpx
import pyzenodo3
from packaging import version
from tenacity import retry, stop_after_attempt
from tqdm.asyncio import tqdm


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


@retry(stop=stop_after_attempt(3))
def get_zenodo_version(zenodo_doi: str) -> Optional[str]:
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
        return None

    return rec.data["metadata"]["version"]


def get_cached_file(zenodo_doi: str, fname: str) -> BytesIO:
    """Get data from a file stored in Zenodo (or from cache, if available).

    Parameters
    ----------
    zenodo_doi : str
        the DOI of the Zenodo entry.
    fname : str
        the filename to fetch.

    Returns
    -------
    DataFrame
        the data contained in the file.

    """
    cache_directory = pathlib.Path(
        appdirs.user_cache_dir(appname="equilibrator")
    )
    online_version = get_zenodo_version(zenodo_doi)

    with diskcache.Cache(cache_directory) as cache:
        cached_version = cache.get(zenodo_doi + "/VERSION", None)

        if online_version is None:
            if cached_version is None:
                raise KeyError(
                    "Zenodo is unreachable and there is no local cached "
                    f"version for {zenodo_doi}."
                )
            # We have no connection to Zeonodo, so we use whatever version
            # that is cached.
            return cache[zenodo_doi][fname]

        if cached_version is not None:
            if version.parse(online_version) == version.parse(cached_version):
                return cache[zenodo_doi][fname]
            if version.parse(online_version) < version.parse(cached_version):
                raise KeyError(
                    f"The cached version of {zenodo_doi} ({cached_version}) "
                    f"is newer than the one online ({online_version})"
                )

        # If we are here, it means that there is no cached version or that
        # it is outdated. So, we download the files and update the version.
        dataframe_dict = get_zenodo_files(zenodo_doi)
        cache[zenodo_doi] = dataframe_dict
        cache[zenodo_doi + "/VERSION"] = online_version
        return cache[zenodo_doi][fname]
