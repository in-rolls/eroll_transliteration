"""Download files from a Harvard Dataverse dataset (Native + data-access API).

Companion to the *upload* tooling in ``../soundscape`` (which has no downloader). Used to
fetch raw electoral-roll files into the data dir before transliteration, e.g. the Gujarat
/ Daman / Dadra rolls::

    export DATAVERSE_API_TOKEN=...
    python -m eroll.dataverse list --doi doi:10.7910/DVN/XXXXXX
    python -m eroll.dataverse download --doi doi:10.7910/DVN/XXXXXX --pattern '*gujarat*'

The token (``--token`` or ``DATAVERSE_API_TOKEN``) is only needed for restricted files;
public files download without it. Files are fetched in their *original* form (Dataverse
ingests tabular uploads to ``.tab`` otherwise).
"""

import fnmatch
import os
from pathlib import Path

import click
import requests
from tqdm import tqdm

from .states import data_dir

DEFAULT_SERVER = "https://dataverse.harvard.edu"


def _headers(token: str | None) -> dict[str, str]:
    return {"X-Dataverse-key": token} if token else {}


def list_files(
    doi: str, *, server: str = DEFAULT_SERVER, token: str | None = None
) -> list[dict]:
    """Return ``[{id, filename, filesize}]`` for the dataset's latest version."""
    resp = requests.get(
        f"{server}/api/datasets/:persistentId/versions/:latest/files",
        params={"persistentId": doi},
        headers=_headers(token),
        timeout=60,
    )
    resp.raise_for_status()
    files: list[dict] = []
    for item in resp.json()["data"]:
        df = item["dataFile"]
        # originalFileName/Size are present only for ingested tabular files; fall back
        # to the stored filename for everything else (.csv.gz, .7z, ...).
        files.append(
            {
                "id": df["id"],
                "filename": df.get("originalFileName") or df["filename"],
                "filesize": df.get("originalFileSize") or df.get("filesize"),
            }
        )
    return files


def download_file(
    file_id: int, dest: Path, *, server: str = DEFAULT_SERVER, token: str | None = None
) -> Path:
    """Stream one datafile (original form) to ``dest`` atomically, with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(
        f"{server}/api/access/datafile/{file_id}",
        params={"format": "original"},
        headers=_headers(token),
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(tmp, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.group()
def cli() -> None:
    """Download files from a Harvard Dataverse dataset."""


def _token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("DATAVERSE_API_TOKEN")


@cli.command("list")
@click.option(
    "--doi", required=True, help="Dataset persistentId, e.g. doi:10.7910/DVN/XXXXXX"
)
@click.option("--server", default=DEFAULT_SERVER, show_default=True)
@click.option(
    "--token", default=None, help="Dataverse API token (or DATAVERSE_API_TOKEN)."
)
def list_cmd(doi: str, server: str, token: str | None) -> None:
    """List files in a dataset's latest version."""
    files = list_files(doi, server=server, token=_token(token))
    click.echo(f"{len(files)} file(s) in {doi}:")
    for f in files:
        size = f["filesize"] or 0
        click.echo(f"  [{f['id']:>10}] {f['filename']:<50} {size / 1e6:>10.1f} MB")


@cli.command()
@click.option(
    "--doi", required=True, help="Dataset persistentId, e.g. doi:10.7910/DVN/XXXXXX"
)
@click.option("--pattern", default="*", show_default=True, help="Glob over filenames.")
@click.option("--out", default=None, help="Output dir (default: the eroll data dir).")
@click.option("--server", default=DEFAULT_SERVER, show_default=True)
@click.option(
    "--token", default=None, help="Dataverse API token (or DATAVERSE_API_TOKEN)."
)
@click.option("--overwrite", is_flag=True, help="Re-download files that already exist.")
def download(
    doi: str,
    pattern: str,
    out: str | None,
    server: str,
    token: str | None,
    overwrite: bool,
) -> None:
    """Download files matching ``--pattern`` from a dataset to the data dir."""
    token = _token(token)
    out_dir = Path(out) if out else data_dir()
    files = list_files(doi, server=server, token=token)
    chosen = [f for f in files if fnmatch.fnmatch(f["filename"], pattern)]
    if not chosen:
        click.echo(f"No files in {doi} match {pattern!r} (of {len(files)} total).")
        return
    click.echo(f"Downloading {len(chosen)} file(s) -> {out_dir}")
    for f in chosen:
        dest = out_dir / f["filename"]
        if dest.exists() and not overwrite:
            click.echo(f"  skip (exists): {dest.name}")
            continue
        download_file(f["id"], dest, server=server, token=token)
    click.echo("done.")


if __name__ == "__main__":
    cli()
