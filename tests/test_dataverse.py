"""Tests for the Dataverse download helper (file-list parsing; no network)."""

import unittest
from unittest import mock

from eroll import dataverse


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestDataverse(unittest.TestCase):
    def test_list_files_parses_native_and_ingested(self):
        payload = {
            "data": [
                # Non-ingested file (e.g. .csv.gz): use stored filename/filesize.
                {
                    "dataFile": {
                        "id": 1,
                        "filename": "gujarat_all.csv.gz",
                        "filesize": 100,
                    }
                },
                # Ingested tabular file: prefer the original name/size.
                {
                    "dataFile": {
                        "id": 2,
                        "filename": "daman.tab",
                        "filesize": 5,
                        "originalFileName": "daman.csv",
                        "originalFileSize": 200,
                    }
                },
            ]
        }
        with mock.patch.object(
            dataverse.requests, "get", return_value=_Resp(payload)
        ) as g:
            files = dataverse.list_files("doi:10.7910/DVN/XXXX", token="t")
        # Token is sent as the X-Dataverse-key header.
        self.assertEqual(g.call_args.kwargs["headers"], {"X-Dataverse-key": "t"})
        self.assertEqual(
            files,
            [
                {"id": 1, "filename": "gujarat_all.csv.gz", "filesize": 100},
                {"id": 2, "filename": "daman.csv", "filesize": 200},
            ],
        )

    def test_no_token_sends_no_auth_header(self):
        with mock.patch.object(
            dataverse.requests, "get", return_value=_Resp({"data": []})
        ) as g:
            dataverse.list_files("doi:10.7910/DVN/XXXX")
        self.assertEqual(g.call_args.kwargs["headers"], {})


if __name__ == "__main__":
    unittest.main()
