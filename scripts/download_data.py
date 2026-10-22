"""Download the course's Abalone CSV and verify its exact content."""

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.kaggle.com/api/v1/datasets/download/rodolfomendes/abalone-dataset"
SHA256 = "6fda3c4b0c0cff9bd5f7a46434509179069ed620c8fc7090d872a08d2c1a729a"


def download(output: Path = Path("data/abalone.csv")) -> Path:
    """Cache only the expected CSV, failing clearly if the source changes."""
    if output.exists():
        data = output.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as archive:
            data = archive.read("abalone.csv")
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Dataset checksum differs from the course snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output


if __name__ == "__main__":
    print(download())
