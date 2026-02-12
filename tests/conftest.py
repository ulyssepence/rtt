import pathlib
import dotenv
import pytest

dotenv.load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def sample_video(data_dir):
    path = data_dir / "KnifeThr1950_512kb.mp4"
    if not path.exists():
        pytest.skip("data/sample/KnifeThr1950_512kb.mp4 not found")
    return path
