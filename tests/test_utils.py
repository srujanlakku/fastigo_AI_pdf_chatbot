from src.utils import compute_upload_fingerprint


class _FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._content = content

    def getvalue(self):
        return self._content

    def tell(self):
        return 0

    def seek(self, pos):
        return None


def test_compute_upload_fingerprint_is_stable():
    files = [_FakeUpload("a.pdf", b"hello"), _FakeUpload("b.pdf", b"world")]
    assert compute_upload_fingerprint(files) == compute_upload_fingerprint(files)
