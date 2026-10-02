import io
import pytest
from typing import List


def make_test_pdf(pages_text: List[str]) -> bytes:
    """Generate minimal valid PDF bytes containing the specified text on each page."""
    objects = []
    num_pages = len(pages_text)
    page_obj_ids = [3 + i * 3 for i in range(num_pages)]

    # Obj 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Obj 2: Pages
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n")

    for i, text in enumerate(pages_text):
        pid = 3 + i * 3
        fid = pid + 1
        cid = pid + 2
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines = safe_text.split("\n")
        stream_ops = ["BT", "/F1 12 Tf", "72 712 Td"]
        for line in lines:
            stream_ops.append(f"({line}) Tj")
            stream_ops.append("0 -16 Td")
        stream_ops.append("ET")
        stream_bytes = "\n".join(stream_ops).encode("latin1")

        page_obj = (
            f"{pid} 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {fid} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {cid} 0 R >>\nendobj\n"
        )
        font_obj = f"{fid} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        content_obj = f"{cid} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n" + stream_bytes.decode("latin1") + "\nendstream\nendobj\n"
        objects.extend([page_obj, font_obj, content_obj])

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj.encode("latin1"))
    startxref = out.tell()
    out.write(f"xref\n0 {len(offsets)}\n".encode("latin1"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("latin1"))
    out.write(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF".encode("latin1"))
    return out.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Fixture returning sample HR handbook PDF bytes."""
    return make_test_pdf([
        "1. Welcome to Nexain\nWelcome to our company! This guide introduces our core workplace values.",
        "2. Working Hours and Leave Policy\nStandard office hours are 09:00 to 18:00.\nEmployees receive 12 days annual leave.",
    ])
