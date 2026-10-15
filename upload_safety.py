"""Bounded PDF/ZIP intake for untrusted user uploads.

Limits protect application memory; they are not a malware scanner.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile, BadZipFile

@dataclass(frozen=True)
class IntakeLimits:
    max_pdf_bytes: int = 35_000_000
    max_total_bytes: int = 200_000_000
    max_pdfs: int = 200
    max_ratio: float = 100.0

class IntakeError(ValueError):
    pass

def extract_pdfs(uploads, limits=IntakeLimits()):
    """Return (safe PDF name, bytes) with hard limits before extraction."""
    results, total = [], 0
    for upload in uploads:
        name = upload.name
        blob = upload.getvalue()
        if name.lower().endswith('.pdf'):
            entries = [(PurePosixPath(name.replace('\\', '/')).name, blob)]
        elif name.lower().endswith('.zip'):
            entries = []
            try:
                with ZipFile(BytesIO(blob)) as archive:
                    infos = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith('.pdf')]
                    if len(results) + len(infos) > limits.max_pdfs:
                        raise IntakeError('Too many PDFs in uploaded archives')
                    if sum(i.file_size for i in infos) + total > limits.max_total_bytes:
                        raise IntakeError('Uncompressed PDF size limit exceeded')
                    for info in infos:
                        if info.flag_bits & 1:
                            raise IntakeError('Encrypted ZIP entries are unsupported')
                        if info.file_size > limits.max_pdf_bytes:
                            raise IntakeError('Individual PDF size limit exceeded')
                        if info.file_size / max(1, info.compress_size) > limits.max_ratio:
                            raise IntakeError('Suspicious ZIP compression ratio')
                        path = PurePosixPath(info.filename.replace('\\', '/'))
                        if path.is_absolute() or '..' in path.parts:
                            raise IntakeError('Unsafe ZIP member path')
                        data = archive.read(info)
                        entries.append((path.name, data))
            except (BadZipFile, RuntimeError) as exc:
                raise IntakeError('Invalid or unsupported ZIP') from exc
        else:
            raise IntakeError('Only PDF or ZIP accepted')
        for entry_name, data in entries:
            if len(results) >= limits.max_pdfs:
                raise IntakeError('PDF count limit exceeded')
            if len(data) > limits.max_pdf_bytes or total + len(data) > limits.max_total_bytes:
                raise IntakeError('PDF size limit exceeded')
            if not data.lstrip()[:5] == b'%PDF-':
                raise IntakeError(f'Not a valid PDF header: {entry_name}')
            results.append((entry_name, data))
            total += len(data)
    return results
