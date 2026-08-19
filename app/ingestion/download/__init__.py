"""Raw PDF acquisition: explicit ingestion, streaming download, and durable
local storage of the original PDF artifact.

Boundary: ``PdfAcquisitionService -> PdfDownloadClient -> arXiv`` for the
network side, ``PdfAcquisitionService -> PaperStorage -> filesystem`` for
persistence. Stops at a validated, checksummed, stored PDF -- no parsing,
no chunking, no text extraction happens here.
"""
