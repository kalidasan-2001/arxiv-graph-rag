"""arXiv discovery: search arXiv and normalize results into domain `Paper`s.

Boundary: ``PaperDiscoveryService -> ArxivClient -> arXiv``. Nothing
outside this package should depend on `ArxivPaperResult` or on the arXiv
Atom response shape -- only the normalized domain `Paper` /
`PaperDiscoveryResult` cross this package's boundary.
"""
