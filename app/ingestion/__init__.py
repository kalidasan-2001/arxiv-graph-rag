"""Ingestion pipeline: discovery, acquisition, and processing of papers.

Kept separate from query-time reasoning (CLAUDE.md #11). Only the
`discovery` stage exists so far -- it stops at metadata, never touches a
PDF, and never triggers ingestion job creation.
"""
