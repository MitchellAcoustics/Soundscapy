"""
Soundscapy Databases Module.

This module handles connections to and operations on soundscape databases,
primarily focused on the International Soundscape Database (ISD) and the
Soundscape Attributes Translation Project (SATP).
"""

from soundscapy.databases import isd, satp, satp_testing

__all__ = ["isd", "satp", "satp_testing"]
