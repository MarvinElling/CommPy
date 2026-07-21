"""Tests for top-level commpy package metadata."""

import re

import commpy


def test_version_is_a_valid_semver_string():
    assert re.match(r'^\d+\.\d+\.\d+', commpy.__version__)
