"""Shared pytest fixtures for all test modules."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_cross_encoder() -> pytest.FixtureRequest:
    """Prevent CrossEncoder from downloading models during tests.

    All tests that need controlled reranker behaviour mock it locally.
    This fixture is a safety net that stops network calls in every other test.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8]
    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        yield
