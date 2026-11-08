"""A residual correlation that cannot be estimated is reported as unavailable, never 0.00."""

from quantred.analytics.reporting import _verdict_information_lines


def test_unavailable_residual_correlation_is_printed_as_unavailable():
    """Rendering must not turn a missing rho into 0.00 and a zero-deflation breadth."""
    text = "\n".join(
        _verdict_information_lines(
            {
                "information": {
                    "periods": 20,
                    "rho_bar": None,
                    "breadth_effective": None,
                    "predicted_ir": None,
                    "n_names": 5,
                    "realized_ir": 0.3,
                    "ir_standard_error": 0.5,
                }
            }
        )
    )
    assert "breadth unavailable" in text
    assert "IR predicted unavailable" in text
