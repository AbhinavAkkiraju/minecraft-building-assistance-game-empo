"""
Human-power / empowerment objective for MBAG's AlphaZero assistant.

Implements the objective of Heitzig & Potham, "Model-Based Soft Maximization
of Suitable Metrics of Long-Term Human Power" (arXiv:2508.00159), as an
alternative to the assistance-games/CIRL objective in
``mbag.rllib.alpha_zero``. Selected via the ``objective="power"`` config flag;
the CIRL code path is left untouched so both objectives remain runnable and
comparable.
"""
