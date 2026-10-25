"""Pure-endowment variance against an independent Bernoulli reference."""
from decimal import Decimal, localcontext

import pytest

from actuarialmath import Uniform


@pytest.mark.parametrize("remaining", [20, 60, 90])
@pytest.mark.parametrize("rate", ["0", "0.03", "0.1"])
@pytest.mark.parametrize("term_kind", ["zero", "one", "half", "whole"])
@pytest.mark.parametrize("moment", [Uniform.VARIANCE, 1, 2, 3])
def test_uniform_pure_endowment_moment(remaining, rate, term_kind, moment):
    terms = {"zero": 0, "one": 1, "half": remaining // 2,
             "whole": remaining}
    term = terms[term_kind]
    # s=0 isolates this selector from the separately reported conditioning
    # age issue. A unit pure endowment is v^term times a Bernoulli variable.
    with localcontext() as ctx:
        ctx.prec = 60
        survival = Decimal(remaining - term) / Decimal(remaining)
        discount = (Decimal(1) + Decimal(rate)) ** (-term)
        if moment == Uniform.VARIANCE:
            expected = discount ** 2 * survival * (1 - survival)
        else:
            expected = discount ** moment * survival
    life = Uniform(omega=100).set_interest(i=float(rate))
    actual = life.E_x(x=100 - remaining, s=0, t=term, moment=moment)
    assert actual == pytest.approx(float(expected), rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("moment", [0, -99])
def test_uniform_invalid_moment_still_rejected(moment):
    life = Uniform(omega=100).set_interest(i=0.03)
    with pytest.raises(AssertionError):
        life.E_x(x=40, t=10, moment=moment)
