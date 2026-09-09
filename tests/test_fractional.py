import pytest

from actuarialmath import LifeTable


@pytest.fixture
def life():
    return LifeTable(udd=True).set_table(q={45: 0.5, 46: 0.6, 47: 1.0})


@pytest.mark.parametrize("r,t", [(0.25, 0.5), (0.5, 0.5), (0.75, 0.25)])
def test_udd_fractional_density_integrates_to_mortality(life, r, t):
    assert t * life.f_r(45, r=r) == pytest.approx(life.q_r(45, r=r, t=t))


@pytest.mark.parametrize("r", [0.25, 0.5, 0.75])
def test_udd_fractional_density_at_zero_equals_force(life, r):
    assert life.f_r(45, r=r) == pytest.approx(life.mu_r(45, r=r))
