from mo.api import app


def test_decisions_route_is_plural() -> None:
    paths = {route.path for route in app.routes}
    assert "/v1/decisions" in paths
    assert "/v1/decide" not in paths
