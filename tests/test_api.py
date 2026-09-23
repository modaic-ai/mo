from mo.api import DecisionRequest, app


def test_decisions_route_is_plural() -> None:
    paths = {route.path for route in app.routes}
    assert "/v1/decisions" in paths
    assert "/v1/decide" not in paths


def test_decision_request_accepts_image_data_urls() -> None:
    request = DecisionRequest(
        state={},
        question="Choose.",
        options={"A": "red", "B": "blue"},
        images=["data:image/png;base64,iVBORw0KGgo="],
    )
    assert request.images == ["data:image/png;base64,iVBORw0KGgo="]
