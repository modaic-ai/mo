.PHONY: check serve smoke

check:
	uv run --extra dev ruff check .
	uv run --extra dev pytest -q

serve:
	docker compose up --build

smoke:
	uv run mo-smoke --base-url http://localhost:8080
