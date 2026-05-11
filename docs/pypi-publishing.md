# Publishing GuardLoop to PyPI

GuardLoop is published on PyPI at https://pypi.org/project/guardloop/.
Publishing uses PyPI Trusted Publishing from GitHub Actions. This avoids
long-lived PyPI API tokens and gives GitHub a visible `pypi` deployment in the
repository sidebar after a successful publish.

## Trusted Publisher Setup

The trusted publisher configuration should stay aligned with:

- PyPI project name: `guardloop`
- GitHub owner: `awesome-pro`
- GitHub repository: `guardloop`
- Workflow filename: `publish-pypi.yml`
- PyPI environment name: blank / `(Any)`

The GitHub workflow still uses a GitHub environment named `pypi` so the
repository sidebar shows a clean `pypi` deployment after publishing. A PyPI
publisher configured with `Environment name: (Any)` accepts that workflow.

## Manual Publish

PyPI versions are immutable. Before publishing a new release, bump `version` in
`pyproject.toml`, build and test locally, then run:

```bash
gh workflow run publish-pypi.yml --repo awesome-pro/guardloop --ref main
```

Then watch it:

```bash
gh run list --repo awesome-pro/guardloop --workflow publish-pypi.yml --limit 1
```

When the run succeeds, PyPI should show the new version at:

```text
https://pypi.org/project/guardloop/
```

The package can then be installed with:

```bash
pip install guardloop                    # core
pip install "guardloop[langgraph]"       # + the LangGraph adapter
pip install "guardloop[openai-agents]"   # + the OpenAI Agents SDK adapter
```

Optional extras (`otel`, `langgraph`, `openai-agents`) ship in the same wheel —
`uv build` packages all of `src/guardloop/**`, including `guardloop/adapters/`.
After a build you can confirm with `unzip -l dist/guardloop-*.whl | grep adapters`
(it lists `guardloop/adapters/__init__.py`, `.../langgraph.py`, `.../openai_agents.py`).

## Future Releases

For future versions:

1. Bump `version` in `pyproject.toml`.
2. Build and test locally.
3. Commit and tag the release.
4. Create a GitHub release.
5. Run the publish workflow if the release event did not already trigger it.

The `Publish to PyPI` workflow also runs automatically when a GitHub release is
published, so normal releases will publish to PyPI without a manual workflow
dispatch.
