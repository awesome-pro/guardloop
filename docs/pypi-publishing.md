# Publishing AgentRuntime to PyPI

AgentRuntime is configured for PyPI Trusted Publishing from GitHub Actions.
This avoids long-lived PyPI API tokens and gives GitHub a visible `pypi`
deployment in the repository sidebar after a successful publish.

## One-Time PyPI Setup

Create or log in to your PyPI account at https://pypi.org. The package name is
`agentruntime`.

Because the project does not exist on PyPI yet, add a pending trusted publisher:

- Go to your PyPI account sidebar and open **Publishing**.
- Add a new GitHub pending publisher.
- Use owner `awesome-pro`.
- Use repository `agent-runtime`.
- Use workflow filename `publish-pypi.yml`.
- Use environment `pypi`.
- Use project name `agentruntime`.

Pending publishers do not reserve the package name until the first successful
publish, so run the first publish soon after creating it.

## First Publish

After the pending publisher exists on PyPI, run the GitHub workflow manually:

```bash
gh workflow run publish-pypi.yml --repo awesome-pro/agent-runtime --ref main
```

Then watch it:

```bash
gh run list --repo awesome-pro/agent-runtime --workflow publish-pypi.yml --limit 1
```

When the run succeeds, PyPI should show:

```text
https://pypi.org/project/agentruntime/
```

The package can then be installed with:

```bash
pip install agentruntime
```

## Future Releases

For future versions:

1. Bump `version` in `pyproject.toml`.
2. Build and test locally.
3. Commit and tag the release.
4. Create a GitHub release.

The `Publish to PyPI` workflow also runs automatically when a GitHub release is
published, so normal releases will publish to PyPI without a manual workflow
dispatch.
