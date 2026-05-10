# Publishing GuardLoop to PyPI

GuardLoop is configured for PyPI Trusted Publishing from GitHub Actions.
This avoids long-lived PyPI API tokens and gives GitHub a visible `pypi`
deployment in the repository sidebar after a successful publish.

## One-Time PyPI Setup

Create or log in to your PyPI account at https://pypi.org. The package name is
`guardloop`.

Because the project does not exist on PyPI yet, add a pending trusted publisher:

- Go to your PyPI account sidebar and open **Publishing**.
- Add a new GitHub pending publisher.
- Use owner `awesome-pro`.
- Use repository `guardloop`.
- Use workflow filename `publish-pypi.yml`.
- Leave environment blank so PyPI shows `Environment name: (Any)`.
- Use project name `guardloop`.

Pending publishers do not reserve the package name until the first successful
publish, so run the first publish soon after creating it.

The GitHub workflow still uses a GitHub environment named `pypi` so the
repository sidebar shows a clean `pypi` deployment after publishing. A PyPI
publisher configured with `Environment name: (Any)` accepts that workflow.

## First Publish

After the pending publisher exists on PyPI, run the GitHub workflow manually:

```bash
gh workflow run publish-pypi.yml --repo awesome-pro/guardloop --ref main
```

Then watch it:

```bash
gh run list --repo awesome-pro/guardloop --workflow publish-pypi.yml --limit 1
```

When the run succeeds, PyPI should show:

```text
https://pypi.org/project/guardloop/
```

The package can then be installed with:

```bash
pip install guardloop
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
