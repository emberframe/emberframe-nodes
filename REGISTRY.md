# Comfy Registry Publishing

This repository is prepared for Comfy Registry publishing, but publishing requires a registry publisher account and an API key.

## One-Time Setup

1. Go to `https://registry.comfy.org`.
2. Create or use the publisher id `emberframe`.
3. Create a Registry publishing API key for that publisher.
4. In GitHub, open this repository.
5. Go to `Settings > Secrets and variables > Actions > New repository secret`.
6. Add a secret named exactly:

```text
REGISTRY_ACCESS_TOKEN
```

7. Paste the Registry API key as the secret value.

Do not paste the API key into README files, workflow files, issues, commits, or chat logs.

## Publishing

After the secret exists, run the GitHub Action named `Publish to Comfy Registry` manually from the Actions tab.

For future releases:

1. Update `version` in `pyproject.toml`.
2. Commit the change.
3. Create a matching Git tag and GitHub release.
4. Run the registry publishing action.

The workflow intentionally uses `workflow_dispatch` only, so it will not publish automatically on every push.
