# Publishing Guide for Arc

This guide covers the manual process for building and publishing Arc to PyPI (Python Package Index).

**Important**: The package is published as `arc-ml` (to avoid naming conflicts), but users still run the `arc` command after installation.

## Prerequisites

Before publishing, ensure you have:

1. **TestPyPI Account** (for testing)
   - Create account at: https://test.pypi.org/account/register/
   - Verify your email address

2. **PyPI Account** (for production releases)
   - Create account at: https://pypi.org/account/register/
   - Verify your email address

3. **API Tokens**
   - Generate TestPyPI token: https://test.pypi.org/manage/account/token/
   - Generate PyPI token: https://pypi.org/manage/account/token/
   - **Recommended**: Use project-scoped tokens (more secure)
   - Store tokens securely - they won't be shown again!

## Environment Setup

Store your API tokens as environment variables:

```bash
# For TestPyPI
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmc..."  # TestPyPI token

# For production PyPI (when ready)
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmc..."  # PyPI token
```

**Tip**: Add these to your `~/.bashrc` or `~/.zshrc` for persistence, or use a `.env` file (never commit it!).

## Publishing Workflow

### Step 1: Update Version

Update the version in `pyproject.toml`:

```bash
# Manual edit
vim pyproject.toml  # Change version = "0.1.0" to "0.2.0"

# Or use uv's version command
uv version 0.2.0                    # Set exact version
uv version --bump patch             # 0.1.0 -> 0.1.1
uv version --bump minor             # 0.1.0 -> 0.2.0
uv version --bump major             # 0.1.0 -> 1.0.0
```

### Step 2: Run Tests

Ensure all tests pass before publishing:

```bash
# Run full test suite
uv run pytest

# Run with coverage
uv run pytest --cov

# Run linting
uv run ruff check .
```

**Important**: Never publish without passing tests!

### Step 3: Build the Package

Clean previous builds and create fresh distribution files:

```bash
# Clean old builds
rm -rf dist/

# Build package (creates wheel + source distribution)
uv build

# Verify build artifacts
ls -lh dist/
# Should show:
# - arc-X.Y.Z-py3-none-any.whl (wheel)
# - arc-X.Y.Z.tar.gz (source distribution)
```

**Best Practice**: Test build without custom sources:

```bash
uv build --no-sources
```

### Step 4: Publish to TestPyPI

**Always test on TestPyPI first!**

```bash
# Set TestPyPI token
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmc..."  # Your TestPyPI token

# Publish to TestPyPI
uv publish --index testpypi

# Expected output:
# Uploading arc_ml-X.Y.Z-py3-none-any.whl
# Uploading arc_ml-X.Y.Z.tar.gz
# Successfully published 2 files
```

**View your package**: https://test.pypi.org/project/arc-ml/

### Step 5: Test Installation from TestPyPI

Verify the package installs correctly:

```bash
# Create a test virtual environment
uv venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from TestPyPI
# Note: TestPyPI doesn't have all dependencies, so use --index-url with fallback
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arc-ml

# Test the installation (note: command is still "arc" despite package name being "arc-ml")
arc --help
arc chat  # Test interactive mode

# Deactivate and clean up
deactivate
rm -rf test-env
```

**Alternative test with uv**:

```bash
# Test import without installing
uv run --with arc-ml --no-project -- python -c "import arc; print('Success!')"

# Force refresh (if testing updated version)
uv run --with arc-ml --refresh-package arc-ml --no-project -- arc --help
```

### Step 6: Publish to Production PyPI

**Only after successful TestPyPI verification!**

```bash
# Set production PyPI token
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmc..."  # Your PyPI token

# Publish to production PyPI
uv publish

# Expected output:
# Uploading arc_ml-X.Y.Z-py3-none-any.whl
# Uploading arc_ml-X.Y.Z.tar.gz
# Successfully published 2 files
```

**View your package**: https://pypi.org/project/arc-ml/

### Step 7: Create Git Tag

Tag the release in git:

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag to GitHub
git push origin v0.2.0

# Optional: Create GitHub Release via web UI or CLI
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes here..."
```

## Quick Reference Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Run tests: `uv run pytest`
- [ ] Run linting: `uv run ruff check .`
- [ ] Clean old builds: `rm -rf dist/`
- [ ] Build package: `uv build`
- [ ] Set TestPyPI token: `export UV_PUBLISH_TOKEN="..."`
- [ ] Publish to TestPyPI: `uv publish --index testpypi`
- [ ] Test installation from TestPyPI: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arc-ml`
- [ ] Set PyPI token: `export UV_PUBLISH_TOKEN="..."`
- [ ] Publish to PyPI: `uv publish`
- [ ] Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Create GitHub Release (optional)

## Troubleshooting

### Package Name vs. Command Name

The package is named `arc-ml` (to avoid conflicts on PyPI), but the command users run is still `arc`:

- **Install with**: `pip install arc-ml`
- **Run with**: `arc chat`

This is configured in `pyproject.toml` under `[project.scripts]`.

### Invalid API Token

```bash
# Error: Invalid credentials
# Solution: Regenerate token and update environment variable
export UV_PUBLISH_TOKEN="pypi-NEW-TOKEN-HERE"
```

### Version Already Published

```bash
# Error: File already exists
# Solution: Bump version number
uv version --bump patch
rm -rf dist/
uv build
uv publish
```

### Missing Build Artifacts

```bash
# Error: No files to upload
# Solution: Ensure build succeeded
rm -rf dist/
uv build
ls dist/  # Should show .whl and .tar.gz files
```

## Best Practices

1. **Always test on TestPyPI first** - Catch issues before production
2. **Use project-scoped API tokens** - More secure than account-wide tokens
3. **Run full test suite** - Never publish broken code
4. **Semantic versioning** - Follow SemVer (MAJOR.MINOR.PATCH)
5. **Tag releases in git** - Maintain clear version history
6. **Write release notes** - Document changes for users
7. **Clean builds** - Remove `dist/` before building

## Automated Publishing (Future)

Once you're comfortable with manual publishing, consider setting up GitHub Actions for automated releases:

1. Configure PyPI Trusted Publishing (no tokens needed!)
2. Create `.github/workflows/publish.yml`
3. Publish automatically when creating GitHub Releases

See: https://docs.pypi.org/trusted-publishers/

## Additional Resources

- **uv documentation**: https://docs.astral.sh/uv/guides/package/
- **PyPI Publishing Guide**: https://packaging.python.org/tutorials/packaging-projects/
- **TestPyPI**: https://test.pypi.org/
- **Production PyPI**: https://pypi.org/
- **Semantic Versioning**: https://semver.org/
