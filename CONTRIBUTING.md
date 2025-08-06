# Contributing to PointStream

Thank you for your interest in contributing to PointStream! This document provides guidelines and information for contributors.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/pointstream.git
   cd pointstream
   ```

2. **Set up the development environment**
   ```bash
   make dev-setup
   # or manually:
   conda env create -f environment.yml
   conda activate pointstream
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Install MMlab dependencies** (if needed)
   ```bash
   make install-mmlab
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make ci-check  # Runs all checks
   # or individually:
   make format    # Format code
   make lint      # Run linting
   make type-check # Type checking
   make test      # Run tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   Pre-commit hooks will run automatically.

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Code Style
- **Python**: Follow PEP 8 with line length of 88 characters
- **Formatting**: Use `black` and `isort` (run with `make format`)
- **Linting**: Code must pass `flake8` checks
- **Type hints**: Use type hints for all public functions

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

### Testing
- Write unit tests for all new functions
- Integration tests for pipeline components
- Test files should be named `test_*.py`
- Use pytest for testing framework

### Documentation
- Add docstrings to all public functions and classes
- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes

## Project Structure

```
pointstream/
├── cli.py              # Command line interface
├── config.py           # Configuration settings
├── client/             # Client-side reconstruction
├── models/             # AI model wrappers
├── pipeline/           # Core processing stages
├── scripts/            # Entry point scripts
└── utils/              # Utility functions
```

## Adding New Features

### New Pipeline Stage
1. Create module in `pointstream/pipeline/`
2. Follow the existing stage pattern
3. Add configuration to `config.py`
4. Write comprehensive tests
5. Update documentation

### New Model Support
1. Add model wrapper in `pointstream/models/`
2. Update `MODEL_REGISTRY` in `config.py`
3. Add model-specific tests
4. Document usage in README

### New CLI Command
1. Add function to `pointstream/cli.py`
2. Update `pyproject.toml` console scripts
3. Add tests for CLI functionality
4. Update documentation

## Research Contributions

PointStream is a research project. When contributing:

1. **Experimental features**: Use feature flags or separate branches
2. **Placeholder improvements**: Document what the ideal implementation would be
3. **Performance comparisons**: Include benchmarking data
4. **Paper reproduction**: Provide clear documentation of methods

## Testing

### Running Tests
```bash
# All tests
make test

# Specific test file
pytest tests/test_stage_01_analyzer.py -v

# With coverage
pytest tests/ --cov=pointstream --cov-report=html
```

### Test Data
- Small test files are in `tests/data/`
- Larger test datasets should be downloaded separately
- Mock external API calls in tests

## Getting Help

- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md and inline documentation

## Code Review Process

1. All changes require a pull request
2. At least one approval from a maintainer
3. All CI checks must pass
4. Code coverage should not decrease

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
