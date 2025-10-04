# CONTRIBUTING.md

## Contributing to the Omega Scanner Project

Thank you for your interest in the Omega Scanner. This project aims to detect long-range informational structure in sequential data through rigorous information-theoretic methods.

### Ways to Contribute

#### 1. **Replication and Validation**
The most valuable contribution is independent verification:
- Run the scanner on provided datasets and verify published results
- Test on new datasets and report findings (positive, negative, or null)
- Document any discrepancies between your results and published claims
- **Report negative results** - these are as scientifically valuable as positive findings

#### 2. **Code Improvements**
- Performance optimizations (especially for large datasets)
- Bug fixes with test cases demonstrating the issue
- Enhanced error handling and input validation
- Additional statistical tests or controls

#### 3. **Methodological Extensions**
- Alternative labeling schemes beyond IB clustering
- Different statistical frameworks for detecting Ω-signatures
- Novel controls for validating results
- Theoretical analysis of the method's assumptions and limitations

#### 4. **Documentation**
- Clarifications to EXTENDED.md where procedures are unclear
- Examples of common use cases
- Troubleshooting guides for typical errors
- Translation of documentation to other languages

### Submission Guidelines

#### Before Submitting
1. **Test thoroughly** - Ensure your changes work across different datasets
2. **Document changes** - Update relevant .md files if behavior changes
3. **Maintain reproducibility** - Include random seeds and parameters used
4. **Check existing issues** - Avoid duplicate submissions

#### Pull Request Process
1. Fork the repository
2. Create a descriptive branch name (`fix/bootstrap-bias`, `feature/new-labeler`, `docs/cli-examples`)
3. Make your changes with clear commit messages
4. Include tests or validation results where applicable
5. Update documentation to reflect changes
6. Submit PR with detailed description of:
   - What changed and why
   - How you tested the changes
   - Any breaking changes or new dependencies

#### Reporting Issues
When reporting bugs or unexpected behavior:
- Include the **exact command** that produced the issue
- Provide **sample data** (or describe data properties if sensitive)
- Specify your **environment** (OS, Python version, RAM)
- Include **full error output** or unexpected results
- State **expected vs actual behavior**

### Code Standards

- **Python 3.8+** compatibility
- **No external dependencies** beyond what's in requirements.txt without discussion
- **Reproducible results** - use explicit random seeds
- **Clear variable names** in new code (existing code uses terse names for space)
- **Comments for non-obvious logic**, especially in statistical procedures

### Scientific Standards

This project maintains high scientific rigor:

1. **Pre-register hypotheses** - Define success/failure criteria before running experiments
2. **Report all results** - Including negative findings and failed approaches
3. **Transparent methodology** - Document all parameters, not just successful runs
4. **Reproducible claims** - Include sufficient detail for independent replication
5. **Acknowledge limitations** - State what the method can and cannot demonstrate

### Collaboration Etiquette

- **Respectful discourse** - Critique ideas, not people
- **Evidence-based arguments** - Support claims with data or rigorous reasoning
- **Acknowledge uncertainty** - Distinguish proven results from preliminary findings
- **Credit appropriately** - Cite prior work that influenced your contribution

### What We're Looking For

**High Priority:**
- Independent replication on diverse datasets
- Identification of failure modes or edge cases
- Computational optimizations for large-scale analysis
- Theoretical analysis of statistical properties

**Medium Priority:**
- Alternative implementations (R, Julia, etc.)
- Visualization tools for results
- Integration with other information-theoretic methods

**Low Priority (but welcome):**
- Minor documentation fixes
- Code style improvements
- Performance micro-optimizations without validation

### Questions?

- Open an issue for methodology questions
- Use discussions for theoretical debates
- Email maintainers for sensitive topics

### License

By contributing, you agree that your contributions will be licensed under the same terms as the project (see LICENSE file).


