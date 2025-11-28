# CLAUDE.md - AI Assistant Guide

This document provides comprehensive guidance for AI assistants working with this ML learning repository.

## Repository Overview

**Purpose**: 35-day intensive AI/ML learning program - from Python engineer to ML expert
**Owner**: ML learner focused on transitioning to ML Quality Engineering
**Started**: November 27, 2025
**Current Status**: Day 1 - Environment setup complete

### Key Goals
1. Master ML fundamentals through deep learning
2. Build 8+ portfolio projects
3. Specialize in ML Quality Engineering
4. Target: ML/AI roles at companies like Netskope

## Repository Structure

```
ml-learning/
├── notebooks/          # Jupyter notebooks for learning and experiments
├── projects/           # Portfolio projects (target: 8+)
├── resources/          # Course materials and references
├── notes/              # Daily notes and learnings
├── .gitignore          # Python, Jupyter, ML, and macOS exclusions
└── README.md           # Main repository documentation
```

**Note**: Most directories are planned but not yet created. Create them as needed when the user starts working on specific days/topics.

## Development Environment

### Hardware & OS
- **Platform**: macOS with GPU support (Metal/MPS)
- **Primary Use**: Local development with GPU acceleration

### Key Technologies
- **Python**: Primary language (version TBD)
- **TensorFlow**: Configured with Metal backend for GPU acceleration
- **PyTorch**: Configured with MPS (Metal Performance Shaders) backend
- **JupyterLab**: Interactive notebook environment
- **VS Code**: IDE with ML extensions

### Environment Setup
- Virtual environment: Use `venv/` or `.venv/` (gitignored)
- Package management: Likely pip (confirm with user if needed)
- GPU: Leverage Metal/MPS for TensorFlow and PyTorch

## Git Workflow

### Branch Strategy
- **Main branch**: Default branch for stable content
- **Feature branches**: Use `claude/` prefix for AI-assisted development
- **Current branch**: `claude/claude-md-miirx99cacoa40wv-01JnR94Q5ihBatem9sB6NuYk`

### Commit Conventions
- Use clear, descriptive commit messages
- Focus on "why" rather than "what"
- Examples:
  - "Complete Day 2 linear regression exercises"
  - "Add image classification project with CNN"
  - "Document PyTorch training pipeline learnings"

### Push Protocol
- Always push to the designated `claude/` branch
- Use: `git push -u origin <branch-name>`
- Branch names must start with `claude/` and match session ID
- Retry on network failures (up to 4 times with exponential backoff)

## File Conventions

### Data Files
**IMPORTANT**: Large data files are gitignored and should NOT be committed:
- CSV files (*.csv)
- Parquet files (*.parquet)
- JSON data files (*.json)
- `data/` and `datasets/` directories

When working with datasets:
1. Document data sources in README or project documentation
2. Provide download links or instructions
3. Include data in `.gitignore`
4. Consider using DVC or similar for data versioning if needed

### ML Model Files
Trained models are gitignored:
- HDF5 files (*.h5)
- Pickle files (*.pkl)
- PyTorch models (*.pth)
- Checkpoints (*.ckpt)
- MLflow runs (mlruns/)
- Weights & Biases (wandb/)

Document model architectures and training procedures instead of committing large model files.

### Jupyter Notebooks
- Store in `notebooks/` directory
- Checkpoint files are gitignored (.ipynb_checkpoints)
- Clear outputs before committing if they contain large data
- Include markdown cells with explanations and learnings

## Code Style & Quality

### Python Conventions
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings for functions and classes
- Type hints are encouraged but not required (learning context)

### ML-Specific Guidelines
1. **Reproducibility**: Set random seeds for experiments
2. **Documentation**: Comment hyperparameters and design decisions
3. **Modularity**: Separate data loading, model definition, training, and evaluation
4. **Experimentation**: Track experiments with clear naming conventions

### Comments & Documentation
- Explain "why" not "what" in comments
- Document learning insights and discoveries
- Note challenges and solutions for future reference
- Keep notes in `notes/` directory for daily learnings

## AI Assistant Best Practices

### When Working with This Repository

1. **Respect the Learning Journey**
   - This is a learning repository, not production code
   - Encourage good practices but don't over-engineer
   - Provide explanations suitable for someone learning ML
   - Balance code quality with learning objectives

2. **File Creation**
   - Create directory structure (`notebooks/`, `projects/`, `resources/`, `notes/`) as needed
   - Don't create directories until they're actually needed
   - Follow the planned structure from README.md

3. **Code Assistance**
   - Explain ML concepts when introducing new techniques
   - Reference documentation and learning resources
   - Suggest experiments and variations to try
   - Help debug but encourage understanding

4. **Project Development**
   - Each project should be self-contained in `projects/`
   - Include README.md in each project folder
   - Document approach, results, and learnings
   - Keep projects portfolio-ready

5. **Notebook Practices**
   - Use clear section headers with markdown
   - Include visualizations and explanations
   - Document experiment results
   - Keep notebooks focused (one topic/experiment per notebook)

### Common Tasks

#### Starting a New Day's Work
```bash
# Create daily directory if needed
mkdir -p notes/day-XX
mkdir -p notebooks/day-XX

# Start with a clear plan in notes
```

#### Creating a New Project
```bash
# Create project structure
mkdir -p projects/project-name/{data,notebooks,src,models}

# Add project README
# Document: goal, dataset, approach, results, learnings
```

#### Running Experiments
- Use Jupyter notebooks for exploration
- Move production code to `.py` files in `src/`
- Track experiments in notes
- Save key results and visualizations

### Security & Safety

1. **No Credentials**: Never commit API keys, passwords, or tokens
2. **Data Privacy**: Don't commit personal or sensitive data
3. **Large Files**: Respect gitignore for large datasets and models
4. **Dependencies**: Document required packages (consider requirements.txt)

## Resource Management

### What to Commit
- Source code (.py files)
- Jupyter notebooks (with outputs cleared if large)
- Documentation and notes
- Project READMEs
- Configuration files (non-sensitive)
- Small sample datasets (< 1MB)

### What NOT to Commit
- Virtual environments
- Large datasets (use data/ directory)
- Trained models (document instead)
- Temporary files and caches
- IDE-specific files (except .vscode/ if useful)
- Sensitive information

## Progress Tracking

### Daily Structure
Each day should include:
1. Update to main README.md with day's focus
2. Notes in `notes/day-XX/` documenting learnings
3. Code/notebooks in appropriate directories
4. Commit summarizing the day's work

### Project Milestones
- Track portfolio projects (target: 8+)
- Document each project thoroughly
- Focus on quality over quantity
- Align projects with ML Quality Engineering goals

## Learning Path Considerations

### Week 1-2: Fundamentals
- Python for ML, NumPy, Pandas
- Linear algebra, calculus basics
- Basic ML algorithms

### Week 3-4: Deep Learning
- Neural networks fundamentals
- TensorFlow and PyTorch
- CNNs, RNNs, transformers

### Week 5: Specialization
- ML Quality Engineering focus
- Testing ML systems
- Model evaluation and monitoring
- Portfolio refinement

## Questions & Clarifications

When uncertain:
1. **Environment details**: Ask about Python version, specific library versions
2. **Project scope**: Confirm complexity level appropriate for learning
3. **Data sources**: Request dataset information when needed
4. **Time allocation**: Check if task fits current day's focus

## Version History

- **2025-11-28**: Initial CLAUDE.md creation (Day 1 - Environment setup complete)

---

**Remember**: This is a learning journey. Prioritize understanding over perfect code, encourage experimentation, and help build a strong portfolio that demonstrates both ML skills and quality engineering mindset.
