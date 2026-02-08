# Blackjack Card Counting - GitHub Simulation

Automated training and evaluation of blackjack strategies using GitHub Actions.

## 🤖 Automated Workflows

This project includes two GitHub Actions workflows that run automatically:

### 1. Monte Carlo EV Simulation
- **Schedule**: Every 2 hours
- **Workers**: 4 parallel workers
- **File**: `.github/workflows/monte_carlo.yml`

### 2. DQN Training
- **Schedule**: Once a week (Sunday 2:00 AM UTC)
- **File**: `.github/workflows/dqn_training.yml`

## 📚 Documentation

See [WORKFLOW.md](WORKFLOW.md) for detailed information about:
- How the workflows work
- Manual triggering
- Local testing
- Results and artifacts

## 🚀 Quick Start

1. Push this repository to GitHub
2. Workflows start automatically - no setup required!
3. Monitor progress in the Actions tab

For detailed information, see [WORKFLOW.md](WORKFLOW.md).
s
