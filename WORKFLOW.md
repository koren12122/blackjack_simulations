# DQN Training & Monte Carlo Workflow

This repository contains automated workflows for training a DQN agent and running Monte Carlo EV simulations on a blackjack card counting environment.

## 🤖 GitHub Actions Automation

Once you push this repository to GitHub, the workflows will automatically run on the following schedules:

### Monte Carlo Simulation
- **Schedule**: Every 2 hours
- **Workers**: 4 parallel workers
- **Hands per worker**: 3,500,000
- **Workflow file**: `.github/workflows/monte_carlo.yml`

### DQN Training
- **Schedule**: Once a week (Sunday at 2:00 AM UTC)
- **Training timesteps**: 2,000,000
- **Evaluation hands**: 300,000
- **Workflow file**: `.github/workflows/dqn_training.yml`

## 🚀 Getting Started

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add DQN and Monte Carlo workflows"
   git push
   ```

2. **Automatic Execution**: 
   - The workflows will start running automatically based on their schedules
   - No manual intervention required!

3. **Monitor Progress**:
   - Go to your GitHub repository
   - Click on the "Actions" tab
   - View the status of running or completed workflows

## 📊 Results

### Monte Carlo Results
- Each worker saves: `results/mc_worker_<ID>.pkl`
- Merged results: `results/mc_ev_combined.pkl` and `results/mc_ev_combined_ev_table.pkl`
- Results are automatically committed back to the repository

### DQN Results
- Trained model: `results/dqn_blackjack_counter.zip`
- Evaluation results: `results/dqn_eval_results.pkl`
- Results are automatically committed back to the repository

## 🔧 Manual Triggering

You can also manually trigger workflows from the GitHub Actions UI:

1. Go to **Actions** tab in your repository
2. Select the workflow (Monte Carlo or DQN Training)
3. Click **Run workflow**
4. Optionally adjust parameters:
   - Monte Carlo: `total_hands` per worker
   - DQN: `timesteps` and `eval_hands`

## 🛠️ Local Testing

To test locally before pushing:

### Monte Carlo (single worker, reduced size)
```bash
python run_monte_carlo.py --worker-id 0 --total-hands 100000 --output-dir test_results
```

### DQN (reduced parameters)
```bash
python run_dqn.py --timesteps 100000 --eval-hands 10000 --output-dir test_results
```

## 📝 Configuration

### Monte Carlo Workflow
- Runs 4 parallel workers simultaneously
- Each worker uses a different seed for diverse shoe shuffles
- Results are merged automatically after all workers complete
- Merged results are committed to the repository

### DQN Workflow
- Single process (no parallelization)
- Trains a DQN agent with 2 hidden layers of 256 neurons each
- Evaluates strategy by bucketing average reward by pre-deal true count
- Model and evaluation results are committed to the repository

## 🔄 Workflow Details

### Schedule Syntax
- Monte Carlo: `0 */2 * * *` (every 2 hours)
- DQN: `0 2 * * 0` (Sunday at 2:00 AM UTC)

### Dependencies
All required dependencies are listed in `requirements.txt`:
```
numpy
gymnasium
stable-baselines3[extra]
```

## 📦 Artifacts

GitHub Actions saves the following artifacts:

**Monte Carlo** (retention: 1 day):
- `mc-worker-0`, `mc-worker-1`, `mc-worker-2`, `mc-worker-3`

**DQN** (retention: 30 days):
- `dqn-model` (trained model)
- `dqn-eval-results` (evaluation results)

Artifacts can be downloaded from the Actions UI for analysis.

## 🎯 Next Steps

After pushing to GitHub:
1. ✅ Workflows are automatically scheduled
2. ✅ Check the Actions tab to monitor progress
3. ✅ Results will be committed to the repository automatically
4. ✅ You can trigger manual runs anytime via the Actions UI

No additional setup required! 🎉
