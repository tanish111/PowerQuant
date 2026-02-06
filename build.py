#!/usr/bin/env python3
"""
PowerQuant Build System

Usage:
    python build.py collect-dataset [--dataset DATASET]
    python build.py build-model <exp> [--file FILE] [--dataset DATASET]
    python build.py list-experiments
    python build.py --help
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Setup
console = Console()
ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = str(ROOT_DIR)

# Load environment variables
load_dotenv(ROOT_DIR / ".env")
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# Directories
DATASET_2_DIR = ROOT_DIR / "Dataset Collection" / "Dataset-2(works for Python PyTorch Models)" / "Benchmark Suite" / "KernelBench"
MODEL_TRAINING_DIR = ROOT_DIR / "Model Training"
DATA_DIR = MODEL_TRAINING_DIR / "data"
KERNELBENCH_SCRIPTS_DIR = DATASET_2_DIR / "scripts"


def setup_data_symlink():
    """Create symlink to dataset if it doesn't exist."""
    dataset_source = ROOT_DIR / "Dataset Collection" / "Dataset-2(works for Python PyTorch Models)" / "Dataset"
    if dataset_source.exists() and not DATA_DIR.exists():
        os.makedirs(DATA_DIR.parent, exist_ok=True)
        try:
            os.symlink(dataset_source, DATA_DIR)
            console.print(f"[green]✓[/green] Created symlink: {DATA_DIR} -> {dataset_source}")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not create symlink: {e}")


def run_command(cmd: List[str], cwd: Optional[Path] = None, description: str = "") -> int:
    """Run a shell command and return exit code."""
    if description:
        console.print(f"\n[cyan]→[/cyan] {description}")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    
    try:
        # Set PYTHONPATH and PROJECT_ROOT for Model Training directory
        env = os.environ.copy()
        if cwd:
            env["PYTHONPATH"] = str(cwd)
            # Set PROJECT_ROOT to Model Training directory for training scripts
            env["PROJECT_ROOT"] = str(cwd)
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        return result.returncode
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        return 1


@click.group()
@click.version_option()
def cli():
    """PowerQuant Build System - Manage data collection and model training."""
    pass


@cli.command()
@click.option(
    "--dataset",
    type=str,
    default="data/combined_static_20260127_092347.csv",
    help="Dataset CSV file (relative to Model Training/)"
)
def collect_dataset(dataset: str):
    """Collect baseline timing data from KernelBench models."""
    setup_data_symlink()
    
    script_path = KERNELBENCH_SCRIPTS_DIR / "generate_baseline_time.py"
    if not script_path.exists():
        console.print(f"[red]✗ Script not found:[/red] {script_path}")
        sys.exit(1)
    
    # Change to KernelBench directory for proper imports
    console.print(f"\n[bold cyan]Collecting Dataset[/bold cyan]")
    console.print(f"Script: {script_path}")
    console.print(f"Working directory: {DATASET_2_DIR}")
    
    exit_code = run_command(
        ["python", str(script_path)],
        cwd=DATASET_2_DIR,
        description="Running generate_baseline_time.py"
    )
    
    if exit_code == 0:
        console.print(f"[green]✓[/green] Dataset collection completed successfully")
    else:
        console.print(f"[red]✗[/red] Dataset collection failed with exit code {exit_code}")
    
    sys.exit(exit_code)


@cli.command()
@click.argument("exp", type=str)
@click.option(
    "--file",
    type=str,
    default=None,
    help="Specific file to run (e.g., run_quant_catboost.py). If not specified, runs all."
)
@click.option(
    "--dataset",
    type=str,
    default="data/combined_static_20260127_092347.csv",
    help="Dataset CSV file (relative to Model Training/)"
)
def build_model(exp: str, file: Optional[str], dataset: str):
    """Build/train model for a specific experiment."""
    setup_data_symlink()
    
    # Validate experiment
    exp_dir = MODEL_TRAINING_DIR / "experiments" / exp
    if not exp_dir.exists():
        console.print(f"[red]✗ Experiment not found:[/red] {exp_dir}")
        console.print(f"[dim]Available experiments:[/dim]")
        for e in sorted(MODEL_TRAINING_DIR.glob("experiments/exp*")):
            console.print(f"  - {e.name}")
        sys.exit(1)
    
    console.print(f"\n[bold cyan]Building Model[/bold cyan]")
    console.print(f"Experiment: {exp}")
    console.print(f"Directory: {exp_dir}")
    console.print(f"Dataset: {dataset}")
    
    # Find files to run
    if file:
        # Run specific file
        script_path = exp_dir / file
        if not script_path.exists():
            console.print(f"[red]✗ File not found:[/red] {script_path}")
            console.print(f"[dim]Available files in {exp}:[/dim]")
            for f in sorted(exp_dir.glob("run_*.py")):
                console.print(f"  - {f.name}")
            sys.exit(1)
        files_to_run = [script_path]
    else:
        # Run all run_*.py files
        files_to_run = sorted(exp_dir.glob("run_*.py"))
        if not files_to_run:
            console.print(f"[red]✗ No run_*.py files found in[/red] {exp_dir}")
            sys.exit(1)
    
    # Run each file
    failed = []
    succeeded = []
    
    for script_path in files_to_run:
        console.print(f"\n[cyan]→ Running:[/cyan] {script_path.name}")
        
        exit_code = run_command(
            ["python", str(script_path), "--dataset", dataset],
            cwd=MODEL_TRAINING_DIR,
            description=f"Training {script_path.stem}"
        )
        
        if exit_code == 0:
            console.print(f"[green]✓[/green] {script_path.name} completed")
            succeeded.append(script_path.name)
        else:
            console.print(f"[red]✗[/red] {script_path.name} failed with exit code {exit_code}")
            failed.append(script_path.name)
    
    # Summary
    console.print(f"\n[bold cyan]Summary[/bold cyan]")
    console.print(f"[green]Succeeded:[/green] {len(succeeded)}/{len(files_to_run)}")
    for name in succeeded:
        console.print(f"  [green]✓[/green] {name}")
    
    if failed:
        console.print(f"[red]Failed:[/red] {len(failed)}/{len(files_to_run)}")
        for name in failed:
            console.print(f"  [red]✗[/red] {name}")
        sys.exit(1)
    else:
        console.print(f"[green]✓ All files completed successfully[/green]")


@cli.command()
def list_experiments():
    """List available experiments."""
    exp_dir = MODEL_TRAINING_DIR / "experiments"
    
    if not exp_dir.exists():
        console.print(f"[red]✗ Experiments directory not found:[/red] {exp_dir}")
        sys.exit(1)
    
    console.print(f"\n[bold cyan]Available Experiments[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Experiment", style="cyan")
    table.add_column("Files", style="green")
    table.add_column("Path")
    
    for exp_path in sorted(exp_dir.glob("exp*")):
        if not exp_path.is_dir():
            continue
        
        run_files = list(exp_path.glob("run_*.py"))
        file_count = len(run_files)
        file_list = ", ".join(f.stem for f in sorted(run_files)[:3])
        if file_count > 3:
            file_list += f", ... (+{file_count - 3} more)"
        
        table.add_row(exp_path.name, f"{file_count} files", file_list)
    
    console.print(table)


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed paths")
def status(verbose: bool):
    """Show build system status."""
    console.print(f"\n[bold cyan]PowerQuant Build System Status[/bold cyan]\n")
    
    items = {
        "Root Directory": (ROOT_DIR, ROOT_DIR.exists()),
        "Dataset-2": (DATASET_2_DIR, DATASET_2_DIR.exists()),
        "KernelBench Scripts": (KERNELBENCH_SCRIPTS_DIR, KERNELBENCH_SCRIPTS_DIR.exists()),
        "Model Training": (MODEL_TRAINING_DIR, MODEL_TRAINING_DIR.exists()),
        "Data Symlink": (DATA_DIR, DATA_DIR.exists()),
        "PROJECT_ROOT env": (None, os.getenv("PROJECT_ROOT") == PROJECT_ROOT),
    }
    
    for name, (path, exists) in items.items():
        status_icon = "[green]✓[/green]" if exists else "[red]✗[/red]"
        if path:
            if verbose:
                console.print(f"{status_icon} {name}: {path}")
            else:
                console.print(f"{status_icon} {name}")
        else:
            console.print(f"{status_icon} {name}")
    
    # Count experiments
    exp_count = len(list(MODEL_TRAINING_DIR.glob("experiments/exp*")))
    console.print(f"\n[cyan]Experiments:[/cyan] {exp_count}")


if __name__ == "__main__":
    # Ensure PROJECT_ROOT is set
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
