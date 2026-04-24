"""
EcoDrive-Sentinel | Master Pipeline Runner
==========================================
Executes the full Phase 1 → Phase 2 → Phase 3 pipeline end-to-end.
Use this as your primary entry point for development and demonstration.

Usage:
    python run_pipeline.py --phase all          # Full pipeline
    python run_pipeline.py --phase features     # Phase 1 only
    python run_pipeline.py --phase train        # Phase 1 + 2
    python run_pipeline.py --phase agent        # Phase 3 demo (requires trained model)
    python run_pipeline.py --phase api          # Start FastAPI server
"""

from __future__ import annotations

import sys
import time
import typer
from pathlib import Path
from loguru import logger
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

# Configure logger
logger.remove()
logger.add(
    sys.stdout, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
)
logger.add("logs/pipeline.log", rotation="10 MB", retention="7 days", level="DEBUG")

app = typer.Typer(help="EcoDrive-Sentinel Pipeline Runner")


@app.command()
def run(
    phase: str = typer.Option("all", help="Pipeline phase: all | features | train | agent | api"),
    epochs: int = typer.Option(None, help="Override training epochs"),
    no_export: bool = typer.Option(False, help="Skip ONNX export"),
):
    """Run the EcoDrive-Sentinel pipeline."""

    rprint(Panel.fit(
        "[bold cyan]EcoDrive-Sentinel[/bold cyan]\n"
        "[dim]Predictive Maintenance for EV Batteries[/dim]\n"
        "[dim]Mercedes-Benz BEVisoneers | EU Battery Passport 2026[/dim]",
        border_style="cyan"
    ))

    if phase in ("all", "features", "train"):
        run_feature_engine()

    if phase in ("all", "train"):
        run_training(epochs_override=epochs, export_onnx=not no_export)

    if phase in ("all", "agent"):
        run_agent_demo()

    if phase == "api":
        run_api_server()


def run_feature_engine():
    """Phase 1: Feature Engineering."""
    rprint("\n[bold yellow]━━━ Phase 1: Feature Engine ━━━[/bold yellow]")
    from feature_engine import FeatureEngine
    from config import DATA_DIR

    engine = FeatureEngine()
    feature_df = engine.build_feature_matrix()

    # Save
    out_path = DATA_DIR / "feature_matrix.parquet"
    feature_df.to_parquet(out_path, index=False)

    table = Table(title="Feature Matrix Summary", style="cyan", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="yellow")
    table.add_row("Total Samples", f"{len(feature_df):,}")
    table.add_row("Unique Batteries", str(feature_df["battery_id"].nunique()))
    table.add_row("Data Sources", str(feature_df["source"].unique().tolist()))
    table.add_row("RUL Range", f"[{feature_df['rul'].min():.0f}, {feature_df['rul'].max():.0f}] cycles")
    table.add_row("Avg Capacity Fade", f"{feature_df['capacity_fade'].mean():.4f}")
    table.add_row("Saved to", str(out_path))
    rprint(table)
    return feature_df


def run_training(epochs_override: int | None = None, export_onnx: bool = True):
    """Phase 2: Model Training + ONNX Export."""
    rprint("\n[bold yellow]━━━ Phase 2: Predictive Core ━━━[/bold yellow]")
    import pandas as pd
    from predictive_core import ModelTrainer, export_to_onnx
    from config import DATA_DIR, settings

    if epochs_override:
        settings.epochs = epochs_override

    feature_path = DATA_DIR / "feature_matrix.parquet"
    if not feature_path.exists():
        logger.warning("Feature matrix not found. Running feature engine first...")
        run_feature_engine()

    feature_df = pd.read_parquet(feature_path)
    trainer = ModelTrainer()
    model, scaler, metrics = trainer.train(feature_df)

    table = Table(title="Training Results", style="green", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="yellow")
    for k, v in metrics.items():
        table.add_row(
            k.replace("_", " ").title(),
            f"{v:.4f}" if isinstance(v, float) else str(v)
        )
    rprint(table)

    if export_onnx:
        rprint("\n[bold cyan]Exporting to ONNX...[/bold cyan]")
        onnx_path = export_to_onnx(model, scaler)
        rprint(f"[bold green]✓ ONNX model ready:[/bold green] {onnx_path}")

        rprint("\n[bold]NPU Quantization Command (AMD Vitis-AI):[/bold]")
        rprint("[dim]  vai_q_onnx quantize_static \\[/dim]")
        rprint(f"[dim]    --input_model {onnx_path} \\[/dim]")
        rprint("[dim]    --calib_data_reader CalibrationDataReader \\[/dim]")
        rprint("[dim]    --quant_format QDQ \\[/dim]")
        rprint("[dim]    --activation_type QInt8 \\[/dim]")
        rprint("[dim]    --weight_type QInt8[/dim]")

    return model, scaler, metrics


def run_agent_demo():
    """Phase 3: Agentic Pipeline Demo."""
    import asyncio
    rprint("\n[bold yellow]━━━ Phase 3: Agentic Layer Demo ━━━[/bold yellow]")
    from agentic_layer import run_diagnostic_pipeline
    from config import SensorReading, ChemistryType

    scenarios = [
        ("🔴 CRITICAL — High Cycle Count", SensorReading(
            battery_id="MERC-EQS-B007",
            timestamp=time.time(),
            voltage=3.41,
            current=-12.5,
            temperature=38.2,
            cycle_count=390,
            chemistry=ChemistryType.NMC,
        )),
        ("🟡 WARNING — Mid Degradation", SensorReading(
            battery_id="MERC-EQS-B012",
            timestamp=time.time(),
            voltage=3.58,
            current=-9.0,
            temperature=30.1,
            cycle_count=220,
            chemistry=ChemistryType.NMC,
        )),
        ("🟢 NORMAL — Healthy Battery", SensorReading(
            battery_id="MERC-EQS-B001",
            timestamp=time.time(),
            voltage=3.72,
            current=-8.0,
            temperature=25.5,
            cycle_count=42,
            chemistry=ChemistryType.NMC,
        )),
    ]

    for label, sensor in scenarios:
        rprint(f"\n[bold]{label}[/bold]")
        rprint(f"  Battery: {sensor.battery_id} | Cycle: {sensor.cycle_count} | V: {sensor.voltage}V")

        t0 = time.perf_counter()
        report = asyncio.run(run_diagnostic_pipeline(sensor))
        elapsed = (time.perf_counter() - t0) * 1000

        rprint(f"  [bold]RUL:[/bold] [yellow]{report.rul_cycles:.1f}[/yellow] cycles")
        rprint(f"  [bold]Status:[/bold] {report.maintenance_status.value}")
        rprint(f"  [bold]Protocols:[/bold] {report.retrieved_protocols or ['None required']}")
        if report.recommended_actions:
            rprint("  [bold]Top Action:[/bold]", report.recommended_actions[0])
        rprint(f"  [dim]Pipeline time: {elapsed:.0f}ms[/dim]")


def run_api_server():
    """Start FastAPI server."""
    import uvicorn
    rprint("\n[bold yellow]━━━ Starting FastAPI Server ━━━[/bold yellow]")
    rprint("[bold cyan]API Docs:[/bold cyan] http://localhost:8000/docs")
    rprint("[bold cyan]ReDoc:[/bold cyan] http://localhost:8000/redoc")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    app()
