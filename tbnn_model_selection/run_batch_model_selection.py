#!/usr/bin/env python
"""
Batch runner for TBNN model selection across multiple iterations.
Runs model_selection_tbnn.py for each iteration with clean, non-verbose output.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_batch_model_selection():
    """Run model selection on multiple iterations and compile clean results."""
    
    # List of iteration folders to process
    iterations = [
        "cy_param_01_20251011_032016",
        "cy_param_03_20251011_032025", 
        "cy_param_04_20251011_032025",
        "cy_param_06_20251011_032025",
        "cy_param_07_20251011_032036",
        "cy_param_09_20251011_032025",
        "cy_param_11_20251011_032025"
    ]
    
    output_file = "model_selection_batch_results_clean.txt"
    
    print("="*70)
    print("TBNN MODEL SELECTION BATCH RUNNER")
    print("="*70)
    print(f"Processing {len(iterations)} iterations...")
    print(f"Output file: {output_file}")
    
    # Initialize output file with header
    with open(output_file, 'w') as f:
        f.write("TBNN MODEL SELECTION BATCH RESULTS (CLEAN)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total iterations: {len(iterations)}\n")
        f.write("="*70 + "\n\n")
    
    # Process each iteration
    for i, iteration in enumerate(iterations):
        print(f"\nProcessing {i+1}/{len(iterations)}: {iteration}")
        
        try:
            # Modify the Python script to run in non-verbose mode for batch
            # We'll temporarily modify the config in model_selection_tbnn.py
            result = subprocess.run(
                [sys.executable, "model_selection_tbnn.py", iteration],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per iteration
            )
            
            # Extract key results from stdout
            stdout_lines = result.stdout.strip().split('\n')
            stderr_lines = result.stderr.strip().split('\n') if result.stderr.strip() else []
            
            # Write results to output file
            with open(output_file, 'a') as f:
                f.write(f"ITERATION {i+1}: {iteration}\n")
                f.write("="*50 + "\n")
                
                # Extract and write key information only
                in_model_comparison = False
                for line in stdout_lines:
                    # Skip verbose progress bars and debug info
                    if any(skip_phrase in line for skip_phrase in [
                        "Loss:", "it/s]", "DEBUG:", "Using converted file:", 
                        "Using original file:", "Parameter compatibility verified",
                        "Generated single stress-strain", "Loaded viscosity curve",
                        "Inferred viscosity bounds", "Reconstructed TBNN model"
                    ]):
                        continue
                        
                    # Include key results - EXPANDED to catch fitted parameter values
                    if any(key_phrase in line for key_phrase in [
                        "Using results directory:", "Reading training parameters",
                        "Parsed training parameters:", "Model type:", "etainf:", "eta0:", "lam:", "n:", "a:",
                        "initial guesses", "Newtonian fit completed", "Carreau-Yasuda fit succeeded!",
                        "MODEL COMPARISON RESULTS", "Newtonian:", "CarreauYasuda:", "BIC:", " BEST MODEL:",
                        "Model fitting and BIC comparison completed!", "", "Failed", "SUCCESS",
                        # ADDED: Fitted parameter results
                        "Viscosity:", "(zero shear):", "(infinite shear):", "(time constant):", 
                        "(power index):", "(Yasuda param):", "Parameters:"
                    ]):
                        f.write(line + "\n")
                        
                    # Track when we're in the model comparison section
                    if "MODEL COMPARISON RESULTS" in line:
                        in_model_comparison = True
                    elif in_model_comparison and line.strip() == "":
                        in_model_comparison = False
                    elif in_model_comparison:
                        f.write(line + "\n")
                
                # Add any critical errors from stderr
                if stderr_lines and any(line.strip() for line in stderr_lines):
                    f.write("\nERRORS:\n")
                    for line in stderr_lines:
                        if line.strip():
                            f.write(f"  {line}\n")
                
                f.write("\n" + "="*50 + "\n\n")
            
            print(f"  Completed: {iteration}")
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout: {iteration} (skipped)")
            with open(output_file, 'a') as f:
                f.write(f"ITERATION {i+1}: {iteration}\n")
                f.write("STATUS: TIMEOUT (>5 minutes)\n")
                f.write("="*50 + "\n\n")
                
        except Exception as e:
            print(f"  Failed: {iteration} - {e}")
            with open(output_file, 'a') as f:
                f.write(f"ITERATION {i+1}: {iteration}\n")
                f.write(f"STATUS: ERROR - {e}\n")
                f.write("="*50 + "\n\n")
    
    # Add summary
    with open(output_file, 'a') as f:
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total iterations: {len(iterations)}\n\n")
        f.write("Quick summary (run this to extract key results):\n")
        f.write("  grep ' BEST MODEL:' model_selection_batch_results_clean.txt\n")
        f.write("  grep 'BIC: ' model_selection_batch_results_clean.txt\n")
    
    print("")
    print(f" Clean results saved to: {output_file}")
    print(f"{len(iterations)} iterations processed")
    
    # Show quick summary
    print(f"\n QUICK SUMMARY:")
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Extract best models
        best_models = []
        for line in content.split('\n'):
            if ' BEST MODEL:' in line:
                best_models.append(line.strip())
        
        for i, best in enumerate(best_models):
            print(f"  {iterations[i]}: {best}")
            
    except Exception as e:
        print(f"  Could not generate summary: {e}")

if __name__ == "__main__":
    run_batch_model_selection()
