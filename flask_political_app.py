#!/usr/bin/env python3
"""
Flask Web Application for Political Evolution Genetic Algorithm

This web interface allows users to run political party evolution simulations
through a user-friendly web interface with real-time progress tracking.

Author: Phil Moyer (phil@moyer.ai)
Date: June 2025
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
import subprocess
import os
import json
import threading
import time
import uuid
from datetime import datetime
import zipfile
import tempfile
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store running simulations
active_simulations = {}

class SimulationRunner:
    def __init__(self, session_id, params):
        self.session_id = session_id
        self.params = params
        self.process = None
        self.output_dir = None
        self.status = "preparing"
        self.progress = 0
        self.logs = []
        
    def run(self):
        """Execute the simulation in a separate thread"""
        try:
            self.status = "running"
            self.emit_update()
            
            # Build command
            cmd = self.build_command()
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output
            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    self.process_output_line(output.strip())
            
            # Check result
            if self.process.returncode == 0:
                self.status = "completed"
                self.progress = 100
                self.find_output_directory()
            else:
                self.status = "error"
                self.add_log("Simulation failed with error", "error")
            
        except Exception as e:
            self.status = "error"
            self.add_log(f"Error: {str(e)}", "error")
        
        self.emit_update()
    
    def build_command(self):
        """Build the CLI command from parameters"""
        cmd = ["python", "political_cli.py"]
        
        # Add mode
        cmd.append(self.params['mode'])
        
        # Add common parameters
        if 'num_parties' in self.params:
            cmd.extend(['--num-parties', str(self.params['num_parties'])])
        if 'num_voters' in self.params:
            cmd.extend(['--num-voters', str(self.params['num_voters'])])
        if 'num_dimensions' in self.params:
            cmd.extend(['--num-dimensions', str(self.params['num_dimensions'])])
        if 'mutation_rate' in self.params:
            cmd.extend(['--mutation-rate', str(self.params['mutation_rate'])])
        if 'crossover_rate' in self.params:
            cmd.extend(['--crossover-rate', str(self.params['crossover_rate'])])
        if 'generations' in self.params:
            cmd.extend(['--generations', str(self.params['generations'])])
        
        # Add MCMC-specific parameters
        if self.params['mode'] == 'mcmc':
            if 'mcmc_runs' in self.params:
                cmd.extend(['--mcmc-runs', str(self.params['mcmc_runs'])])
            if 'mutation_rate_min' in self.params:
                cmd.extend(['--mutation-rate-min', str(self.params['mutation_rate_min'])])
            if 'mutation_rate_max' in self.params:
                cmd.extend(['--mutation-rate-max', str(self.params['mutation_rate_max'])])
            if 'crossover_rate_min' in self.params:
                cmd.extend(['--crossover-rate-min', str(self.params['crossover_rate_min'])])
            if 'crossover_rate_max' in self.params:
                cmd.extend(['--crossover-rate-max', str(self.params['crossover_rate_max'])])
        
        # Set output directory
        cmd.extend(['--output-dir', './web_output'])
        
        return cmd
    
    def process_output_line(self, line):
        """Process a line of output from the simulation"""
        self.add_log(line, "info")
        
        # Try to extract progress information
        if "Generation" in line and "/" in line:
            try:
                # Extract generation info like "Generation 50/300"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Generation" and i + 1 < len(parts):
                        gen_info = parts[i + 1]
                        if "/" in gen_info:
                            current, total = gen_info.split("/")
                            self.progress = int((int(current) / int(total)) * 100)
                            break
            except:
                pass
        
        # MCMC progress
        if "MCMC Run" in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Run" and i + 1 < len(parts):
                        run_info = parts[i + 1]
                        if "/" in run_info:
                            current, total = run_info.split("/")
                            self.progress = int((int(current) / int(total)) * 100)
                            break
            except:
                pass
        
        # Check for completion messages
        if "Analysis complete!" in line:
            self.progress = 100
        
        self.emit_update()
    
    def add_log(self, message, level="info"):
        """Add a log message"""
        self.logs.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level
        })
        
        # Keep only last 100 log entries
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
    
    def find_output_directory(self):
        """Find the output directory created by the simulation"""
        web_output = Path("./web_output")
        if web_output.exists():
            # Find the most recent run directory
            run_dirs = [d for d in web_output.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if run_dirs:
                self.output_dir = str(max(run_dirs, key=lambda d: d.stat().st_mtime))
                self.add_log(f"Results saved to: {self.output_dir}", "success")
    
    def emit_update(self):
        """Emit status update to the client"""
        socketio.emit('simulation_update', {
            'session_id': self.session_id,
            'status': self.status,
            'progress': self.progress,
            'logs': self.logs[-10:],  # Send last 10 log entries
            'output_dir': self.output_dir
        }, room=self.session_id)
    
    def stop(self):
        """Stop the running simulation"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status = "stopped"
            self.add_log("Simulation stopped by user", "warning")
            self.emit_update()

@app.route('/')
def index():
    """Main page with simulation interface"""
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    try:
        params = request.json
        session_id = str(uuid.uuid4())
        
        # Validate parameters
        if not validate_parameters(params):
            return jsonify({'error': 'Invalid parameters'}), 400
        
        # Create simulation runner
        runner = SimulationRunner(session_id, params)
        active_simulations[session_id] = runner
        
        # Start simulation in background thread
        thread = threading.Thread(target=runner.run)
        thread.daemon = True
        thread.start()
        
        return jsonify({'session_id': session_id, 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop a running simulation"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id in active_simulations:
            active_simulations[session_id].stop()
            return jsonify({'status': 'stopped'})
        else:
            return jsonify({'error': 'Simulation not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_results/<session_id>')
def download_results(session_id):
    """Download simulation results as a ZIP file"""
    try:
        if session_id not in active_simulations:
            return "Simulation not found", 404
        
        runner = active_simulations[session_id]
        if not runner.output_dir or not os.path.exists(runner.output_dir):
            return "Results not available", 404
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            output_path = Path(runner.output_dir)
            
            for file_path in output_path.rglob('*'):
                if file_path.is_file():
                    # Add file to ZIP with relative path
                    arcname = file_path.relative_to(output_path.parent)
                    zipf.write(file_path, arcname)
        
        temp_zip.close()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"political_simulation_results_{timestamp}.zip"
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        return f"Error creating download: {str(e)}", 500

def validate_parameters(params):
    """Validate simulation parameters"""
    required_fields = ['mode']
    
    for field in required_fields:
        if field not in params:
            return False
    
    # Validate numeric parameters
    numeric_params = {
        'num_parties': (1, 10),
        'num_voters': (100, 100000),
        'num_dimensions': (1, 20),
        'mutation_rate': (0.001, 0.5),
        'crossover_rate': (0.001, 0.5),
        'generations': (10, 1000),
        'mcmc_runs': (1, 500)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in params:
            try:
                value = float(params[param])
                if not (min_val <= value <= max_val):
                    return False
            except (ValueError, TypeError):
                return False
    
    return True

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_simulation')
def join_simulation(data):
    """Join a simulation room for updates"""
    session_id = data['session_id']
    session['simulation_id'] = session_id
    socketio.emit('joined', {'session_id': session_id})

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs('./web_output', exist_ok=True)
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)