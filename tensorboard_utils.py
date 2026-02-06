#!/usr/bin/env python3
"""
Utility script to start/stop TensorBoard manually.
"""

import argparse
import subprocess
import webbrowser
import time
import socket
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
    except Exception:
        return False


def start_tensorboard(logdir: str, port: int, open_browser: bool = True):
    """Start TensorBoard server."""
    try:
        if not is_port_available(port):
            logger.info(f"TensorBoard already running on http://localhost:{port}")
            if open_browser:
                webbrowser.open(f"http://localhost:{port}")
            return True
        
        logger.info(f"Starting TensorBoard on http://localhost:{port}")
        logger.info(f"Log directory: {logdir}")
        
        cmd = [
            'tensorboard',
            '--logdir', logdir,
            '--port', str(port),
            '--host', 'localhost'
        ]
        
        # Start tensorboard process
        process = subprocess.Popen(cmd, start_new_session=True)
        
        # Wait a moment for tensorboard to start
        time.sleep(3)
        
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
            logger.info("TensorBoard opened in browser")
        
        logger.info("TensorBoard started successfully!")
        logger.info("Press Ctrl+C to stop TensorBoard")
        
        # Keep the script running
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping TensorBoard...")
            process.terminate()
            process.wait()
            logger.info("TensorBoard stopped")
        
        return True
        
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install with: pip install tensorboard")
        return False
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return False


def stop_tensorboard(port: int):
    """Stop TensorBoard server."""
    try:
        import psutil
        
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port and 'tensorboard' in proc.info['name'].lower():
                        logger.info(f"Found TensorBoard process with PID {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        logger.info("TensorBoard stopped successfully")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"No TensorBoard process found on port {port}")
        return False
        
    except ImportError:
        logger.warning("psutil not available. Install with: pip install psutil")
        return False
    except Exception as e:
        logger.error(f"Failed to stop TensorBoard: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="TensorBoard utility for satellite coordinate prediction")
    parser.add_argument("action", choices=["start", "stop", "status"], 
                       help="Action to perform")
    parser.add_argument("--logdir", type=str, default="logs",
                       help="TensorBoard log directory")
    parser.add_argument("--port", type=int, default=6006,
                       help="TensorBoard port")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    if args.action == "start":
        success = start_tensorboard(args.logdir, args.port, not args.no_browser)
        sys.exit(0 if success else 1)
    
    elif args.action == "stop":
        success = stop_tensorboard(args.port)
        sys.exit(0 if success else 1)
    
    elif args.action == "status":
        if is_port_available(args.port):
            logger.info(f"Port {args.port} is available - TensorBoard not running")
        else:
            logger.info(f"TensorBoard appears to be running on http://localhost:{args.port}")
            logger.info("Use 'python tensorboard_utils.py stop' to stop it")


if __name__ == "__main__":
    main()