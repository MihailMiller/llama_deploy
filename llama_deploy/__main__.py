"""
Entry point: python -m llama_deploy [deploy|tokens] [options]

Routes to cli.dispatch() which handles:
  - TTY detection → interactive wizard
  - non-TTY / --batch → argparse flags
  - tokens subcommand → TokenStore operations
"""

from llama_deploy.cli import dispatch

if __name__ == "__main__":
    dispatch()
