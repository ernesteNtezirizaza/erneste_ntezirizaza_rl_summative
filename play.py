"""
play.py
=======
Rubric entry point for running the trained posture-coaching agent.

This wrapper delegates to main.py so all model-loading and runtime
behavior remains in one place.

Usage:
    python play.py
    python play.py --model ppo
    python play.py --model dqn
    python play.py --model reinforce
    python play.py --episodes 5
    python play.py --no-render
    python play.py --export-json
"""

from main import main


if __name__ == "__main__":
    main()
