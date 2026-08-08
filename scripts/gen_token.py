#!/usr/bin/env python3
"""
Generate a JWT for local Cortex API testing.

Usage:
    python scripts/gen_token.py
    python scripts/gen_token.py --user-id my-user --tenant acme
    TOKEN=$(python scripts/gen_token.py)
    curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/runs ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path so we can import cortex without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env if present
from dotenv import load_dotenv
load_dotenv()

from cortex.api.auth import create_access_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cortex API token")
    parser.add_argument("--user-id", default="dev-user", help="User ID to encode in token")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--scopes", nargs="*", default=["read", "write"], help="Token scopes")
    args = parser.parse_args()

    try:
        token = create_access_token(
            user_id=args.user_id,
            tenant=args.tenant,
            scopes=args.scopes,
        )
        print(token)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Make sure SECRET_KEY is set in your .env file.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
