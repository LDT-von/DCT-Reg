"""Plan by default; --run explicitly delegates patient execution to this process."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.plugins.transport_only.slotspe_compare import main

if __name__ == "__main__":
    main()
