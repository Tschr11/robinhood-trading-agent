"""python -m src.data_import <spec.json> <raw.csv> [--new-version]"""

import sys

from src.data_import.importer import main

sys.exit(main())
