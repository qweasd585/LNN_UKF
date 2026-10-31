#!/usr/bin/env python3
"""Batch-convert a directory of CSV result files to one editable XLSX workbook.

PyCharm usage
-------------
1. Open this file in PyCharm and choose a Python interpreter with
   ``pandas`` and ``openpyxl`` installed.
2. Set ``DEFAULT_INPUT_DIRECTORY`` below, or set script parameters such as:

       --input results/synthetic_validation --output results/analysis.xlsx

3. Click Run.  Every CSV becomes an independently editable worksheet.  Excel
   font colours, widths, filters and number formatting are retained only when
   you save the generated ``.xlsx`` file, not when saving back to CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


# Edit these two values directly when running this script with PyCharm's green
# Run button and no command-line parameters.
DEFAULT_INPUT_DIRECTORY = Path("results/synthetic_validation")
DEFAULT_OUTPUT_FILE: Path | None = None


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read UTF-8 or common Chinese CSV encodings with a useful error message."""
    failures: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            failures.append(f"{encoding}: {error}")
    raise UnicodeDecodeError("csv", b"", 0, 1, f"cannot decode {path}; tried {'; '.join(failures)}")


def sheet_name(relative_path: Path, occupied: set[str]) -> str:
    """Derive a unique Excel-safe sheet name from a relative CSV path."""
    base = re.sub(r"[\\/*?:\[\]]", "_", "_".join(relative_path.with_suffix("").parts))[:31] or "Sheet"
    candidate = base
    suffix = 2
    while candidate in occupied:
        ending = f"_{suffix}"
        candidate = base[: 31 - len(ending)] + ending
        suffix += 1
    occupied.add(candidate)
    return candidate


def style_worksheet(worksheet) -> None:
    """Apply a compact, editable presentation format to one result sheet."""
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")
    for column_cells in worksheet.columns:
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 12), 36)
        for cell in column_cells[1:]:
            if isinstance(cell.value, (float, np.floating)):
                cell.number_format = "0.0000"


def convert_csv_directory(input_directory: Path, output_file: Path | None = None, recursive: bool = True) -> Path:
    """Convert all CSV files under ``input_directory`` into one XLSX workbook."""
    source = input_directory.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"CSV directory does not exist: {source}")
    csv_files = sorted(source.rglob("*.csv") if recursive else source.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no CSV files found under {source}")
    destination = (output_file or source / "csv_results.xlsx").expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    occupied: set[str] = set()
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for csv_file in csv_files:
            worksheet_name = sheet_name(csv_file.relative_to(source), occupied)
            read_csv_safely(csv_file).to_excel(writer, sheet_name=worksheet_name, index=False)
            style_worksheet(writer.book[worksheet_name])
    return destination


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-convert CSV result files into one formatted XLSX workbook")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIRECTORY, help="directory containing CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="XLSX output path; default is <input>/csv_results.xlsx")
    parser.add_argument("--no-recursive", action="store_true", help="convert only CSV files in the selected directory")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = convert_csv_directory(arguments.input, arguments.output, recursive=not arguments.no_recursive)
    print(f"Created editable Excel workbook: {output}")


if __name__ == "__main__":
    main()
