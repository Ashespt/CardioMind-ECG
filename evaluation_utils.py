from __future__ import annotations
import ast
import json
import math
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
LABEL_NAMES = json.loads((Path(__file__).parent / "config/model_labels.json").read_text())
CHINESE_LABELS = ("\u7aa6\u6027\u5fc3\u5f8b", "\u6b63\u5e38\u5fc3\u7535\u56fe", "ST-T\u6539\u53d8", "\u5176\u5b83", "\u623f\u6027\u5fc3\u5f8b\u5931\u5e38", "\u5fc3\u5ba4\u5185\u963b\u6ede", "\u5fc3\u808c\u6897\u6b7b", "\u5ba4\u6027\u5fc3\u5f8b", "\u5fc3\u5ba4\u80a5\u5927", "\u4f20\u5bfc\u963b\u6ede", "\u5fc3\u623f\u80a5\u5927", "\u623f\u5ba4\u8fde\u63a5\u5904\u5fc3\u5f8b", "Q-T\u95f4\u671f\u5f02\u5e38", "\u5fc3\u5ba4\u9884\u6fc0")
def normalize_label(label: object) -> str:
    return (
        str(label).strip()
        .replace("arrythmia", "arrhythmia")
        .replace("atiral", "atrial")
        .replace("myocaridal", "myocardial")
    )

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def load_raw_targets(manifest: list[dict], ground_truth: Path) -> np.ndarray:
    frame = pd.read_csv(ground_truth)
    if "Filename" in frame:
        filenames = frame["Filename"].map(
            lambda value: os.path.basename(str(value)).removesuffix(".dat")
        )
        label_column = "Diagnosis1_labels_en" if "Diagnosis1_labels_en" in frame else "Diagnosis1"
        labels_are_english = True
    else:
        visit_ids = pd.read_csv(ground_truth, dtype={"\u5c31\u8bca\u53f7": str})["\u5c31\u8bca\u53f7"]
        dates = pd.to_datetime(frame["\u68c0\u67e5\u65e5\u671f"])
        filenames = (
            frame["\u59d3\u540d"].astype(str).str.strip()
            + "_" + visit_ids.astype(str).str.strip()
            + "_" + dates.dt.strftime("%Y-%m-%d%H%M%S")
        )
        label_column = "Diagnosis1"
        labels_are_english = False
    if filenames.duplicated().any():
        raise ValueError(f"Duplicate filenames in {ground_truth}")
    labels_by_filename = dict(zip(filenames, frame[label_column]))
    label_index = {normalize_label(label): index for index, label in enumerate(LABEL_NAMES)}
    targets = np.zeros((len(manifest), len(LABEL_NAMES)), dtype=bool)
    for row_index, row in enumerate(manifest):
        filename = row["id"].split("/", 1)[-1]
        raw_labels = labels_by_filename.get(filename)
        if raw_labels is None or (isinstance(raw_labels, float) and math.isnan(raw_labels)):
            raise ValueError(f"Missing ground truth for {row['id']}")
        for raw_label in ast.literal_eval(raw_labels):
            if labels_are_english:
                normalized = normalize_label(raw_label)
            else:
                chinese = str(raw_label).strip().replace("\u2011", "-").replace("\uff0d", "-")
                if chinese == "\u623f\u6027\u5fc3\u5f8b":
                    chinese = "\u623f\u6027\u5fc3\u5f8b\u5931\u5e38"
                normalized = normalize_label(LABEL_NAMES[CHINESE_LABELS.index(chinese)]) if chinese in CHINESE_LABELS else ""
            if normalized not in label_index:
                raise ValueError(f"Unknown ground-truth label: {raw_label!r}")
            targets[row_index, label_index[normalized]] = True
    return targets

def write_workbook(path: Path, results: dict[str, list[dict]]) -> None:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def column_name(index: int) -> str:
        name = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def worksheet_xml(rows: list[dict]) -> str:
        columns = list(rows[0])
        values = [columns, *[[row[column] for column in columns] for row in rows]]
        row_xml = []
        for row_index, values_row in enumerate(values, 1):
            cells = []
            for column_index, value in enumerate(values_row, 1):
                reference = f"{column_name(column_index)}{row_index}"
                style = ' s="1"' if row_index == 1 else ""
                if value is not None and isinstance(value, (int, float, np.integer, np.floating)):
                    cells.append(f'<c r="{reference}"{style}><v>{value}</v></c>')
                else:
                    text = escape("" if value is None else str(value))
                    cells.append(
                        f'<c r="{reference}" t="inlineStr"{style}><is><t xml:space="preserve">{text}</t></is></c>'
                    )
            row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
        widths = []
        for column_index, column in enumerate(columns, 1):
            width = min(max(len(str(column)), *(len(str(row[column])) for row in rows)) + 2, 60)
            widths.append(
                f'<col min="{column_index}" max="{column_index}" width="{width}" customWidth="1"/>'
            )
        last_cell = f"{column_name(len(columns))}{len(values)}"
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{namespace}">'
            f'<dimension ref="A1:{last_cell}"/>'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
            'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            f'<cols>{"".join(widths)}</cols><sheetData>{"".join(row_xml)}</sheetData>'
            f'<autoFilter ref="A1:{last_cell}"/></worksheet>'
        )

    sheet_names = list(results)
    content_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheet_names) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{content_overrides}</Types>'
    )
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, 1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{namespace}" xmlns:r="{rel_namespace}"><sheets>{sheets}</sheets></workbook>'
    )
    workbook_relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheet_names) + 1)
    )
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{workbook_relationships}<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{namespace}"><fonts count="2"><font/><font><b/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="2"><xf/><xf fontId="1" applyFont="1"/></cellXfs></styleSheet>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/styles.xml", styles)
        for index, name in enumerate(sheet_names, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(results[name]))
