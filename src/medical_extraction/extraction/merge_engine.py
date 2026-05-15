"""Block merging and ordering helpers."""

from __future__ import annotations


def merge_blocks(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return []

    with_bbox = [dict(block) for block in blocks if _valid_bbox(block.get("bbox"))]
    without_bbox = [dict(block) for block in blocks if not _valid_bbox(block.get("bbox"))]
    ordered_with_bbox = _order_blocks_with_regions_and_columns(with_bbox)
    merged = _merge_adjacent_paragraphs(ordered_with_bbox)
    combined = merged + without_bbox

    for index, block in enumerate(combined, start=1):
        metadata = block.get("metadata") or {}
        metadata["reading_order"] = index
        block["metadata"] = metadata
    return combined


def _valid_bbox(bbox: list[float] | None) -> bool:
    return bool(bbox and len(bbox) == 4)


def _order_blocks_with_regions_and_columns(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return []

    page_height = max(float(block["bbox"][3]) for block in blocks)
    page_width = max(float(block["bbox"][2]) for block in blocks)
    regions: dict[str, list[dict]] = {}

    for block in blocks:
        region_key, region_priority = _resolve_region(block, page_height)
        metadata = block.get("metadata") or {}
        metadata["region"] = region_key
        metadata["region_priority"] = region_priority
        block["metadata"] = metadata
        regions.setdefault(region_key, []).append(block)

    ordered: list[dict] = []
    sorted_region_keys = sorted(
        regions.keys(),
        key=lambda key: (
            regions[key][0].get("metadata", {}).get("region_priority", 99),
            min(float(item["bbox"][1]) for item in regions[key]),
            key,
        ),
    )

    for region_key in sorted_region_keys:
        region_blocks = regions[region_key]
        clusters = _cluster_columns(region_blocks, page_width=page_width)
        for column_index, cluster in enumerate(clusters, start=1):
            cluster_sorted = sorted(
                cluster,
                key=lambda block: (float(block["bbox"][1]), float(block["bbox"][0]), block.get("block_id", "")),
            )
            for block in cluster_sorted:
                metadata = block.get("metadata") or {}
                metadata["column_index"] = column_index
                block["metadata"] = metadata
            ordered.extend(cluster_sorted)
    return ordered


def _resolve_region(block: dict, page_height: float) -> tuple[str, int]:
    metadata = block.get("metadata") or {}
    layout_label = str(metadata.get("layout_label", "")).strip().lower()
    layout_position = metadata.get("layout_position")
    if layout_label:
        priority_map = {
            "title": 0,
            "header": 0,
            "paragraph": 1,
            "other": 1,
            "table": 2,
            "form": 2,
            "footer": 3,
        }
        region_priority = priority_map.get(layout_label, 1)
        region_key = f"{layout_label}:{layout_position if layout_position is not None else 'na'}"
        return region_key, region_priority

    bbox = block["bbox"]
    y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
    if y_center <= page_height * 0.14:
        return "header:auto", 0
    if y_center >= page_height * 0.94:
        return "footer:auto", 3
    if str(block.get("type", "")).lower() in {"table", "form"}:
        return "structured:auto", 2
    return "body:auto", 1


def _cluster_columns(region_blocks: list[dict], page_width: float) -> list[list[dict]]:
    if not region_blocks:
        return []

    threshold = max(24.0, page_width * 0.08)
    sorted_blocks = sorted(region_blocks, key=lambda block: (float(block["bbox"][0]), float(block["bbox"][1])))
    clusters: list[dict] = []

    for block in sorted_blocks:
        bbox = block["bbox"]
        x_left = float(bbox[0])
        x_right = float(bbox[2])
        x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
        chosen = None
        best_score = -1.0
        for cluster in clusters:
            overlap_ratio = _interval_overlap_ratio(
                x_left,
                x_right,
                float(cluster["span_left"]),
                float(cluster["span_right"]),
            )
            center_distance = abs(x_center - float(cluster["mean_x"]))
            left_distance = abs(x_left - float(cluster["mean_left"]))
            same_column = overlap_ratio >= 0.10 and (center_distance <= threshold * 1.5 or left_distance <= threshold)
            if not same_column:
                continue
            score = overlap_ratio - (center_distance / max(page_width, 1.0))
            if score > best_score:
                best_score = score
                chosen = cluster
        if chosen is None:
            clusters.append(
                {
                    "mean_x": x_center,
                    "mean_left": x_left,
                    "span_left": x_left,
                    "span_right": x_right,
                    "items": [block],
                }
            )
            continue
        chosen["items"].append(block)
        item_count = len(chosen["items"])
        chosen["mean_x"] = ((chosen["mean_x"] * (item_count - 1)) + x_center) / item_count
        chosen["mean_left"] = ((chosen["mean_left"] * (item_count - 1)) + x_left) / item_count
        chosen["span_left"] = min(float(chosen["span_left"]), x_left)
        chosen["span_right"] = max(float(chosen["span_right"]), x_right)

    clusters.sort(key=lambda cluster: cluster["mean_x"])
    return [cluster["items"] for cluster in clusters]


def _interval_overlap_ratio(a_left: float, a_right: float, b_left: float, b_right: float) -> float:
    overlap = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    if overlap <= 0:
        return 0.0
    a_width = max(1.0, a_right - a_left)
    b_width = max(1.0, b_right - b_left)
    return overlap / min(a_width, b_width)


def _merge_adjacent_paragraphs(ordered_blocks: list[dict]) -> list[dict]:
    if not ordered_blocks:
        return []

    merged: list[dict] = []
    for block in ordered_blocks:
        if not merged:
            merged.append(block)
            continue
        previous = merged[-1]
        if _can_merge_paragraphs(previous, block):
            merged[-1] = _merge_paragraph_pair(previous, block)
        else:
            merged.append(block)
    return merged


def _can_merge_paragraphs(previous: dict, current: dict) -> bool:
    if str(previous.get("type", "")).lower() != "paragraph":
        return False
    if str(current.get("type", "")).lower() != "paragraph":
        return False

    previous_meta = previous.get("metadata") or {}
    current_meta = current.get("metadata") or {}
    previous_layout = str(previous_meta.get("layout_label", "")).strip().lower()
    current_layout = str(current_meta.get("layout_label", "")).strip().lower()
    if not previous_layout or previous_layout != current_layout:
        return False
    if previous_layout not in {"paragraph", "header", "title", "other"}:
        return False
    if previous_meta.get("region") != current_meta.get("region"):
        return False
    if previous_meta.get("column_index") != current_meta.get("column_index"):
        return False

    previous_bbox = previous["bbox"]
    current_bbox = current["bbox"]
    vertical_gap = max(0.0, float(current_bbox[1]) - float(previous_bbox[3]))
    horizontal_shift = abs(float(previous_bbox[0]) - float(current_bbox[0]))
    if vertical_gap > 22:
        return False
    if horizontal_shift > 45:
        return False

    if str(previous.get("source", "")).lower() != str(current.get("source", "")).lower():
        return False

    return True


def _merge_paragraph_pair(previous: dict, current: dict) -> dict:
    merged_text = "\n".join(part for part in [previous.get("text", "").strip(), current.get("text", "").strip()] if part)
    previous_bbox = previous["bbox"]
    current_bbox = current["bbox"]
    confidence = round(
        (float(previous.get("confidence", 0.7)) + float(current.get("confidence", 0.7))) / 2.0,
        2,
    )

    metadata = previous.get("metadata") or {}
    merged_ids = list(metadata.get("merged_block_ids", []))
    merged_ids.append(current.get("block_id"))
    metadata["merged_block_ids"] = merged_ids
    metadata["merged_line_count"] = int(metadata.get("merged_line_count", 1)) + 1

    previous["text"] = merged_text
    previous["bbox"] = [
        min(float(previous_bbox[0]), float(current_bbox[0])),
        min(float(previous_bbox[1]), float(current_bbox[1])),
        max(float(previous_bbox[2]), float(current_bbox[2])),
        max(float(previous_bbox[3]), float(current_bbox[3])),
    ]
    previous["confidence"] = confidence
    previous["metadata"] = metadata
    previous["needs_review"] = bool(previous.get("needs_review", False) or current.get("needs_review", False))
    return previous
