from __future__ import annotations

import importlib


_FALLBACK_TOTAL_LABELS = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "spinal_cord",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "brain",
    "skull",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "sternum",
    "costal_cartilages",
]

_CARDIOVASCULAR = {
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
}

_BONES = {
    "sacrum",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "skull",
    "sternum",
    "costal_cartilages",
}

_NEURO = {
    "brain",
    "spinal_cord",
}


def _load_total_labels() -> list[str]:
    try:
        module = importlib.import_module("totalsegmentator.map_to_binary")
        class_map = getattr(module, "class_map", {})

        labels = list(class_map["total"].values())
        if labels:
            return labels
    except Exception:
        pass
    return list(_FALLBACK_TOTAL_LABELS)


def _group_name(label: str) -> str:
    if label.startswith("rib_"):
        return "Ribs"
    if label.startswith("vertebrae_"):
        return "Vertebrae"
    if label.startswith(("gluteus_", "autochthon_", "iliopsoas_")):
        return "Muscles"
    if label in _CARDIOVASCULAR:
        return "Cardiovascular"
    if label in _BONES:
        return "Bones"
    if label in _NEURO:
        return "Brain And Cord"
    return "Organs And Soft Tissue"


def _build_groups(labels: list[str]) -> dict[str, list[str]]:
    order = [
        "Organs And Soft Tissue",
        "Cardiovascular",
        "Bones",
        "Muscles",
        "Vertebrae",
        "Brain And Cord",
        "Ribs",
    ]
    grouped = {name: [] for name in order}
    for label in labels:
        grouped[_group_name(label)].append(label)
    return {name: items for name, items in grouped.items() if items}


TOTAL_SEGMENTATOR_STRUCTURES = _load_total_labels()
TOTAL_SEGMENTATOR_STRUCTURE_SET = set(TOTAL_SEGMENTATOR_STRUCTURES)
TOTAL_SEGMENTATOR_STRUCTURE_GROUPS = _build_groups(TOTAL_SEGMENTATOR_STRUCTURES)


def normalize_totalseg_labels(labels: list[str | None]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label is None:
            continue
        text = str(label).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if text in TOTAL_SEGMENTATOR_STRUCTURE_SET:
            valid.append(text)
        else:
            invalid.append(text)
    return valid, invalid
