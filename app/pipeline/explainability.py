from typing import Dict, List, Any


class ExplainabilityEngine:
    """
    Generates human-understandable top contributing features / descriptors
    based on sequence facial micro-dynamics and model predictions.
    """

    def generate_top_features(
        self,
        seq_metrics: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[str]:
        features = []

        blink_rate = seq_metrics.get("blink_rate", 0.0)
        affect_variability = seq_metrics.get("affect_variability", 0.0)
        eyebrow_tension = seq_metrics.get("eyebrow_tension", 0.0)
        detection_rate = seq_metrics.get("detection_rate", 1.0)

        dep_score = scores.get("depression", 0.0)
        anx_score = scores.get("anxiety", 0.0)
        stress_score = scores.get("stress", 0.0)

        # 1. Affect variability rule
        if affect_variability < 0.03 or dep_score > 0.4:
            features.append("flattened affect")
        elif affect_variability > 0.12:
            features.append("dynamic facial expression variation")

        # 2. Blink dynamics rule
        if blink_rate < 0.08:
            features.append("reduced blink rate")
        elif blink_rate > 0.35 or anx_score > 0.4:
            features.append("elevated blink frequency")

        # 3. Eyebrow tension rule
        if eyebrow_tension > 0.04 or stress_score > 0.4:
            features.append("sustained eyebrow furrowing")

        # 4. Gaze stability / face tracking quality
        if detection_rate < 0.8:
            features.append("intermittent head movement / gaze shift")
        else:
            features.append("consistent frontal facial gaze")

        # 5. High stress indicator rule
        if stress_score > 0.5 and "sustained eyebrow furrowing" not in features:
            features.append("heightened facial muscle tension")

        return features[:4]
