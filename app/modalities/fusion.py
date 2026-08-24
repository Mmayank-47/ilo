from typing import Dict, List, Any
from app.config import settings
from app.api.v1.schemas import Scores, ScoreItem, AnalyzeResponse


class ModalityFusionEngine:
    """
    Combines analysis outputs from active modalities.
    Applies single-modality confidence calibration caps, combines indicator scores,
    generates top contributing features, and evaluates the human-review flag.
    """

    def combine_results(
        self,
        session_id: str,
        results_by_modality: Dict[str, Dict[str, Any]]
    ) -> AnalyzeResponse:
        modalities_used = list(results_by_modality.keys())
        is_single_modality = len(modalities_used) == 1

        # Indicators to aggregate
        indicators = ["depression", "anxiety", "stress"]
        aggregated_scores: Dict[str, Dict[str, float]] = {}
        all_features: List[str] = []

        for indicator in indicators:
            vals = []
            confs = []
            for mod_name, res in results_by_modality.items():
                mod_scores = res.get("scores", {})
                if indicator in mod_scores:
                    vals.append(mod_scores[indicator]["value"])
                    confs.append(mod_scores[indicator]["confidence"])
            
            if vals:
                # Weighted average (equal weight for now, or weighted by confidence)
                avg_val = sum(vals) / len(vals)
                avg_conf = sum(confs) / len(confs)
            else:
                avg_val = 0.0
                avg_conf = 0.0

            # Single-modality safety calibration
            if is_single_modality:
                avg_conf = min(avg_conf, settings.SINGLE_MODALITY_CONFIDENCE_CAP)

            aggregated_scores[indicator] = {
                "value": round(float(avg_val), 4),
                "confidence": round(float(avg_conf), 4)
            }

        # Collect and deduplicate top features
        for mod_name, res in results_by_modality.items():
            mod_features = res.get("features", [])
            for feat in mod_features:
                if feat not in all_features:
                    all_features.append(feat)

        # Flag for human review if ANY score exceeds elevated risk threshold
        flag_for_review = any(
            score_data["value"] >= settings.ELEVATED_RISK_THRESHOLD
            for score_data in aggregated_scores.values()
        )

        scores_obj = Scores(
            depression=ScoreItem(**aggregated_scores["depression"]),
            anxiety=ScoreItem(**aggregated_scores["anxiety"]),
            stress=ScoreItem(**aggregated_scores["stress"])
        )

        return AnalyzeResponse(
            session_id=session_id,
            modalities_used=modalities_used,
            scores=scores_obj,
            top_features=all_features[:5],  # Top 5 contributing features
            flag_for_review=flag_for_review
        )
