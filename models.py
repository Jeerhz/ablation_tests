# author: adle ben salem

from pydantic import BaseModel
import numpy as np


class Scores(BaseModel):
    """Scores for one options configuration"""

    precision: float | None
    recall: float | None
    f1_score: float | None
    iou: float | None


class ImageScores(Scores):
    """Scores for one image."""

    image_name: str


class BenchmarkScore(BaseModel):
    """Benchmark score for a model on a specific dataset."""

    options: str
    scores: list[ImageScores]

    def average_scores(self) -> Scores:
        """Calculate average scores across all images."""
        if not self.scores:
            return Scores(precision=None, recall=None, f1_score=None, iou=None)

        else:
            precisions = [s.precision for s in self.scores if s.precision is not None]
            recalls = [s.recall for s in self.scores if s.recall is not None]
            f1_scores = [s.f1_score for s in self.scores if s.f1_score is not None]
            ious = [s.iou for s in self.scores if s.iou is not None]

            avg_precision = float(np.mean(precisions)) if precisions else None
            avg_recall = float(np.mean(recalls)) if recalls else None
            avg_f1_score = float(np.mean(f1_scores)) if f1_scores else None
            avg_iou = float(np.mean(ious)) if ious else None

            return Scores(
                precision=avg_precision,
                recall=avg_recall,
                f1_score=avg_f1_score,
                iou=avg_iou,
            )
