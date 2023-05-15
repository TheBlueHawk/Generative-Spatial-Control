import itertools
import operator
from typing import Dict, List, NamedTuple, Tuple


def _calc_centroid(bbox: list) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    height, width =  abs(y2-y1), abs(x2-x1)
    # Calculate centroid coordinates
    centroid = (x1 + (width / 2), y1 + (height / 2))
    return centroid


class VisorTuple(NamedTuple):
    """Represents a VISOR prompt. Cached to disk as info.json"""
    obj1: str
    obj2: str
    prompt: str
    relationship: str  # left, right, above, below


class DetectionResult(NamedTuple):
    """A single detection result, instantiated from DetectionTuple.__get__"""
    labels: List[int]
    boxes: List[list]
    scores: List[int]
    
    def __post_init__(self):
        # Automatic sanity checks
        assert len(self.labels) == len(self.boxes) == len(self.scores)
        assert {0, 1} >= set(self.labels), "Labels must be 0 or 1"
        for bbox in self.boxes:
            assert len(bbox) == 4, "Bbox must be [x1, y1, x2, y2]"
        for score in self.scores:
            assert 0 <= score <= 1, "Scores must be between 0 and 1"

    @property
    def centroids(self) -> List[Tuple[float, float]]:
        result = []
        for box in self.boxes:
            result.append(_calc_centroid(box))
        return result

    def check_oa(self) -> bool:
        """Return true if Object Accuracy (OA) is 1."""
        return len(set(self.labels)) == 2

    def check_visor(self, vtup: VisorTuple) -> bool:
        """Return true if the VISOR prompt `vtup` is satisfied by the detection result.
        (Assumes that result corresponds to the VISOR prompt `vtup`.)
        """
        if 0 not in self.labels or 1 not in self.labels:
            return False
        idx1 = self.labels.index(0)
        idx2 = self.labels.index(1)
        cent1 = self.centroids[idx1]
        cent2 = self.centroids[idx2]
        if vtup.relationship == "left":
            return cent1[0] < cent2[0]
        elif vtup.relationship == "right":
            return cent1[0] > cent2[0]
        elif vtup.relationship == "above":
            return cent1[1] < cent2[1]
        elif vtup.relationship == "below":
            return cent1[1] > cent2[1]
        else:
            raise ValueError(vtup.relationship)


# def _calc_trials_success(N, s, k) -> float:
#     # Suppose that we have N trials, and s successes.
#     # Now randomly draw from k of those N trials.
#     # Returns the chance that the k trials are all successful.
#     assert 0 <= s <= N
#     assert 0 < k <= N
#     if s > k:
#         return 0
#     if s == 0:
#         return 0
# 
#     # (s choose k) / (N choose k)
#     #   = (s * (s-1) ... * (s-k+1)) / (N * (N-1) * (N-k+1))
#     result = 1
#     for num in range(s, s-k, -1):
#         result *= num
#     for denom in range(N, N-k, -1):
#         denom *= num
# def _make_aggregates(success: List[bool]) -> List[bool]:
#     return [_calc_trials_success(N, s, k) for k in range(1, N+1)]

def _make_aggregates(success: List[bool]) -> List[bool]:
    ok = True
    result = []
    for s in success:
        ok = ok and s
        result.append(ok)
    return result


class DetectionTuple(NamedTuple):
    """Cached to disk as boxes.json"""
    img_names: list
    results: List[dict]  # dict keys: boxes, scores, labels

    def __len__(self) -> int:
        return len(self.results)

    def get_result(self, i) -> DetectionResult:
        res = self.results[i]
        return DetectionResult(res["labels"], res["boxes"], res["scores"])

    def __iter__(self) -> List[DetectionResult]:
        return [self.get_result(i) for i in range(len(self))]

    def calc_visors(self, vtup: VisorTuple) -> dict:
        """Returns a dictionary summarizing individual and aggregate visor scores."""
        assert len(self) == 4
        # Mapping from image name to visor 
        individual_scores = {}
        for i in range(len(self)):
            visor = self.get_result(i).check_visor(vtup)
            individual_scores[self.img_names[i]] = visor
        
        ret_dict = {"by_img": individual_scores,
                    "aggregates": _make_aggregates(individual_scores.values()),
                    }
        return ret_dict

    def calc_oas(self, vtup: VisorTuple):
        """Returns a dictionary summarizing individual and aggregate OA scores."""
        individual_scores = {}
        for i in range(len(self)):
            oa = self.get_result(i).check_oa()
            individual_scores[self.img_names[i]] = oa
        ret_dict = {"by_img": individual_scores,
                    "aggregates": _make_aggregates(individual_scores.values()),
                    }
        return ret_dict

    def calc_oa1(self, vtup: VisorTuple):
        """Returns a dictionary summarizing individual and aggregate OA scores."""
        individual_scores = {}
        for i in range(len(self)):
            individual_scores[self.img_names[i]] = 0 in self.get_result(i).labels
        ret_dict = {"by_img": individual_scores,
                    "aggregates": _make_aggregates(individual_scores.values()),
                    }
        return ret_dict

    def calc_oa2(self, vtup: VisorTuple):
        """Returns a dictionary summarizing individual and aggregate OA scores."""
        individual_scores = {}
        for i in range(len(self)):
            individual_scores[self.img_names[i]] = 1 in self.get_result(i).labels
        ret_dict = {"by_img": individual_scores,
                    "aggregates": _make_aggregates(individual_scores.values()),
                    }
        return ret_dict

    def get_metrics() -> dict:
        """Returns a dictionary summarizing individual and aggregate visor scores."""
        # TODO(shwang): Return metrics dictionary and start evaluating. Close the loop.