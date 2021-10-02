from typing import Union, Tuple
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from datetime import datetime

from ._utls import calc_iou
from ._utls import Counter
from ._kalman import KalmanBoxTracker


class Tracker:
    def __init__(self, *args, **kwargs):
        super(Tracker, self).__init__(*args, **kwargs)

    def detect(self, *args, **kwargs):
        raise NotImplementedError


class SortTracker(Tracker):
    def __init__(self, detect_interval: int, iou_threshold: float, max_age: int, min_hit: int,
                 *args, **kwargs):
        super(SortTracker, self).__init__(*args, **kwargs)
        self._detect_interval = detect_interval
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hit = min_hit

        self._frame_count = 0  # =====> should change to Counter

        self._trackers = []

    def _associate_detections_to_trackers(self, detections: np.ndarray, trackers: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=np.int), np.arange(len(detections)), np.empty((0, 5), dtype=np.int)

        iou_mat = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_mat[d, t] = calc_iou(det, trk)

        matched_indices = linear_assignment(-iou_mat)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matched = []
        for m in matched_indices:
            if iou_mat[m[0], m[1]] < self._iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matched.append(m.reshape(1, 2))
        if len(matched) == 0:
            matched = np.empty((0, 2), dtype=np.int)
        else:
            matched = np.concatenate(matched, axis=0)
        return matched, np.array(unmatched_detections), np.array(unmatched_trackers)

    def _update(self, detects: Union[np.ndarray, list], frame_size: Tuple[int, int]):
        self._frame_count += 1
        tracks = np.zeros((len(self._trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(tracks):
            pos = self._trackers[t].predict()
            trk[:] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))

        for t in reversed(to_del):
            self._trackers.pop(t)

        if not detects == []:
            matched, unmatched_detections, unmatched_trackers = self._associate_detections_to_trackers(detects, tracks)

            for t, trk in enumerate(self._trackers):
                if t not in unmatched_trackers:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(detects[d, :][0])

            for i in unmatched_detections:
                trk = KalmanBoxTracker(bbox=detects[i, :])
                self._trackers.append(trk)

        _i = len(self._trackers)

        for trk in reversed(self._trackers):
            if detects == []:
                trk.update([])
            d = trk.get_state()

            if (trk.time_since_update < 1) and (trk.hit_streak >= self._min_hit or self._frame_count <= self._min_hit):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

            _i -= 1

            if trk.time_since_update >= self._max_age or trk.predict_num >= self._detect_interval or d[2] < 0 or d[
                3] < 0 or d[
                0] > \
                    frame_size[1] or d[1] > frame_size[0]:
                self._trackers.pop(_i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def detect(self, road_objets: np.ndarray, frame: np.ndarray, frame_size: Tuple[int, int], *args, **kwargs):

        final_cars = road_objets if len(road_objets) > 0 else []

        trackers = self._update(final_cars, frame_size)

        return trackers


class TrackerContainer:
    KNOWN = "known"
    UNKNOWN = "unrecognized"

    def __init__(self, tracker_id: int):
        self._trk_id = tracker_id
        self._send = False
        self._modified = self._now()
        self._last_seen_image = None
        self._observation = {
            self.UNKNOWN: Counter()
        }

    @property
    def trk_id(self):
        return self._trk_id

    @property
    def total_counter(self) -> int:
        """
        get total number of tracker id observed
        :return:
        """
        obs_cnt = [obs() for obs in self._observation.values()]
        return sum(obs_cnt) if sum(obs_cnt) != 0 else 1

    @property
    def unknown_counter(self) -> int:
        return self._observation[self.UNKNOWN]()

    def summary(self):
        """
        get summary of all identity name in observations
        :return:
        """
        total_cnt = self.total_counter
        if total_cnt == 0:
            ans = {key: 0 for key in self._observation.keys()}
            return ans
        ans = {key: item() / total_cnt for key, item in self._observation}
        return ans

    @property
    def most_valuable_id(self) -> Union[Tuple[str, int], Tuple[None, None]]:
        """
        find most cnt value in observations
        :return:
        """
        _ids = []
        _values = []
        for key, value in self._observation.items():
            if key == self.UNKNOWN:
                continue
            _ids.append(key)
            _values.append(value())

        if not _ids and not _values:
            return None, None
        _top_idx = np.argmax(_values)
        return _ids[_top_idx], _values[_top_idx]

    @property
    def image(self) -> np.ndarray:
        return self._last_seen_image

    @image.setter
    def image(self, mat: np.ndarray) -> None:
        self._last_seen_image = mat

    def _now(self) -> datetime:
        return datetime.now()

    def _delta(self, _e: datetime, _s: datetime) -> float:
        """
        get total seconds
        :param _e:
        :param _s:
        :return:
        """
        return (_e - _s).total_seconds()

    @property
    def delta(self) -> float:
        _now = datetime.now()
        return self._delta(_now, self._modified)

    def __call__(self, mat: Union[None, np.ndarray], identity: str, *args, **kwargs):
        try:
            self._observation[identity].next()

        except KeyError:
            self._observation[identity] = Counter()
            self._observation[identity].next()

        if mat is not None:
            self._last_seen_image = mat
        self._modified = self._now()

    @property
    def send(self) -> bool:
        return self._send

    def sent(self):
        self._send = True
