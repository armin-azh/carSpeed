import datetime
import numpy as np


def calc_iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes IUO between two bounding boxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


class Counter:
    def __init__(self, limit=None):
        """
        if limit is none: the counter is infinite
        :param limit:
        """
        self._cnt = 0
        self._limit = limit

    def __is_infinite(self) -> bool:
        return True if self._limit is None else False

    def reset(self) -> None:
        self._cnt = 0

    def next(self) -> None:
        if self.__is_infinite() or self._cnt != self._limit - 1:
            self._cnt += 1
        else:
            self._cnt = 0

    def __call__(self) -> int:
        return self._cnt


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._n_frame = Counter(limit=None)

    def start(self):
        self._start = datetime.datetime.now()

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._n_frame.next()

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self) -> float:
        try:
            if self._end is None:
                return self._n_frame() / (datetime.datetime.now() - self._start).total_seconds()
            return self._n_frame() / self.elapsed()
        except ZeroDivisionError:
            return 0.
