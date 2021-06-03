import numpy as np

# Virtual Line Tracer version 1
class LineTracerVer1:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        None

    def act(self, color):
        black = 5
        white = 75
        threshold = (black + white) / 2
        turning_ratio = 10

        denormalized_color = self.normalizer.inverse_transform(np.array([[color, 0]], dtype="object"))[0][0]
        # print('color:', color)
        # print('denormalized color:', denormalized_color)

        if denormalized_color > threshold:
            None
        elif denormalized_color < threshold:
            turning_ratio = -turning_ratio
        else:
            turning_ratio = 0

        normalized_turning_ratio = self.normalizer.fit_transform([[0, turning_ratio]])[0][1]
        # print('turning:', turning_ratio)
        # print('normalized turning:', normalized_turning_ratio)
        return normalized_turning_ratio