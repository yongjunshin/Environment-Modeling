import numpy as np


# Virtual Line Tracer version 1
class LineTracerVer1:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        None

    def act(self, color: float) -> float:
        """
        System operation

        :param color: normalized color value (float)
        :return normalized turning ratio (float)
        """
        black = 5
        white = 75
        threshold = (black + white) / 2
        turning_ratio = 30

        denormalized_color = self.normalizer.inverse_transform(np.array([[color, 0, 0]], dtype="object"))[0][0]
        # print('color:', color)
        # print('denormalized color:', denormalized_color)

        if denormalized_color > threshold:
            None
        elif denormalized_color < threshold:
            turning_ratio = -turning_ratio
        else:
            turning_ratio = 0

        normalized_turning_ratio = self.normalizer.transform([[0, turning_ratio, 0]])[0][1]
        # print('turning:', turning_ratio)
        # print('normalized turning:', normalized_turning_ratio)
        return normalized_turning_ratio

    def act_sequential(self, colors: np.ndarray) -> np.ndarray:
        """
        System operation

        :param colors: array of normalized color values (np.ndarray of float)
        :return array of normalized turning ratios (np.ndarray of float)
        """
        black = 5
        white = 75
        threshold = (black + white) / 2
        turning_ratio = 30

        concated_colors = np.concatenate((colors, np.zeros((colors.shape[0], 2))), axis=1)
        denormalized_colors = self.normalizer.inverse_transform(concated_colors)
        denormalized_colors = denormalized_colors[:, 0]

        turning_ratios = np.zeros(len(denormalized_colors))
        turning_ratios[np.argwhere(denormalized_colors > threshold)] = turning_ratio
        turning_ratios[np.argwhere(denormalized_colors < threshold)] = -turning_ratio
        turning_ratios[np.argwhere(denormalized_colors == threshold)] = 0

        turning_ratios = np.reshape(turning_ratios, (turning_ratios.shape[0], 1))
        concated_turning_ratios = np.concatenate((np.zeros(turning_ratios.shape), turning_ratios, np.zeros(turning_ratios.shape)), axis=1)
        normalized_turning_ratio = self.normalizer.transform(concated_turning_ratios)[:, [1]]

        return normalized_turning_ratio

    def get_normalizer(self):
        return self.normalizer
