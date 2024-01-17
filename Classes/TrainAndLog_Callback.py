from datetime import datetime
import os
from stable_baselines3.common.callbacks import BaseCallback


def get_formatted_datetime():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%b%d-%H%M").upper()

    return formatted_datetime


class TrainAndLog_Callback(BaseCallback):
    def __init__(self, model_name, check_freq, level, verbose=1, adjustments=False):
        super(TrainAndLog_Callback, self).__init__(verbose)
        self.model_name = model_name
        self.check_freq = check_freq
        self.save_path = f'./Data/Train/train_{level}/{get_formatted_datetime()}'
        self._init_callback()
        self.adjustments = adjustments

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        mod = self.n_calls % self.check_freq
        div = int(self.n_calls / self.check_freq)

        if mod == 0:
            model_path = os.path.join(self.save_path, f'{self.model_name}_{div * 10}')

            if self.adjustments:
                model_path += "+"

            self.model.save(model_path)

        return super()._on_step()
