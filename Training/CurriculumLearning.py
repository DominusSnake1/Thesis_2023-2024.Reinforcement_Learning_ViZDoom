from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes import TrainAndLog_Callback
from Training import CNNFeatureExtractor
from stable_baselines3 import PPO
from Models import Doom_Models


def deadly_corridor(self: Doom_Models, callback: TrainAndLog_Callback, timesteps: int):
    for skill in range(1, 5):
        current_level = f"{self.level}_s{skill}"

        doom = ViZDoom_Gym(level=current_level,
                           render=self.render,
                           reward_shaping=self.technique.reward_shaping,
                           curriculum=self.technique.curriculum)

        model = PPO(env=doom,
                    policy=self.technique.policy,
                    learning_rate=self.technique.learning_rate,
                    n_steps=self.technique.n_steps,
                    ent_coef=self.technique.ent_coef,
                    policy_kwargs={
                        'features_extractor_class': CNNFeatureExtractor,
                        'features_extractor_kwargs': {
                            'observation_space': doom.observation_space,
                            'number_of_actions': self.technique.number_of_actions
                        }
                    },
                    device=self.device,
                    tensorboard_log=self.log_dir,
                    verbose=1)

        log_name = f'{callback.get_formatted_datetime()}_{self.technique.algorithm}'
        model.learn(total_timesteps=timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=False)
