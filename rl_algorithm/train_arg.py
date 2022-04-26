"""
['run.py 所在的路径', '--alg=ppo2', '--env=Humanoid-v2', '--network=mlp', '--num_timesteps=2e7', '--ent_coef=0.1', '--num_hidden=32', '--num_layers=3', '--value_network=copy']

"""
class arg:
    def __init__(self):
        self.env = 'CartPole-v0'
        self.env_type = None
        self.alg = 'ppo2'
        self.num_timesteps = 1e6
        self.network = 'mlp'
        self.num_env = 0
        self.reward_scale = 1.0
        self.save_path =None
        self.log_path = None


class extra_args:
    def __init__(self):
        self.ent_coef = 0.1
        self.num_layers = 3
        self.num_hidden = 32
        self.value_network = copy

def main(args):
    # argparser参数部分  如果利用类的方式 可以将这里注释掉
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    # 调用模型和环境
    model, env = train(args, extra_args)
    if args.play:
        logger.log("Running trained model")
        # 环境重置
        obs = env.reset()
        episode_rew = 0
        # GYM式循环不断训练
        while True:
            #  下一步
            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()
    env.close()
    return model

