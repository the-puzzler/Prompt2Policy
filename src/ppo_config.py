"""PPO hyperparameters for FR3 reach task."""

PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(
        share_features_extractor=True,
        features_extractor_kwargs=dict(cnn_output_dim=128),
        net_arch=[128, 128],
    ),
}
