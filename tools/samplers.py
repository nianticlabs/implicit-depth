import torch


def add_noise_to_depth(
    source_depth_1hw,
    random_depth_sample_ratio=0.2,
    random_depth_sample_max=5,
    random_depth_sample_min=0.01,
    random_depth_mult_noise_sigma=0.1,
):
    """Sample depth values around the source_depth_1hw maps provided
    Args:
        source_depth_1hw: start point for jittering
        random_depth_sample_ratio: ratio of final point count to be random uniform depths within range
        random_depth_sample_max: max range for random sampling
        random_depth_sample_min: min range for random sampling
        random_depth_mult_noise_sigma: multiplicative ratio noise sigma added to source_depth_1hw
            output_depth = depth * (1 + sigma*randn)
    Returns:
        sample_depths_1hw: wiggled depth with random values with a ratio defined by random_depth_sample_ratio
        random_depth_maskb_1hw: mask where True marks random samples
    """

    sample_depths_1hw = source_depth_1hw.clone()

    # sample a single depth for each pixel. Random sample which to get random depth values within range.

    # wiggle depth
    sample_depths_1hw = sample_depths_1hw * (
        torch.randn(sample_depths_1hw.shape) * random_depth_mult_noise_sigma + 1.0
    )

    random_depth_samples_1hw = (
        torch.rand(sample_depths_1hw.shape) * random_depth_sample_max + random_depth_sample_min
    )

    random_depth_maskb_1hw = torch.rand(sample_depths_1hw.shape) < random_depth_sample_ratio

    sample_depths_1hw[random_depth_maskb_1hw] = random_depth_samples_1hw[random_depth_maskb_1hw]

    return sample_depths_1hw, random_depth_maskb_1hw
