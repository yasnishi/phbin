import numpy as np
import matplotlib.pyplot as plt
from generator_porespy import generate_porespy_image
from ph_analysis import image_to_pd_pi
from config import PI_BINS, PI_SIGMA, PI_WEIGHT, PI_SUPERLEVEL


def compare_optimized_to_sem(sem_binary, mesh_range, optimized_params,
                             dist_weight, sampling_mode="separate", show=True):
    names = ['loc_small', 'scale_small', 'loc_large',
             'scale_large', 'porosity', 'r_min']
    alpha = dict(zip(names, optimized_params))
    generated = generate_porespy_image(
        sem_binary.shape, alpha, dist_weight,
        sampling_mode=sampling_mode
    )
    pd_sem, pi_sem, _ = image_to_pd_pi(
        sem_binary, mesh_range, bins=PI_BINS, sigma=PI_SIGMA, weight=PI_WEIGHT, superlevel=PI_SUPERLEVEL)
    pd_gen, pi_gen, _ = image_to_pd_pi(
        generated, mesh_range, bins=PI_BINS, sigma=PI_SIGMA, weight=PI_WEIGHT, superlevel=PI_SUPERLEVEL)
    mae = float(np.mean(np.abs(pi_sem-pi_gen)))
    rmse = float(np.sqrt(np.mean((pi_sem-pi_gen)**2)))
    if show:
        fig, ax = plt.subplots(2, 2, figsize=(11, 9))
        ax[0, 0].imshow(sem_binary, cmap='gray')
        ax[0, 0].set_title('Real SEM')
        ax[0, 0].axis('off')
        ax[0, 1].imshow(generated, cmap='gray')
        ax[0, 1].set_title('Generated')
        ax[0, 1].axis('off')
        ax[1, 0].scatter(pd_sem[0].births, pd_sem[0].deaths,
                         s=8, alpha=.5, label='SEM')
        ax[1, 0].scatter(pd_gen[0].births, pd_gen[0].deaths,
                         s=8, alpha=.5, label='Generated')
        ax[1, 0].set_xlabel('Birth')
        ax[1, 0].set_ylabel('Death')
        ax[1, 0].legend()
        ax[1, 0].set_title('PD')
        ax[1, 1].plot(pi_sem-pi_gen)
        ax[1, 1].set_title(f'PI difference / MAE={mae:.6f}')
        ax[1, 1].axis('off')
        plt.tight_layout()
        plt.show()
    return dict(pd_sem=pd_sem, pd_generated=pd_gen, pi_sem=pi_sem, pi_generated=pi_gen,
                pi_mae=mae, pi_rmse=rmse, generated_binary=generated, optimized_params=optimized_params)
