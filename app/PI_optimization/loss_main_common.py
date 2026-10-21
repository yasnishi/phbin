import numpy as np
from config import BOUNDS, RATIO, LOC_SMALL, NBINS, GAP_THRESHOLD, PI_BINS
from optimization import make_bounds, optimize_parameters, pi_loss
from particle_analysis import estimate_bimodal_parameters
from real_sem import prepare_real_sem
from optimized_comparison import compare_optimized_to_sem
from generator_porespy import generate_porespy_image
from ph_analysis import image_to_pd_pi

TARGET_SEM = './64.jpg'
L_LAMBDA = 0.8


def run(loss_name, large_weight=5.0, diag_weight=0.15, show=True):
    sem = estimate_bimodal_parameters(TARGET_SEM, ratio=RATIO,
                                      threshold=128, min_area=20,
                                      show_result=show)

    if sem['peak_gap'] >= GAP_THRESHOLD:
        sampling_mode = 'separate'
    else:
        sampling_mode = 'mixture'

    print(sampling_mode)
    
    sem_binary, pd_real, pi_real, mesh_range = prepare_real_sem(TARGET_SEM, RATIO)
    bounds = make_bounds(sem)
    x_ini = np.array([
        (bounds[0][0] + bounds[0][1]) / 2,
        (bounds[1][0] + bounds[1][1]) / 2,
        sem['large_loc'],
        sem['large_scale'],
        (bounds[4][0] + bounds[4][1]) / 2,
        (bounds[5][0] + bounds[5][1]) / 2,
    ])

    # Use target PD coordinates to define a data-driven left-side birth cutoff.
    births = np.asarray(pd_real[0].births, dtype=float)
    finite_births = births[np.isfinite(births)]
    large_birth_cut = float(np.quantile(finite_births, 0.25)) if finite_births.size else mesh_range[0]
    diag_width = 0.15 * (mesh_range[1] - mesh_range[0])

    result = optimize_parameters(
        pi_real=pi_real, image_shape=sem_binary.shape,
        mesh_range=mesh_range, dist_weight=sem['dist_weight'],
        x_ini=x_ini, bounds=bounds, L_lambda=L_LAMBDA,
        sampling_mode=sampling_mode, loss_name=loss_name,
        large_birth_cut=large_birth_cut, diag_width=diag_width,
        large_weight=large_weight, diag_weight=diag_weight,
    )

    comparison = compare_optimized_to_sem(
        sem_binary=sem_binary, mesh_range=mesh_range,
        optimized_params=result.x, dist_weight=sem['dist_weight'],
        sampling_mode=sampling_mode, show=show,
    )

    generated = comparison['generated_binary']
    _, pi_gen, _ = image_to_pd_pi(
        generated, mesh_range, bins=PI_BINS,
        sigma=1.0, weight=('atan', 0.0001, 3), superlevel=False
    )

    metrics = {
        'loss_name': loss_name,
        'sampling_mode': sampling_mode,
        'peak_gap': sem['peak_gap'],
        'pi_real_loss': pi_loss(pi_real, pi_gen, mesh_range, loss_name=loss_name,
                                L_lambda=L_LAMBDA, large_birth_cut=large_birth_cut,
                                diag_width=diag_width, large_weight=large_weight,
                                diag_weight=diag_weight),
        'global_hybrid_loss': pi_loss(pi_real, pi_gen, mesh_range, loss_name='global_hybrid', L_lambda=L_LAMBDA),
        'pi_mae': comparison['pi_mae'],
        'pi_rmse': comparison['pi_rmse'],
        'large_birth_cut': large_birth_cut,
        'diag_width': diag_width,
        'result': result,
        'comparison': comparison,
    }
    print('\n=== Loss:', loss_name, '===')
    print('sampling_mode =', sampling_mode)
    print('peak_gap      =', sem['peak_gap'])
    print('optimized x   =', result.x)
    print('objective     =', result.fun)
    print('global hybrid =', metrics['global_hybrid_loss'])
    print('PI MAE        =', comparison['pi_mae'])
    print('PI RMSE       =', comparison['pi_rmse'])
    return metrics
