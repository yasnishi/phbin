from config import PI_BINS, PI_SIGMA, PI_WEIGHT, PI_SUPERLEVEL, MESH_MARGIN
import phbin
from ph_analysis import image_to_pd, pd_to_pi, determine_mesh_range_from_pd


def prepare_real_sem(target_sem, ratio):
    img, sem_bin = phbin.make_input_imgobj(target_sem)
    re_img = phbin.resize_img(img, ratio)
    particle = phbin.binarization(re_img)
    pd = image_to_pd(particle)
    mesh = determine_mesh_range_from_pd(pd, MESH_MARGIN)
    pi = pd_to_pi(pd, mesh, PI_BINS, PI_SIGMA, PI_WEIGHT, PI_SUPERLEVEL)
    return particle, pd, pi, mesh
