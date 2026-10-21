from loss_main_common import run

if __name__ == '__main__':
    run('diag_suppressed', large_weight=5.0, diag_weight=0.10, show=False)
