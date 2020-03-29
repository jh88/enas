import tensorflow as tf

from src.nasbench101.controller import Controller


def get_controller_ops():
    controller = Controller(
        num_branches=4,
        num_cells=7,
        lstm_size=32,
        lstm_num_layers=2,
        tanh_constant=None,
        op_tanh_reduce=1.0,
        temperature=None,
        lr_init=1e-3,
        lr_dec_start=0,
        lr_dec_every=100,
        lr_dec_rate=0.9,
        l2_reg=0,
        entropy_weight=None,
        clip_mode=None,
        grad_bound=None,
        use_critic=False,
        bl_dec=0.999,
        optim_algo="adam",
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None,
        name="controller",   
    )

    controller_model.build_trainer(child_model)

    controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "valid_acc": controller_model.valid_acc,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
        "skip_rate": controller_model.skip_rate
    }

    return controller_ops
