from nasbench import api
import tensorflow as tf

from src.nasbench101.child import Child
from src.nasbench101.controller import Controller


def get_ops(nasbench):
    controller_model = Controller(
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

    child_model = Child(nasbench)

    child_model.connect_controller(controller_model)
    controller_model.build_trainer(child_model)

    controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "reward": controller_model.reward,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
        "skip_rate": controller_model.skip_rate
    }

    child_ops = {
        'train_acc': child_model.train_acc,
        'valid_acc': child_model.valid_acc,
        'test_acc': child_model.test_acc
    }

    ops = {
        'controller': controller_ops,
        'child': child_opsl
    }

    return ops


def train(nasbench, epoch=2):
    g = tf.Graph()
    with g.as_default():
        ops = get_ops(nasbench)

        child_ops = ops['child']
        controller_ops = ops['controller']

        with tf.train.SingularMonitoredSession() as sess:
            for i in range(epoch):
                run_ops = [
                    controller_ops['sample_arc'],
                    controller_ops['reward']
                ]

                arc, acc = sess.run(run_ops)

                print('epoch: {}\narc: {}\nacc: {}'.format(i, arc, acc))


def main():
    nasbench = api.NASBench('nasbench_only108.tfrecord')
    train(nasbench, epoch=1)
