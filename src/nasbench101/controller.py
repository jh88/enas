import numpy as np
import tensorflow as tf

from src.common_ops import stack_lstm
from src.utils import get_train_ops


class Controller():
    def __init__(
        self,
        num_branches=4, # number of possible operations
        num_cells=7, # number of nodes except the input node
        lstm_size=32,
        lstm_num_layers=2, # number of lstm layers in the controller 
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
        **kwargs
    ):
        self.num_cells = num_cells
        self.num_branches = num_branches

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers 
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.use_critic = use_critic
        self.bl_dec = bl_dec

        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.name = name

        self._create_params()
        (
            self.sample_arc, self.sample_entropy, self.sample_log_prob, _, _
        ) = self._build_sampler()

    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope('lstm'):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope(f'layer_{layer_id}'):
                        w = tf.get_variable(
                            'w', [2 * self.lstm_size, 4 * self.lstm_size]
                        )
                        self.w_lstm.append(w)

            self.g_emb = tf.get_variable('g_emb', [1, self.lstm_size])

            with tf.variable_scope('emb'):
                self.w_emb = tf.get_variable(
                    'w', [self.num_branches, self.lstm_size]
                )

            with tf.variable_scope('softmax'):
                self.w_soft = tf.get_variable(
                    'w', [self.lstm_size, self.num_branches]
                )
                b_init = np.array([0] * self.num_branches, dtype=np.float32)
                self.b_soft = tf.get_variable(
                    'b',
                    [1, self.num_branches],
                    initializer=tf.constant_initializer(b_init)
                )

            with tf.variable_scope('attention'):
                self.w_attn_1 = tf.get_variable(
                    'w_1', [self.lstm_size, self.lstm_size]
                )
                self.w_attn_2 = tf.get_variable(
                    'w_2', [self.lstm_size, self.lstm_size]
                )
                self.v_attn = tf.get_variable('v', [self.lstm_size, 1])

    def _build_sampler(self, prev_c=None, prev_h=None):
        anchors = tf.TensorArray(
            tf.float32, size=self.num_cells+1, clear_after_read=False
        )
        anchors_w_1 = tf.TensorArray(
            tf.float32, size=self.num_cells+1, clear_after_read=False
        )

        arc_seq = tf.TensorArray(tf.int32, size=self.num_cells * 2)

        if prev_c is None or prev_h is None:
            prev_c = [
                tf.zeros([1, self.lstm_size], tf.float32)
                for _ in range(self.lstm_num_layers)
            ]
            prev_h = [
                tf.zeros([1, self.lstm_size], tf.float32)
                for _ in range(self.lstm_num_layers)
            ]

        inputs = self.g_emb

        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h
        anchors = anchors.write(0, tf.zeros_like(next_h[-1]))
        anchors_w_1 = anchors_w_1.write(
            0, tf.matmul(next_h[-1], self.w_attn_1)
        )

        def _condition(layer_id, *args):
            return tf.less(layer_id, self.num_cells + 2)

        def _body(
            layer_id,
            inputs,
            prev_c,
            prev_h,
            anchors,
            anchors_w_1,
            arc_seq,
            entropy,
            log_prob
        ):
            indices = tf.range(0, layer_id, dtype=tf.int32)
            start_id = 2 * (layer_id - 1)
            prev_layers = []

            # index
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            query = anchors_w_1.gather(indices)
            query = tf.reshape(query, [layer_id, self.lstm_size])
            query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
            query = tf.matmul(query, self.v_attn)
            logits = tf.reshape(query, [1, layer_id])

            if self.temperature:
                logits /= self.temperature
            if self.tanh_constant:
                logits = self.tanh_constant * tf.tanh(logits)

            index = tf.multinomial(logits, 1)
            index = tf.to_int32(index)
            index = tf.reshape(index, [1])
            arc_seq = arc_seq.write(start_id, index)

            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=index
            )
            log_prob += curr_log_prob
            curr_ent = tf.stop_gradient(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.nn.softmax(logits)
                )
            )
            entropy += curr_ent

            prev_layers.append(anchors.read(tf.reduce_sum(index)))
            inputs = prev_layers[-1]

            # op
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c,next_h
            logits = tf.matmul(next_h[-1], self.w_soft) + self.b_soft

            if self.temperature:
                logits /= self.temperature
            if self.tanh_constant:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * tf.tanh(logits)

            op_id = tf.multinomial(logits, 1)
            op_id = tf.to_int32(op_id)
            op_id = tf.reshape(op_id, [1])
            arc_seq = arc_seq.write(start_id + 1, op_id)

            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=op_id
            )
            log_prob += curr_log_prob
            curr_ent = tf.stop_gradient(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.nn.softmax(logits)
                )
            )
            entropy += curr_ent
            inputs = tf.nn.embedding_lookup(self.w_emb, op_id)

            return (
                layer_id + 1,
                inputs,
                next_c,
                next_h,
                anchors,
                anchors_w_1,
                arc_seq,
                entropy,
                log_prob
            )

        loop_vars = [
            tf.constant(1, dtype=tf.int32, name='layer_id'),
            inputs,
            prev_c,
            prev_h,
            anchors,
            anchors_w_1,
            arc_seq,
            tf.constant([0.0], dtype=tf.float32, name='entropy'),
            tf.constant([0.0], dtype=tf.float32, name='log_prob')
        ]

        loop_outputs = tf.while_loop(
            _condition, _body, loop_vars, parallel_iterations=1
        )

        arc_seq = loop_outputs[-3].stack()
        arc_seq = tf.reshape(arc_seq, [-1])
        entropy = tf.reduce_sum(loop_outputs[-2])
        log_prob = tf.reduce_sum(loop_outputs[-1])

        last_c = loop_outputs[2]
        last_h = loop_outputs[3]

        return arc_seq, entropy, log_prob, last_c, last_h

    def build_trainer(self, nasbench):
        self.reward = tf.to_float(nasbench.get_acc())
        self.reward = self.valid_acc

        if self.entropy_weight:
            self.reward += self.entropy_weight * self.sample_entropy

        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        self.baseline = tf.Variable(
            0.0, dtype=tf.float32, trainable=False, name='baseline'
        )
        baseline_update = tf.assign_sub(
            self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward)
        )

        with tf.control_dependencies([baseline_update]):
            self.reward = tf.identity(self.reward)

        self.loss = self.sample_log_prob * (self.reward - self.baseline)
        self.train_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='train_step'
        )

        tf_variables = [
            var for var in tf.trainable_variables()
            if var.name.startswith(self.name)
        ]

        for var in tf_variables:
            print(var)

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            self.loss,
            tf_variables,
            self.train_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas
        )

        self.skip_rate = tf.constant(0.0, dtype=tf.float32)
